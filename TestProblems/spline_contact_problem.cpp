#include "spline_contact_problem.hpp"

#include "../continuationsolvers/utilities.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace
{

std::array<HYPRE_BigInt, 2> MakeOffsets(int local_size, MPI_Comm comm)
{
   int rank = 0;
   int number_of_ranks = 1;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &number_of_ranks);
   std::vector<int> local_sizes(static_cast<std::size_t>(number_of_ranks), 0);
   MPI_Allgather(&local_size, 1, MPI_INT, local_sizes.data(), 1,
                 MPI_INT, comm);

   HYPRE_BigInt first = 0;
   for (int r = 0; r < rank; ++r)
   {
      first += static_cast<HYPRE_BigInt>(
         local_sizes[static_cast<std::size_t>(r)]);
   }
   return {first, first + static_cast<HYPRE_BigInt>(local_size)};
}

} // namespace

ElasticSplineContactProblem::ElasticSplineContactProblem(
   mfem::ParFiniteElementSpace &displacement_space,
   mfem::ParBilinearForm &elasticity,
   mfem::ParLinearForm &load,
   const mfem::Array<int> &essential_true_dofs,
   int contact_boundary_attribute,
   const spline::QuinticSmoothingSpline1D &surface,
   mfem::real_t surface_x_min,
   mfem::real_t surface_x_max)
   : displacement_space_(displacement_space),
     mesh_(*displacement_space.GetParMesh()),
     surface_(surface),
     comm_(mesh_.GetComm()),
     contact_boundary_attribute_(contact_boundary_attribute),
     surface_x_min_(surface_x_min),
     surface_x_max_(surface_x_max),
     full_local_true_dof_count_(displacement_space.GetTrueVSize())
{
   MFEM_VERIFY(mesh_.Dimension() == 2 && mesh_.SpaceDimension() == 2,
               "ElasticSplineContactProblem requires a two-dimensional mesh.");
   MFEM_VERIFY(displacement_space_.GetVDim() == 2,
               "The displacement finite element space must be vector valued "
               "with two components.");
   MFEM_VERIFY(displacement_space_.GetMaxElementOrder() == 1,
               "The first implementation requires linear H1 displacement "
               "elements.");
   MFEM_VERIFY(contact_boundary_attribute_ > 0,
               "A positive contact boundary attribute is required.");
   MFEM_VERIFY(std::isfinite(surface_x_min_) &&
                  std::isfinite(surface_x_max_) &&
                  surface_x_min_ < surface_x_max_,
               "The spline domain must be a finite, nonempty interval.");

   std::vector<HYPRE_Int> free_mask(
      static_cast<std::size_t>(full_local_true_dof_count_), 1);
   for (int i = 0; i < essential_true_dofs.Size(); ++i)
   {
      const int true_dof = essential_true_dofs[i];
      MFEM_VERIFY(true_dof >= 0 && true_dof < full_local_true_dof_count_,
                  "An essential true DOF is outside the local range.");
      free_mask[static_cast<std::size_t>(true_dof)] = 0;
   }

   full_to_reduced_local_.assign(
      static_cast<std::size_t>(full_local_true_dof_count_), -1);
   int local_free_dof_count = 0;
   for (int full_dof = 0; full_dof < full_local_true_dof_count_; ++full_dof)
   {
      if (free_mask[static_cast<std::size_t>(full_dof)] != 0)
      {
         full_to_reduced_local_[static_cast<std::size_t>(full_dof)] =
            local_free_dof_count++;
      }
   }

   std::array<HYPRE_BigInt, 2> reduced_offsets =
      MakeOffsets(local_free_dof_count, comm_);
   HYPRE_BigInt *full_offsets = displacement_space_.GetTrueDofOffsets();
   restriction_.reset(GenerateProjector(reduced_offsets.data(), full_offsets,
                                        free_mask.data()));
   prolongation_.reset(restriction_->Transpose());

   BuildContactNodes(reduced_offsets[0]);
   const int local_constraint_count =
      static_cast<int>(contact_nodes_.size());
   std::array<HYPRE_BigInt, 2> constraint_offsets =
      MakeOffsets(local_constraint_count, comm_);

   Init(reduced_offsets.data(), constraint_offsets.data());
   MFEM_VERIFY(GetDimMGlb() > 0,
               "No nodes were found on the requested contact boundary.");

   std::unique_ptr<mfem::HypreParMatrix> full_stiffness(
      elasticity.ParallelAssemble());
   MFEM_VERIFY(full_stiffness != nullptr,
               "Unable to assemble the parallel elasticity matrix.");
   stiffness_.reset(mfem::RAP(full_stiffness.get(), prolongation_.get()));

   std::unique_ptr<mfem::HypreParVector> full_load(load.ParallelAssemble());
   MFEM_VERIFY(full_load != nullptr,
               "Unable to assemble the parallel load vector.");
   reduced_load_.SetSize(local_free_dof_count);
   restriction_->Mult(*full_load, reduced_load_);

   full_displacement_.SetSize(full_local_true_dof_count_);
   full_displacement_ = 0.0;
   stiffness_times_displacement_.SetSize(local_free_dof_count);
   stiffness_times_displacement_ = 0.0;
}

void ElasticSplineContactProblem::BuildContactNodes(
   HYPRE_BigInt first_reduced_global_dof)
{
   std::set<int> inserted_owned_y_true_dofs;
   std::vector<ContactNode> nodes;
   mfem::Array<int> vertices;
   mfem::Array<int> scalar_dofs;

   for (int boundary_element = 0;
        boundary_element < mesh_.GetNBE(); ++boundary_element)
   {
      if (mesh_.GetBdrAttribute(boundary_element) !=
          contact_boundary_attribute_)
      {
         continue;
      }

      mesh_.GetBdrElementVertices(boundary_element, vertices);
      for (int j = 0; j < vertices.Size(); ++j)
      {
         const int vertex = vertices[j];
         displacement_space_.GetVertexDofs(vertex, scalar_dofs);
         MFEM_VERIFY(scalar_dofs.Size() == 1,
                     "A contact vertex must have exactly one scalar P1 DOF.");

         const int scalar_dof = scalar_dofs[0];
         const int x_vector_dof =
            displacement_space_.DofToVDof(scalar_dof, 0);
         const int y_vector_dof =
            displacement_space_.DofToVDof(scalar_dof, 1);
         const int x_true_dof =
            displacement_space_.GetLocalTDofNumber(x_vector_dof);
         const int y_true_dof =
            displacement_space_.GetLocalTDofNumber(y_vector_dof);

         // A shared boundary node is represented on several ranks, but only
         // its true-DOF owner creates the constraint row.
         if (y_true_dof < 0 ||
             !inserted_owned_y_true_dofs.insert(y_true_dof).second)
         {
            continue;
         }
         MFEM_VERIFY(x_true_dof >= 0,
                     "The two components of an owned contact node must be "
                     "owned by the same MPI rank.");

         ContactNode node;
         node.local_vertex = vertex;
         node.x = mesh_.GetVertex(vertex)[0];
         node.y = mesh_.GetVertex(vertex)[1];
         node.full_x_true_dof = x_true_dof;
         node.full_y_true_dof = y_true_dof;
         node.reduced_x_true_dof =
            full_to_reduced_local_[static_cast<std::size_t>(x_true_dof)];
         node.reduced_y_true_dof =
            full_to_reduced_local_[static_cast<std::size_t>(y_true_dof)];
         if (node.reduced_x_true_dof >= 0)
         {
            node.reduced_x_global_true_dof =
               first_reduced_global_dof + node.reduced_x_true_dof;
         }
         if (node.reduced_y_true_dof >= 0)
         {
            node.reduced_y_global_true_dof =
               first_reduced_global_dof + node.reduced_y_true_dof;
         }
         nodes.push_back(node);
      }
   }

   std::sort(nodes.begin(), nodes.end(),
             [](const ContactNode &a, const ContactNode &b)
             {
                if (a.x != b.x) { return a.x < b.x; }
                return a.y < b.y;
             });
   contact_nodes_ = std::move(nodes);
}

void ElasticSplineContactProblem::ProlongateToFullDofs(
   const mfem::Vector &d, mfem::Vector &full_d) const
{
   MFEM_VERIFY(d.Size() == GetDimU(),
               "The reduced displacement has an incompatible local size.");
   full_d.SetSize(full_local_true_dof_count_);
   prolongation_->Mult(d, full_d);
}

mfem::Vector ElasticSplineContactProblem::DisplacedAbscissae(
   const mfem::Vector &d)
{
   ProlongateToFullDofs(d, full_displacement_);
   mfem::Vector points(LocalContactNodeCount());
   for (int i = 0; i < LocalContactNodeCount(); ++i)
   {
      const ContactNode &node = contact_nodes_[static_cast<std::size_t>(i)];
      points[i] = node.x + full_displacement_[node.full_x_true_dof];
   }
   return points;
}

bool ElasticSplineContactProblem::PointsAreInSurfaceDomain(
   const mfem::Vector &points) const
{
   int local_points_are_valid = 1;
   for (int i = 0; i < points.Size(); ++i)
   {
      if (!std::isfinite(points[i]) || points[i] < surface_x_min_ ||
          points[i] > surface_x_max_)
      {
         local_points_are_valid = 0;
      }
   }
   int global_points_are_valid = 0;
   MPI_Allreduce(&local_points_are_valid, &global_points_are_valid, 1,
                 MPI_INT, MPI_MIN, comm_);
   return global_points_are_valid != 0;
}

double ElasticSplineContactProblem::E(const mfem::Vector &d, int &eval_err)
{
   MFEM_VERIFY(d.Size() == GetDimU(),
               "The displacement has an incompatible local size.");
   stiffness_->Mult(d, stiffness_times_displacement_);
   const mfem::real_t elastic_energy =
      0.5 * mfem::InnerProduct(comm_, d, stiffness_times_displacement_);
   const mfem::real_t load_potential =
      mfem::InnerProduct(comm_, reduced_load_, d);
   eval_err = 0;
   return elastic_energy - load_potential;
}

void ElasticSplineContactProblem::DdE(const mfem::Vector &d,
                                      mfem::Vector &gradient)
{
   MFEM_VERIFY(d.Size() == GetDimU(),
               "The displacement has an incompatible local size.");
   gradient.SetSize(GetDimU());
   stiffness_->Mult(d, gradient);
   gradient.Add(-1.0, reduced_load_);
}

mfem::Operator *ElasticSplineContactProblem::DddE(
   const mfem::Vector & /*d*/)
{
   return stiffness_.get();
}

void ElasticSplineContactProblem::g(const mfem::Vector &d,
                                    mfem::Vector &gap,
                                    int &eval_err)
{
   const mfem::Vector points = DisplacedAbscissae(d);
   gap.SetSize(LocalContactNodeCount());
   if (!PointsAreInSurfaceDomain(points))
   {
      gap = 0.0;
      eval_err = 1;
      return;
   }

   const mfem::Vector surface_values = surface_.Evaluate(points, 0);
   for (int i = 0; i < LocalContactNodeCount(); ++i)
   {
      const ContactNode &node = contact_nodes_[static_cast<std::size_t>(i)];
      gap[i] = node.y + full_displacement_[node.full_y_true_dof] -
               surface_values[i];
   }
   eval_err = 0;
}

mfem::Operator *ElasticSplineContactProblem::Ddg(const mfem::Vector &d)
{
   const mfem::Vector points = DisplacedAbscissae(d);
   MFEM_VERIFY(PointsAreInSurfaceDomain(points),
               "Cannot differentiate the gap outside the spline domain.");
   const mfem::Vector surface_first_derivative = surface_.Evaluate(points, 1);

   const int number_of_rows = LocalContactNodeCount();
   std::vector<int> I(static_cast<std::size_t>(number_of_rows + 1), 0);
   std::vector<HYPRE_BigInt> J;
   std::vector<mfem::real_t> data;
   J.reserve(static_cast<std::size_t>(2 * number_of_rows));
   data.reserve(static_cast<std::size_t>(2 * number_of_rows));

   for (int row = 0; row < number_of_rows; ++row)
   {
      const ContactNode &node = contact_nodes_[static_cast<std::size_t>(row)];
      if (node.reduced_x_true_dof >= 0)
      {
         J.push_back(node.reduced_x_global_true_dof);
         data.push_back(-surface_first_derivative[row]);
      }
      if (node.reduced_y_true_dof >= 0)
      {
         J.push_back(node.reduced_y_global_true_dof);
         data.push_back(1.0);
      }
      I[static_cast<std::size_t>(row + 1)] = static_cast<int>(J.size());
   }

   gap_jacobian_ = std::make_unique<mfem::HypreParMatrix>(
      comm_, number_of_rows, static_cast<HYPRE_BigInt>(GetDimMGlb()),
      static_cast<HYPRE_BigInt>(GetDimUGlb()), I.data(), J.data(),
      data.data(), GetDofOffsetsM(), GetDofOffsetsU());
   return gap_jacobian_.get();
}

mfem::Operator *ElasticSplineContactProblem::Dddgl(
   const mfem::Vector &d, const mfem::Vector &multiplier)
{
   MFEM_VERIFY(multiplier.Size() == LocalContactNodeCount(),
               "The contact multiplier has an incompatible local size.");
   const mfem::Vector points = DisplacedAbscissae(d);
   MFEM_VERIFY(PointsAreInSurfaceDomain(points),
               "Cannot differentiate the gap outside the spline domain.");
   const mfem::Vector surface_second_derivative = surface_.Evaluate(points, 2);

   mfem::Vector diagonal(GetDimU());
   diagonal = 0.0;
   for (int i = 0; i < LocalContactNodeCount(); ++i)
   {
      const ContactNode &node = contact_nodes_[static_cast<std::size_t>(i)];
      if (node.reduced_x_true_dof >= 0)
      {
         diagonal[node.reduced_x_true_dof] +=
            -multiplier[i] * surface_second_derivative[i];
      }
   }
   gap_hessian_.reset(
      GenerateHypreParMatrixFromDiagonal(GetDofOffsetsU(), diagonal));
   return gap_hessian_.get();
}

void ElasticSplineContactProblem::SetPotentialContactValues(
   const mfem::Vector &contact_values,
   mfem::ParGridFunction &field) const
{
   MFEM_VERIFY(contact_values.Size() == LocalContactNodeCount(),
               "Expected one value per locally owned contact node.");
   mfem::ParFiniteElementSpace *field_space = field.ParFESpace();
   MFEM_VERIFY(field_space != nullptr &&
                  field_space->GetParMesh() == &mesh_ &&
                  field_space->GetVDim() == 1 &&
                  field_space->GetMaxElementOrder() == 1,
               "The contact output field must use a scalar P1 space on the "
               "elastic mesh.");

   mfem::Vector true_values(field_space->GetTrueVSize());
   true_values = 0.0;
   mfem::Array<int> vertex_dofs;
   for (int i = 0; i < LocalContactNodeCount(); ++i)
   {
      const ContactNode &node = contact_nodes_[static_cast<std::size_t>(i)];
      field_space->GetVertexDofs(node.local_vertex, vertex_dofs);
      MFEM_VERIFY(vertex_dofs.Size() == 1,
                  "A scalar P1 vertex must have one local DOF.");
      const int true_dof =
         field_space->GetLocalTDofNumber(vertex_dofs[0]);
      MFEM_VERIFY(true_dof >= 0,
                  "Contact-node ownership differs between the scalar and "
                  "vector P1 spaces.");
      true_values[true_dof] = contact_values[i];
   }
   field.SetFromTrueDofs(true_values);
}
