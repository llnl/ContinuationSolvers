#include "parametric_spline_contact_problem.hpp"

#include "../../continuationsolvers/utilities.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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

ParametricElasticSplineContactProblem::
ParametricElasticSplineContactProblem(
   mfem::ParFiniteElementSpace &displacement_space,
   mfem::ParBilinearForm &elasticity,
   mfem::ParLinearForm &load,
   const mfem::Array<int> &essential_true_dofs,
   int contact_boundary_attribute,
   spline::QuinticSmoothingSpline1D &surface,
   mfem::ParFiniteElementSpace &surface_data_space,
   const mfem::Vector &initial_surface_data,
   mfem::real_t surface_x_min,
   mfem::real_t surface_x_max,
   const spline::QuinticSmoothingSolveOptions &surface_solve_options,
   spline::ContactGapType gap_type,
   const spline::ClosestPointOptions &closest_point_options)
   : displacement_space_(displacement_space),
     mesh_(*displacement_space.GetParMesh()),
     surface_(surface),
     closest_point_projector_(surface, surface_x_min, surface_x_max,
                              closest_point_options),
     surface_data_space_(surface_data_space),
     comm_(mesh_.GetComm()),
     contact_boundary_attribute_(contact_boundary_attribute),
     gap_type_(gap_type),
     surface_x_min_(surface_x_min),
     surface_x_max_(surface_x_max),
     full_local_true_dof_count_(displacement_space.GetTrueVSize()),
     surface_solve_options_(surface_solve_options)
{
   MFEM_VERIFY(mesh_.Dimension() == 2 && mesh_.SpaceDimension() == 2,
               "ParametricElasticSplineContactProblem requires a "
               "two-dimensional mesh.");
   MFEM_VERIFY(displacement_space_.GetVDim() == 2,
               "The displacement finite element space must have two "
               "components.");
   MFEM_VERIFY(displacement_space_.GetMaxElementOrder() == 1,
               "The initial implementation requires linear H1 displacement "
               "elements.");
   MFEM_VERIFY(surface_data_space_.GetVDim() == 1 &&
                  surface_data_space_.GetMaxElementOrder() == 1,
               "The spline data must use a scalar linear H1 space.");
   MFEM_VERIFY(&surface_.DataSpace() == &surface_data_space_,
               "The supplied spline and spline-data space must refer to the "
               "same parallel finite element space.");
   MFEM_VERIFY(surface_data_space_.GetParMesh() != nullptr,
               "The spline-data space must be defined on a parallel mesh.");
   int communicator_relation = MPI_UNEQUAL;
   MPI_Comm_compare(comm_, surface_data_space_.GetParMesh()->GetComm(),
                    &communicator_relation);
   MFEM_VERIFY(communicator_relation == MPI_IDENT ||
                  communicator_relation == MPI_CONGRUENT,
               "The elastic and spline meshes must use congruent MPI "
               "communicators.");
   MFEM_VERIFY(initial_surface_data.Size() ==
                  surface_data_space_.GetTrueVSize(),
               "Initial spline data must use the locally owned true-DOF "
               "ordering.");
   MFEM_VERIFY(contact_boundary_attribute_ > 0,
               "A positive contact boundary attribute is required.");
   MFEM_VERIFY(std::isfinite(surface_x_min_) &&
                  std::isfinite(surface_x_max_) &&
                  surface_x_min_ < surface_x_max_,
               "The spline domain must be a finite, nonempty interval.");
   MFEM_VERIFY(surface_solve_options_.direct_solver,
               "Data sensitivities require a retained direct spline "
               "factorization.");

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
   const int local_constraint_count = LocalContactNodeCount();
   std::array<HYPRE_BigInt, 2> constraint_offsets =
      MakeOffsets(local_constraint_count, comm_);

   Init(reduced_offsets.data(), constraint_offsets.data());
   InitTheta(initial_surface_data);
   MFEM_VERIFY(GetDimMGlb() > 0,
               "No nodes were found on the requested contact boundary.");

   HYPRE_BigInt *surface_offsets = surface_data_space_.GetTrueDofOffsets();
   MFEM_VERIFY(GetDofOffsetsTheta()[0] == surface_offsets[0] &&
                  GetDofOffsetsTheta()[1] == surface_offsets[1],
               "The design-vector distribution must match the spline-data "
               "finite element space.");

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

   zero_displacement_data_hessian_.reset(
      GenerateNullHypreParMatrix(GetDofOffsetsU(), GetDofOffsetsTheta()));
   BuildContactXMap();

   full_displacement_.SetSize(full_local_true_dof_count_);
   full_displacement_ = 0.0;
   stiffness_times_displacement_.SetSize(local_free_dof_count);
   stiffness_times_displacement_ = 0.0;

   UpdateSurface(initial_surface_data);
}

void ParametricElasticSplineContactProblem::BuildContactNodes(
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

         // Shared nodes appear on multiple ranks.  Only the true-DOF owner
         // creates the corresponding global constraint row.
         if (y_true_dof < 0 ||
             !inserted_owned_y_true_dofs.insert(y_true_dof).second)
         {
            continue;
         }
         MFEM_VERIFY(x_true_dof >= 0,
                     "Both components of an owned contact node must be "
                     "owned by the same rank.");

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

void ParametricElasticSplineContactProblem::BuildContactXMap()
{
   const int number_of_rows = LocalContactNodeCount();
   std::vector<int> I(static_cast<std::size_t>(number_of_rows + 1), 0);
   std::vector<HYPRE_BigInt> J;
   std::vector<mfem::real_t> data;
   J.reserve(static_cast<std::size_t>(number_of_rows));
   data.reserve(static_cast<std::size_t>(number_of_rows));

   for (int row = 0; row < number_of_rows; ++row)
   {
      const ContactNode &node = contact_nodes_[static_cast<std::size_t>(row)];
      if (node.reduced_x_true_dof >= 0)
      {
         J.push_back(node.reduced_x_global_true_dof);
         data.push_back(1.0);
      }
      I[static_cast<std::size_t>(row + 1)] = static_cast<int>(J.size());
   }

   contact_x_map_ = std::make_unique<mfem::HypreParMatrix>(
      comm_, number_of_rows, static_cast<HYPRE_BigInt>(GetDimMGlb()),
      static_cast<HYPRE_BigInt>(GetDimUGlb()), I.data(), J.data(),
      data.data(), GetDofOffsetsM(), GetDofOffsetsU());
   contact_x_map_transpose_.reset(contact_x_map_->Transpose());
}

void ParametricElasticSplineContactProblem::UpdateSurface(
   const mfem::Vector &theta)
{
   MFEM_VERIFY(theta.Size() == GetDimTheta(),
               "The surface design vector has an incompatible local size.");

   int local_data_changed =
      (!surface_data_are_current_ || cached_theta_.Size() != theta.Size())
         ? 1 : 0;
   if (local_data_changed == 0)
   {
      for (int i = 0; i < theta.Size(); ++i)
      {
         if (cached_theta_[i] != theta[i])
         {
            local_data_changed = 1;
            break;
         }
      }
   }

   int global_data_changed = 0;
   MPI_Allreduce(&local_data_changed, &global_data_changed, 1, MPI_INT,
                 MPI_MAX, comm_);
   if (global_data_changed == 0) { return; }

   surface_.SetDataValues(theta);
   const spline::QuinticSmoothingSolveResult result =
      surface_.Solve(surface_solve_options_);
   MFEM_VERIFY(result.converged,
               "The quintic smoothing spline solve failed after updating "
               "the design data.");
   cached_theta_.SetSize(theta.Size());
   cached_theta_ = theta;
   surface_data_are_current_ = true;
}

void ParametricElasticSplineContactProblem::SetSurfaceData(
   const mfem::Vector &theta)
{
   MFEM_VERIFY(theta.Size() == GetDimTheta(),
               "The surface design vector has an incompatible local size.");
   theta_default.SetSize(theta.Size());
   theta_default = theta;
   UpdateSurface(theta);
}

void ParametricElasticSplineContactProblem::ProlongateToFullDofs(
   const mfem::Vector &d, mfem::Vector &full_d) const
{
   MFEM_VERIFY(d.Size() == GetDimU(),
               "The reduced displacement has an incompatible local size.");
   full_d.SetSize(full_local_true_dof_count_);
   prolongation_->Mult(d, full_d);
}

mfem::Vector ParametricElasticSplineContactProblem::DisplacedAbscissae(
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

void ParametricElasticSplineContactProblem::DisplacedContactCoordinates(
   const mfem::Vector &d, mfem::Vector &point_x, mfem::Vector &point_y)
{
   ProlongateToFullDofs(d, full_displacement_);
   point_x.SetSize(LocalContactNodeCount());
   point_y.SetSize(LocalContactNodeCount());
   for (int i = 0; i < LocalContactNodeCount(); ++i)
   {
      const ContactNode &node = contact_nodes_[static_cast<std::size_t>(i)];
      point_x[i] = node.x + full_displacement_[node.full_x_true_dof];
      point_y[i] = node.y + full_displacement_[node.full_y_true_dof];
   }
}

bool ParametricElasticSplineContactProblem::EvaluateClosestPoints(
   const mfem::Vector &d,
   spline::ClosestPointBatchResult &closest_points)
{
   mfem::Vector point_x, point_y;
   DisplacedContactCoordinates(d, point_x, point_y);
   return closest_point_projector_.Project(point_x, point_y,
                                           closest_points);
}

std::unique_ptr<mfem::HypreParMatrix>
ParametricElasticSplineContactProblem::MakeContactDirectionMap(
   const mfem::Vector &x_weights,
   const mfem::Vector &y_weights) const
{
   MFEM_VERIFY(x_weights.Size() == LocalContactNodeCount() &&
                  y_weights.Size() == LocalContactNodeCount(),
               "A contact direction map requires one pair of weights per "
               "locally owned contact node.");

   const int number_of_rows = LocalContactNodeCount();
   std::vector<int> I(static_cast<std::size_t>(number_of_rows + 1), 0);
   std::vector<HYPRE_BigInt> J;
   std::vector<mfem::real_t> data;
   J.reserve(static_cast<std::size_t>(2 * number_of_rows));
   data.reserve(static_cast<std::size_t>(2 * number_of_rows));
   for (int row = 0; row < number_of_rows; ++row)
   {
      const ContactNode &node = contact_nodes_[static_cast<std::size_t>(row)];
      if (node.reduced_x_true_dof >= 0 && x_weights[row] != 0.0)
      {
         J.push_back(node.reduced_x_global_true_dof);
         data.push_back(x_weights[row]);
      }
      if (node.reduced_y_true_dof >= 0 && y_weights[row] != 0.0)
      {
         J.push_back(node.reduced_y_global_true_dof);
         data.push_back(y_weights[row]);
      }
      I[static_cast<std::size_t>(row + 1)] = static_cast<int>(J.size());
   }

   return std::make_unique<mfem::HypreParMatrix>(
      comm_, number_of_rows, static_cast<HYPRE_BigInt>(GetDimMGlb()),
      static_cast<HYPRE_BigInt>(GetDimUGlb()), I.data(), J.data(),
      data.data(), GetDofOffsetsM(), GetDofOffsetsU());
}

bool ParametricElasticSplineContactProblem::PointsAreInSurfaceDomain(
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

double ParametricElasticSplineContactProblem::E(
   const mfem::Vector &d, const mfem::Vector & /*theta*/, int &eval_err)
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

void ParametricElasticSplineContactProblem::DdE(
   const mfem::Vector &d, const mfem::Vector & /*theta*/,
   mfem::Vector &gradient)
{
   MFEM_VERIFY(d.Size() == GetDimU(),
               "The displacement has an incompatible local size.");
   gradient.SetSize(GetDimU());
   stiffness_->Mult(d, gradient);
   gradient.Add(-1.0, reduced_load_);
}

mfem::Operator *ParametricElasticSplineContactProblem::DddE(
   const mfem::Vector & /*d*/, const mfem::Vector & /*theta*/)
{
   return stiffness_.get();
}

mfem::Operator *ParametricElasticSplineContactProblem::DdthE(
   const mfem::Vector & /*d*/, const mfem::Vector & /*theta*/)
{
   return zero_displacement_data_hessian_.get();
}

void ParametricElasticSplineContactProblem::g(
   const mfem::Vector &d, const mfem::Vector &theta,
   mfem::Vector &gap, int &eval_err)
{
   UpdateSurface(theta);
   if (gap_type_ == spline::ContactGapType::ClosestPointNormal)
   {
      spline::ClosestPointBatchResult closest_points;
      const bool success = EvaluateClosestPoints(d, closest_points);
      gap.SetSize(LocalContactNodeCount());
      if (!success)
      {
         gap = 0.0;
         eval_err = 1;
         return;
      }
      gap = closest_points.gap;
      eval_err = 0;
      return;
   }

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

mfem::Operator *ParametricElasticSplineContactProblem::Ddg(
   const mfem::Vector &d, const mfem::Vector &theta)
{
   UpdateSurface(theta);
   if (gap_type_ == spline::ContactGapType::ClosestPointNormal)
   {
      spline::ClosestPointBatchResult closest_points;
      MFEM_VERIFY(EvaluateClosestPoints(d, closest_points),
                  "Cannot differentiate an invalid closest-point gap.");
      gap_displacement_jacobian_ = MakeContactDirectionMap(
         closest_points.normal_x, closest_points.normal_y);
      return gap_displacement_jacobian_.get();
   }

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

   gap_displacement_jacobian_ =
      std::make_unique<mfem::HypreParMatrix>(
         comm_, number_of_rows, static_cast<HYPRE_BigInt>(GetDimMGlb()),
         static_cast<HYPRE_BigInt>(GetDimUGlb()), I.data(), J.data(),
         data.data(), GetDofOffsetsM(), GetDofOffsetsU());
   return gap_displacement_jacobian_.get();
}

mfem::Operator *ParametricElasticSplineContactProblem::Dthg(
   const mfem::Vector &d, const mfem::Vector &theta)
{
   UpdateSurface(theta);
   if (gap_type_ == spline::ContactGapType::ClosestPointNormal)
   {
      spline::ClosestPointBatchResult closest_points;
      MFEM_VERIFY(EvaluateClosestPoints(d, closest_points),
                  "Cannot differentiate an invalid closest-point gap.");
      gap_data_jacobian_ = std::make_unique<mfem::HypreParMatrix>(
         surface_.EvaluateDataJacobian(closest_points.parameter, 0));
      mfem::Vector row_scale(LocalContactNodeCount());
      for (int i = 0; i < LocalContactNodeCount(); ++i)
      {
         row_scale[i] = -closest_points.normal_y[i];
      }
      gap_data_jacobian_->ScaleRows(row_scale);
      return gap_data_jacobian_.get();
   }

   const mfem::Vector points = DisplacedAbscissae(d);
   MFEM_VERIFY(PointsAreInSurfaceDomain(points),
               "Cannot differentiate the gap outside the spline domain.");
   gap_data_jacobian_ = std::make_unique<mfem::HypreParMatrix>(
      surface_.EvaluateDataJacobian(points, 0));
   mfem::Vector scale(gap_data_jacobian_->Height());
   scale = -1.0;
   gap_data_jacobian_->ScaleRows(scale);
   return gap_data_jacobian_.get();
}

mfem::Operator *ParametricElasticSplineContactProblem::Dddgl(
   const mfem::Vector &d, const mfem::Vector &multiplier,
   const mfem::Vector &theta)
{
   MFEM_VERIFY(multiplier.Size() == LocalContactNodeCount(),
               "The contact multiplier has an incompatible local size.");
   UpdateSurface(theta);
   if (gap_type_ == spline::ContactGapType::ClosestPointNormal)
   {
      spline::ClosestPointBatchResult closest_points;
      MFEM_VERIFY(EvaluateClosestPoints(d, closest_points),
                  "Cannot differentiate an invalid closest-point gap.");

      struct Entry
      {
         HYPRE_BigInt column;
         mfem::real_t value;
      };
      std::vector<std::vector<Entry>> rows(
         static_cast<std::size_t>(GetDimU()));
      for (int i = 0; i < LocalContactNodeCount(); ++i)
      {
         const ContactNode &node =
            contact_nodes_[static_cast<std::size_t>(i)];
         const mfem::real_t scale =
            multiplier[i] * closest_points.curvature_factor[i];
         const mfem::real_t tx = closest_points.tangent_x[i];
         const mfem::real_t ty = closest_points.tangent_y[i];
         if (node.reduced_x_true_dof >= 0)
         {
            auto &row = rows[static_cast<std::size_t>(
               node.reduced_x_true_dof)];
            row.push_back({node.reduced_x_global_true_dof,
                           scale * tx * tx});
            if (node.reduced_y_true_dof >= 0)
            {
               row.push_back({node.reduced_y_global_true_dof,
                              scale * tx * ty});
            }
         }
         if (node.reduced_y_true_dof >= 0)
         {
            auto &row = rows[static_cast<std::size_t>(
               node.reduced_y_true_dof)];
            if (node.reduced_x_true_dof >= 0)
            {
               row.push_back({node.reduced_x_global_true_dof,
                              scale * tx * ty});
            }
            row.push_back({node.reduced_y_global_true_dof,
                           scale * ty * ty});
         }
      }

      std::vector<int> I(static_cast<std::size_t>(GetDimU() + 1), 0);
      std::vector<HYPRE_BigInt> J;
      std::vector<mfem::real_t> data;
      for (int row = 0; row < GetDimU(); ++row)
      {
         for (const Entry &entry : rows[static_cast<std::size_t>(row)])
         {
            J.push_back(entry.column);
            data.push_back(entry.value);
         }
         I[static_cast<std::size_t>(row + 1)] =
            static_cast<int>(J.size());
      }
      gap_displacement_hessian_ =
         std::make_unique<mfem::HypreParMatrix>(
            comm_, GetDimU(), static_cast<HYPRE_BigInt>(GetDimUGlb()),
            static_cast<HYPRE_BigInt>(GetDimUGlb()), I.data(), J.data(),
            data.data(), GetDofOffsetsU(), GetDofOffsetsU());
      return gap_displacement_hessian_.get();
   }

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
   gap_displacement_hessian_.reset(
      GenerateHypreParMatrixFromDiagonal(GetDofOffsetsU(), diagonal));
   return gap_displacement_hessian_.get();
}

mfem::Operator *ParametricElasticSplineContactProblem::Ddthgl(
   const mfem::Vector &d, const mfem::Vector &multiplier,
   const mfem::Vector &theta)
{
   MFEM_VERIFY(multiplier.Size() == LocalContactNodeCount(),
               "The contact multiplier has an incompatible local size.");
   UpdateSurface(theta);
   if (gap_type_ == spline::ContactGapType::ClosestPointNormal)
   {
      spline::ClosestPointBatchResult closest_points;
      MFEM_VERIFY(EvaluateClosestPoints(d, closest_points),
                  "Cannot differentiate an invalid closest-point gap.");

      std::unique_ptr<mfem::HypreParMatrix> data_value(
         new mfem::HypreParMatrix(surface_.EvaluateDataJacobian(
            closest_points.parameter, 0)));
      std::unique_ptr<mfem::HypreParMatrix> data_first(
         new mfem::HypreParMatrix(surface_.EvaluateDataJacobian(
            closest_points.parameter, 1)));
      const mfem::Vector surface_second =
         surface_.Evaluate(closest_points.parameter, 2);

      mfem::Vector value_scale(LocalContactNodeCount());
      mfem::Vector first_scale(LocalContactNodeCount());
      mfem::Vector x_weights(LocalContactNodeCount());
      mfem::Vector y_weights(LocalContactNodeCount());
      for (int i = 0; i < LocalContactNodeCount(); ++i)
      {
         const mfem::real_t inverse_metric = closest_points.normal_y[i];
         const mfem::real_t slope =
            closest_points.tangent_y[i] /
            closest_points.tangent_x[i];
         value_scale[i] = slope * surface_second[i] *
                          inverse_metric * inverse_metric * inverse_metric;
         first_scale[i] = -inverse_metric;
         x_weights[i] = 1.0 / closest_points.projection_hessian[i];
         y_weights[i] = slope / closest_points.projection_hessian[i];
      }
      data_value->ScaleRows(value_scale);
      data_first->ScaleRows(first_scale);
      std::unique_ptr<mfem::HypreParMatrix> xi_data_derivative(
         mfem::Add(1.0, *data_value, 1.0, *data_first));
      xi_data_derivative->ScaleRows(multiplier);

      std::unique_ptr<mfem::HypreParMatrix> point_to_parameter(
         MakeContactDirectionMap(x_weights, y_weights));
      std::unique_ptr<mfem::HypreParMatrix> parameter_to_point(
         point_to_parameter->Transpose());
      gap_mixed_hessian_.reset(mfem::ParMult(
         parameter_to_point.get(), xi_data_derivative.get(), true));
      return gap_mixed_hessian_.get();
   }

   const mfem::Vector points = DisplacedAbscissae(d);
   MFEM_VERIFY(PointsAreInSurfaceDomain(points),
               "Cannot differentiate the gap outside the spline domain.");

   std::unique_ptr<mfem::HypreParMatrix> data_first_derivative(
      new mfem::HypreParMatrix(
         surface_.EvaluateDataJacobian(points, 1)));
   mfem::Vector row_scale(multiplier.Size());
   for (int i = 0; i < multiplier.Size(); ++i)
   {
      row_scale[i] = -multiplier[i];
   }
   data_first_derivative->ScaleRows(row_scale);
   gap_mixed_hessian_.reset(mfem::ParMult(
      contact_x_map_transpose_.get(), data_first_derivative.get(), true));
   return gap_mixed_hessian_.get();
}

mfem::Operator *ParametricElasticSplineContactProblem::Dththgl(
   const mfem::Vector &d, const mfem::Vector &multiplier,
   const mfem::Vector &theta)
{
   MFEM_VERIFY(multiplier.Size() == LocalContactNodeCount(),
               "The contact multiplier has an incompatible local size.");
   UpdateSurface(theta);
   if (gap_type_ != spline::ContactGapType::ClosestPointNormal)
   {
      return ParamOptProblem::Dththgl(d, multiplier, theta);
   }

   spline::ClosestPointBatchResult closest_points;
   MFEM_VERIFY(EvaluateClosestPoints(d, closest_points),
               "Cannot differentiate an invalid closest-point gap.");
   std::unique_ptr<mfem::HypreParMatrix> data_value(
      new mfem::HypreParMatrix(surface_.EvaluateDataJacobian(
         closest_points.parameter, 0)));
   std::unique_ptr<mfem::HypreParMatrix> data_first(
      new mfem::HypreParMatrix(surface_.EvaluateDataJacobian(
         closest_points.parameter, 1)));
   const mfem::Vector surface_second =
      surface_.Evaluate(closest_points.parameter, 2);

   mfem::Vector l_value_scale(LocalContactNodeCount());
   mfem::Vector l_first_scale(LocalContactNodeCount());
   mfem::Vector q_value_scale(LocalContactNodeCount());
   mfem::Vector q_first_scale(LocalContactNodeCount());
   mfem::Vector first_term_scale(LocalContactNodeCount());
   mfem::Vector second_term_scale(LocalContactNodeCount());
   for (int i = 0; i < LocalContactNodeCount(); ++i)
   {
      const mfem::real_t inverse_metric = closest_points.normal_y[i];
      const mfem::real_t slope =
         closest_points.tangent_y[i] /
         closest_points.tangent_x[i];
      const mfem::real_t inverse_metric_cubed =
         inverse_metric * inverse_metric * inverse_metric;
      const mfem::real_t surface_minus_point_y =
         -closest_points.gap[i] * inverse_metric;
      l_value_scale[i] =
         slope * surface_second[i] * inverse_metric_cubed;
      l_first_scale[i] = -inverse_metric;
      q_value_scale[i] = slope;
      q_first_scale[i] = surface_minus_point_y;
      first_term_scale[i] =
         multiplier[i] * slope * inverse_metric_cubed;
      second_term_scale[i] =
         -multiplier[i] / closest_points.projection_hessian[i];
   }

   std::unique_ptr<mfem::HypreParMatrix> l_value(
      new mfem::HypreParMatrix(*data_value));
   std::unique_ptr<mfem::HypreParMatrix> l_first(
      new mfem::HypreParMatrix(*data_first));
   l_value->ScaleRows(l_value_scale);
   l_first->ScaleRows(l_first_scale);
   std::unique_ptr<mfem::HypreParMatrix> l_matrix(
      mfem::Add(1.0, *l_value, 1.0, *l_first));

   std::unique_ptr<mfem::HypreParMatrix> q_value(
      new mfem::HypreParMatrix(*data_value));
   std::unique_ptr<mfem::HypreParMatrix> q_first(
      new mfem::HypreParMatrix(*data_first));
   q_value->ScaleRows(q_value_scale);
   q_first->ScaleRows(q_first_scale);
   std::unique_ptr<mfem::HypreParMatrix> q_matrix(
      mfem::Add(1.0, *q_value, 1.0, *q_first));

   std::unique_ptr<mfem::HypreParMatrix> weighted_first(
      new mfem::HypreParMatrix(*data_first));
   weighted_first->ScaleRows(first_term_scale);
   std::unique_ptr<mfem::HypreParMatrix> data_value_transpose(
      data_value->Transpose());
   std::unique_ptr<mfem::HypreParMatrix> first_term(
      mfem::ParMult(data_value_transpose.get(), weighted_first.get(),
                    true));

   q_matrix->ScaleRows(second_term_scale);
   std::unique_ptr<mfem::HypreParMatrix> l_transpose(
      l_matrix->Transpose());
   std::unique_ptr<mfem::HypreParMatrix> second_term(
      mfem::ParMult(l_transpose.get(), q_matrix.get(), true));
   gap_data_hessian_.reset(
      mfem::Add(1.0, *first_term, 1.0, *second_term));
   return gap_data_hessian_.get();
}

mfem::Operator *ParametricElasticSplineContactProblem::Dddgl2(
   const mfem::Vector &d, const mfem::Vector &direction,
   const mfem::Vector &theta)
{
   MFEM_VERIFY(direction.Size() == GetDimU(),
               "The displacement direction has an incompatible local size.");
   UpdateSurface(theta);
   if (gap_type_ == spline::ContactGapType::ClosestPointNormal)
   {
      spline::ClosestPointBatchResult closest_points;
      MFEM_VERIFY(EvaluateClosestPoints(d, closest_points),
                  "Cannot differentiate an invalid closest-point gap.");

      const int number_of_rows = LocalContactNodeCount();
      std::vector<int> I(static_cast<std::size_t>(number_of_rows + 1), 0);
      std::vector<HYPRE_BigInt> J;
      std::vector<mfem::real_t> data;
      J.reserve(static_cast<std::size_t>(2 * number_of_rows));
      data.reserve(static_cast<std::size_t>(2 * number_of_rows));
      for (int row = 0; row < number_of_rows; ++row)
      {
         const ContactNode &node =
            contact_nodes_[static_cast<std::size_t>(row)];
         const mfem::real_t vx = node.reduced_x_true_dof >= 0
            ? direction[node.reduced_x_true_dof] : 0.0;
         const mfem::real_t vy = node.reduced_y_true_dof >= 0
            ? direction[node.reduced_y_true_dof] : 0.0;
         const mfem::real_t tx = closest_points.tangent_x[row];
         const mfem::real_t ty = closest_points.tangent_y[row];
         const mfem::real_t scale =
            closest_points.curvature_factor[row] * (tx * vx + ty * vy);
         if (node.reduced_x_true_dof >= 0)
         {
            J.push_back(node.reduced_x_global_true_dof);
            data.push_back(scale * tx);
         }
         if (node.reduced_y_true_dof >= 0)
         {
            J.push_back(node.reduced_y_global_true_dof);
            data.push_back(scale * ty);
         }
         I[static_cast<std::size_t>(row + 1)] =
            static_cast<int>(J.size());
      }
      gap_displacement_second_contraction_ =
         std::make_unique<mfem::HypreParMatrix>(
            comm_, number_of_rows,
            static_cast<HYPRE_BigInt>(GetDimMGlb()),
            static_cast<HYPRE_BigInt>(GetDimUGlb()), I.data(), J.data(),
            data.data(), GetDofOffsetsM(), GetDofOffsetsU());
      return gap_displacement_second_contraction_.get();
   }

   const mfem::Vector points = DisplacedAbscissae(d);
   MFEM_VERIFY(PointsAreInSurfaceDomain(points),
               "Cannot differentiate the gap outside the spline domain.");
   const mfem::Vector surface_second_derivative = surface_.Evaluate(points, 2);

   const int number_of_rows = LocalContactNodeCount();
   std::vector<int> I(static_cast<std::size_t>(number_of_rows + 1), 0);
   std::vector<HYPRE_BigInt> J;
   std::vector<mfem::real_t> data;
   J.reserve(static_cast<std::size_t>(number_of_rows));
   data.reserve(static_cast<std::size_t>(number_of_rows));
   for (int row = 0; row < number_of_rows; ++row)
   {
      const ContactNode &node = contact_nodes_[static_cast<std::size_t>(row)];
      if (node.reduced_x_true_dof >= 0)
      {
         J.push_back(node.reduced_x_global_true_dof);
         data.push_back(-direction[node.reduced_x_true_dof] *
                        surface_second_derivative[row]);
      }
      I[static_cast<std::size_t>(row + 1)] = static_cast<int>(J.size());
   }

   gap_displacement_second_contraction_ =
      std::make_unique<mfem::HypreParMatrix>(
         comm_, number_of_rows, static_cast<HYPRE_BigInt>(GetDimMGlb()),
         static_cast<HYPRE_BigInt>(GetDimUGlb()), I.data(), J.data(),
         data.data(), GetDofOffsetsM(), GetDofOffsetsU());
   return gap_displacement_second_contraction_.get();
}

mfem::Operator *ParametricElasticSplineContactProblem::Dthdgl2(
   const mfem::Vector &d, const mfem::Vector &direction,
   const mfem::Vector &theta)
{
   MFEM_VERIFY(direction.Size() == GetDimU(),
               "The displacement direction has an incompatible local size.");
   UpdateSurface(theta);
   if (gap_type_ == spline::ContactGapType::ClosestPointNormal)
   {
      spline::ClosestPointBatchResult closest_points;
      MFEM_VERIFY(EvaluateClosestPoints(d, closest_points),
                  "Cannot differentiate an invalid closest-point gap.");
      std::unique_ptr<mfem::HypreParMatrix> data_value(
         new mfem::HypreParMatrix(surface_.EvaluateDataJacobian(
            closest_points.parameter, 0)));
      std::unique_ptr<mfem::HypreParMatrix> data_first(
         new mfem::HypreParMatrix(surface_.EvaluateDataJacobian(
            closest_points.parameter, 1)));
      const mfem::Vector surface_second =
         surface_.Evaluate(closest_points.parameter, 2);

      mfem::Vector value_scale(LocalContactNodeCount());
      mfem::Vector first_scale(LocalContactNodeCount());
      mfem::Vector direction_scale(LocalContactNodeCount());
      for (int i = 0; i < LocalContactNodeCount(); ++i)
      {
         const ContactNode &node =
            contact_nodes_[static_cast<std::size_t>(i)];
         const mfem::real_t vx = node.reduced_x_true_dof >= 0
            ? direction[node.reduced_x_true_dof] : 0.0;
         const mfem::real_t vy = node.reduced_y_true_dof >= 0
            ? direction[node.reduced_y_true_dof] : 0.0;
         const mfem::real_t inverse_metric = closest_points.normal_y[i];
         const mfem::real_t slope =
            closest_points.tangent_y[i] /
            closest_points.tangent_x[i];
         value_scale[i] = slope * surface_second[i] *
                          inverse_metric * inverse_metric * inverse_metric;
         first_scale[i] = -inverse_metric;
         direction_scale[i] =
            (vx + slope * vy) / closest_points.projection_hessian[i];
      }
      data_value->ScaleRows(value_scale);
      data_first->ScaleRows(first_scale);
      gap_mixed_second_contraction_.reset(
         mfem::Add(1.0, *data_value, 1.0, *data_first));
      gap_mixed_second_contraction_->ScaleRows(direction_scale);
      return gap_mixed_second_contraction_.get();
   }

   const mfem::Vector points = DisplacedAbscissae(d);
   MFEM_VERIFY(PointsAreInSurfaceDomain(points),
               "Cannot differentiate the gap outside the spline domain.");

   gap_mixed_second_contraction_ =
      std::make_unique<mfem::HypreParMatrix>(
         surface_.EvaluateDataJacobian(points, 1));
   mfem::Vector row_scale(LocalContactNodeCount());
   for (int i = 0; i < LocalContactNodeCount(); ++i)
   {
      const ContactNode &node = contact_nodes_[static_cast<std::size_t>(i)];
      row_scale[i] = node.reduced_x_true_dof >= 0
                        ? -direction[node.reduced_x_true_dof]
                        : 0.0;
   }
   gap_mixed_second_contraction_->ScaleRows(row_scale);
   return gap_mixed_second_contraction_.get();
}

mfem::Operator *ParametricElasticSplineContactProblem::Ddddgl3(
   const mfem::Vector &d, const mfem::Vector &multiplier,
   const mfem::Vector &direction, const mfem::Vector &theta)
{
   MFEM_VERIFY(multiplier.Size() == LocalContactNodeCount(),
               "The contact multiplier has an incompatible local size.");
   MFEM_VERIFY(direction.Size() == GetDimU(),
               "The displacement direction has an incompatible local size.");
   UpdateSurface(theta);

   struct Entry
   {
      HYPRE_BigInt column;
      mfem::real_t value;
   };
   std::vector<std::vector<Entry>> rows(
      static_cast<std::size_t>(GetDimU()));

   if (gap_type_ == spline::ContactGapType::ClosestPointNormal)
   {
      spline::ClosestPointBatchResult closest_points;
      MFEM_VERIFY(EvaluateClosestPoints(d, closest_points),
                  "Cannot differentiate an invalid closest-point gap.");
      const mfem::Vector surface_second =
         surface_.Evaluate(closest_points.parameter, 2);
      const mfem::Vector surface_third =
         surface_.Evaluate(closest_points.parameter, 3);

      for (int i = 0; i < LocalContactNodeCount(); ++i)
      {
         const ContactNode &node =
            contact_nodes_[static_cast<std::size_t>(i)];
         const mfem::real_t vx = node.reduced_x_true_dof >= 0
            ? direction[node.reduced_x_true_dof] : 0.0;
         const mfem::real_t vy = node.reduced_y_true_dof >= 0
            ? direction[node.reduced_y_true_dof] : 0.0;
         const mfem::real_t inverse_metric = closest_points.normal_y[i];
         const mfem::real_t slope =
            closest_points.tangent_y[i] /
            closest_points.tangent_x[i];
         const mfem::real_t second = surface_second[i];
         const mfem::real_t third = surface_third[i];
         const mfem::real_t projection_hessian =
            closest_points.projection_hessian[i];
         const mfem::real_t surface_minus_point_y =
            -closest_points.gap[i] * inverse_metric;
         const mfem::real_t parameter_rate =
            (vx + slope * vy) / projection_hessian;
         const mfem::real_t slope_rate = second * parameter_rate;
         const mfem::real_t second_rate = third * parameter_rate;
         const mfem::real_t inverse_metric_rate =
            -slope * second * parameter_rate *
            inverse_metric * inverse_metric * inverse_metric;
         const mfem::real_t projection_hessian_rate =
            (3.0 * slope * second +
             surface_minus_point_y * third) * parameter_rate -
            second * vy;

         const mfem::real_t tx = closest_points.tangent_x[i];
         const mfem::real_t ty = closest_points.tangent_y[i];
         const mfem::real_t tx_rate = inverse_metric_rate;
         const mfem::real_t ty_rate =
            slope_rate * inverse_metric +
            slope * inverse_metric_rate;
         const mfem::real_t curvature =
            closest_points.curvature_factor[i];
         const mfem::real_t curvature_rate =
            -(second_rate * inverse_metric +
              second * inverse_metric_rate) /
               projection_hessian +
            second * inverse_metric * projection_hessian_rate /
               (projection_hessian * projection_hessian);
         const mfem::real_t scale = multiplier[i];
         const mfem::real_t hxx = scale *
            (curvature_rate * tx * tx +
             2.0 * curvature * tx_rate * tx);
         const mfem::real_t hxy = scale *
            (curvature_rate * tx * ty +
             curvature * (tx_rate * ty + tx * ty_rate));
         const mfem::real_t hyy = scale *
            (curvature_rate * ty * ty +
             2.0 * curvature * ty_rate * ty);

         if (node.reduced_x_true_dof >= 0)
         {
            auto &row = rows[static_cast<std::size_t>(
               node.reduced_x_true_dof)];
            row.push_back({node.reduced_x_global_true_dof, hxx});
            if (node.reduced_y_true_dof >= 0)
            {
               row.push_back({node.reduced_y_global_true_dof, hxy});
            }
         }
         if (node.reduced_y_true_dof >= 0)
         {
            auto &row = rows[static_cast<std::size_t>(
               node.reduced_y_true_dof)];
            if (node.reduced_x_true_dof >= 0)
            {
               row.push_back({node.reduced_x_global_true_dof, hxy});
            }
            row.push_back({node.reduced_y_global_true_dof, hyy});
         }
      }
   }
   else
   {
      const mfem::Vector points = DisplacedAbscissae(d);
      MFEM_VERIFY(PointsAreInSurfaceDomain(points),
                  "Cannot differentiate the gap outside the spline domain.");
      const mfem::Vector surface_third = surface_.Evaluate(points, 3);
      for (int i = 0; i < LocalContactNodeCount(); ++i)
      {
         const ContactNode &node =
            contact_nodes_[static_cast<std::size_t>(i)];
         if (node.reduced_x_true_dof >= 0)
         {
            const mfem::real_t value =
               -multiplier[i] * surface_third[i] *
               direction[node.reduced_x_true_dof];
            rows[static_cast<std::size_t>(node.reduced_x_true_dof)]
               .push_back({node.reduced_x_global_true_dof, value});
         }
      }
   }

   std::vector<int> I(static_cast<std::size_t>(GetDimU() + 1), 0);
   std::vector<HYPRE_BigInt> J;
   std::vector<mfem::real_t> data;
   for (int row = 0; row < GetDimU(); ++row)
   {
      for (const Entry &entry : rows[static_cast<std::size_t>(row)])
      {
         J.push_back(entry.column);
         data.push_back(entry.value);
      }
      I[static_cast<std::size_t>(row + 1)] =
         static_cast<int>(J.size());
   }
   gap_displacement_third_contraction_ =
      std::make_unique<mfem::HypreParMatrix>(
         comm_, GetDimU(), static_cast<HYPRE_BigInt>(GetDimUGlb()),
         static_cast<HYPRE_BigInt>(GetDimUGlb()), I.data(), J.data(),
         data.data(), GetDofOffsetsU(), GetDofOffsetsU());
   return gap_displacement_third_contraction_.get();
}

mfem::Operator *ParametricElasticSplineContactProblem::Ddthdgl3(
   const mfem::Vector &d, const mfem::Vector &multiplier,
   const mfem::Vector &direction, const mfem::Vector &theta)
{
   MFEM_VERIFY(multiplier.Size() == LocalContactNodeCount(),
               "The contact multiplier has an incompatible local size.");
   MFEM_VERIFY(direction.Size() == GetDimU(),
               "The displacement direction has an incompatible local size.");
   UpdateSurface(theta);

   if (gap_type_ != spline::ContactGapType::ClosestPointNormal)
   {
      const mfem::Vector points = DisplacedAbscissae(d);
      MFEM_VERIFY(PointsAreInSurfaceDomain(points),
                  "Cannot differentiate the gap outside the spline domain.");
      std::unique_ptr<mfem::HypreParMatrix> data_second(
         new mfem::HypreParMatrix(
            surface_.EvaluateDataJacobian(points, 2)));
      mfem::Vector row_scale(LocalContactNodeCount());
      for (int i = 0; i < LocalContactNodeCount(); ++i)
      {
         const ContactNode &node =
            contact_nodes_[static_cast<std::size_t>(i)];
         row_scale[i] = node.reduced_x_true_dof >= 0
            ? -multiplier[i] * direction[node.reduced_x_true_dof]
            : 0.0;
      }
      data_second->ScaleRows(row_scale);
      gap_mixed_third_contraction_.reset(mfem::ParMult(
         contact_x_map_transpose_.get(), data_second.get(), true));
      return gap_mixed_third_contraction_.get();
   }

   spline::ClosestPointBatchResult closest_points;
   MFEM_VERIFY(EvaluateClosestPoints(d, closest_points),
               "Cannot differentiate an invalid closest-point gap.");
   std::unique_ptr<mfem::HypreParMatrix> data_value(
      new mfem::HypreParMatrix(surface_.EvaluateDataJacobian(
         closest_points.parameter, 0)));
   std::unique_ptr<mfem::HypreParMatrix> data_first(
      new mfem::HypreParMatrix(surface_.EvaluateDataJacobian(
         closest_points.parameter, 1)));
   std::unique_ptr<mfem::HypreParMatrix> data_second(
      new mfem::HypreParMatrix(surface_.EvaluateDataJacobian(
         closest_points.parameter, 2)));
   const mfem::Vector surface_second =
      surface_.Evaluate(closest_points.parameter, 2);
   const mfem::Vector surface_third =
      surface_.Evaluate(closest_points.parameter, 3);

   mfem::Vector l_value_scale(LocalContactNodeCount());
   mfem::Vector l_first_scale(LocalContactNodeCount());
   mfem::Vector ldot_value_scale(LocalContactNodeCount());
   mfem::Vector ldot_first_scale(LocalContactNodeCount());
   mfem::Vector ldot_second_scale(LocalContactNodeCount());
   mfem::Vector p_x(LocalContactNodeCount());
   mfem::Vector p_y(LocalContactNodeCount());
   mfem::Vector pdot_x(LocalContactNodeCount());
   mfem::Vector pdot_y(LocalContactNodeCount());
   for (int i = 0; i < LocalContactNodeCount(); ++i)
   {
      const ContactNode &node =
         contact_nodes_[static_cast<std::size_t>(i)];
      const mfem::real_t vx = node.reduced_x_true_dof >= 0
         ? direction[node.reduced_x_true_dof] : 0.0;
      const mfem::real_t vy = node.reduced_y_true_dof >= 0
         ? direction[node.reduced_y_true_dof] : 0.0;
      const mfem::real_t inverse_metric = closest_points.normal_y[i];
      const mfem::real_t slope =
         closest_points.tangent_y[i] /
         closest_points.tangent_x[i];
      const mfem::real_t second = surface_second[i];
      const mfem::real_t third = surface_third[i];
      const mfem::real_t projection_hessian =
         closest_points.projection_hessian[i];
      const mfem::real_t surface_minus_point_y =
         -closest_points.gap[i] * inverse_metric;
      const mfem::real_t parameter_rate =
         (vx + slope * vy) / projection_hessian;
      const mfem::real_t slope_rate = second * parameter_rate;
      const mfem::real_t second_rate = third * parameter_rate;
      const mfem::real_t inverse_metric_rate =
         -slope * second * parameter_rate *
         inverse_metric * inverse_metric * inverse_metric;
      const mfem::real_t projection_hessian_rate =
         (3.0 * slope * second +
          surface_minus_point_y * third) * parameter_rate -
         second * vy;
      const mfem::real_t inverse_metric_cubed =
         inverse_metric * inverse_metric * inverse_metric;
      const mfem::real_t l_value =
         slope * second * inverse_metric_cubed;
      const mfem::real_t l_first = -inverse_metric;
      const mfem::real_t l_value_rate =
         (slope_rate * second + slope * second_rate) *
            inverse_metric_cubed +
         3.0 * slope * second * inverse_metric * inverse_metric *
            inverse_metric_rate;
      const mfem::real_t l_first_rate = -inverse_metric_rate;

      l_value_scale[i] = l_value;
      l_first_scale[i] = l_first;
      ldot_value_scale[i] = l_value_rate;
      ldot_first_scale[i] =
         l_value * parameter_rate + l_first_rate;
      ldot_second_scale[i] = l_first * parameter_rate;
      p_x[i] = 1.0 / projection_hessian;
      p_y[i] = slope / projection_hessian;
      pdot_x[i] = -projection_hessian_rate /
         (projection_hessian * projection_hessian);
      pdot_y[i] = slope_rate / projection_hessian -
         slope * projection_hessian_rate /
            (projection_hessian * projection_hessian);
   }

   std::unique_ptr<mfem::HypreParMatrix> l_value_matrix(
      new mfem::HypreParMatrix(*data_value));
   std::unique_ptr<mfem::HypreParMatrix> l_first_matrix(
      new mfem::HypreParMatrix(*data_first));
   l_value_matrix->ScaleRows(l_value_scale);
   l_first_matrix->ScaleRows(l_first_scale);
   std::unique_ptr<mfem::HypreParMatrix> l_matrix(
      mfem::Add(1.0, *l_value_matrix, 1.0, *l_first_matrix));

   std::unique_ptr<mfem::HypreParMatrix> ldot_value_matrix(
      new mfem::HypreParMatrix(*data_value));
   std::unique_ptr<mfem::HypreParMatrix> ldot_first_matrix(
      new mfem::HypreParMatrix(*data_first));
   std::unique_ptr<mfem::HypreParMatrix> ldot_second_matrix(
      new mfem::HypreParMatrix(*data_second));
   ldot_value_matrix->ScaleRows(ldot_value_scale);
   ldot_first_matrix->ScaleRows(ldot_first_scale);
   ldot_second_matrix->ScaleRows(ldot_second_scale);
   std::unique_ptr<mfem::HypreParMatrix> ldot_partial(
      mfem::Add(1.0, *ldot_value_matrix, 1.0,
                *ldot_first_matrix));
   std::unique_ptr<mfem::HypreParMatrix> ldot_matrix(
      mfem::Add(1.0, *ldot_partial, 1.0, *ldot_second_matrix));

   l_matrix->ScaleRows(multiplier);
   ldot_matrix->ScaleRows(multiplier);
   std::unique_ptr<mfem::HypreParMatrix> p_matrix(
      MakeContactDirectionMap(p_x, p_y));
   std::unique_ptr<mfem::HypreParMatrix> pdot_matrix(
      MakeContactDirectionMap(pdot_x, pdot_y));
   std::unique_ptr<mfem::HypreParMatrix> p_transpose(
      p_matrix->Transpose());
   std::unique_ptr<mfem::HypreParMatrix> pdot_transpose(
      pdot_matrix->Transpose());
   std::unique_ptr<mfem::HypreParMatrix> first_term(
      mfem::ParMult(pdot_transpose.get(), l_matrix.get(), true));
   std::unique_ptr<mfem::HypreParMatrix> second_term(
      mfem::ParMult(p_transpose.get(), ldot_matrix.get(), true));
   gap_mixed_third_contraction_.reset(
      mfem::Add(1.0, *first_term, 1.0, *second_term));
   return gap_mixed_third_contraction_.get();
}

mfem::Operator *ParametricElasticSplineContactProblem::Dththdgl3(
   const mfem::Vector &d, const mfem::Vector &multiplier,
   const mfem::Vector &direction, const mfem::Vector &theta)
{
   MFEM_VERIFY(multiplier.Size() == LocalContactNodeCount(),
               "The contact multiplier has an incompatible local size.");
   MFEM_VERIFY(direction.Size() == GetDimU(),
               "The displacement direction has an incompatible local size.");
   UpdateSurface(theta);
   if (gap_type_ != spline::ContactGapType::ClosestPointNormal)
   {
      return ParamOptProblem::Dththdgl3(d, multiplier, direction, theta);
   }

   spline::ClosestPointBatchResult closest_points;
   MFEM_VERIFY(EvaluateClosestPoints(d, closest_points),
               "Cannot differentiate an invalid closest-point gap.");
   std::unique_ptr<mfem::HypreParMatrix> b0(
      new mfem::HypreParMatrix(surface_.EvaluateDataJacobian(
         closest_points.parameter, 0)));
   std::unique_ptr<mfem::HypreParMatrix> b1(
      new mfem::HypreParMatrix(surface_.EvaluateDataJacobian(
         closest_points.parameter, 1)));
   std::unique_ptr<mfem::HypreParMatrix> b2(
      new mfem::HypreParMatrix(surface_.EvaluateDataJacobian(
         closest_points.parameter, 2)));
   const mfem::Vector surface_second =
      surface_.Evaluate(closest_points.parameter, 2);
   const mfem::Vector surface_third =
      surface_.Evaluate(closest_points.parameter, 3);

   mfem::Vector parameter_rate(LocalContactNodeCount());
   mfem::Vector f(LocalContactNodeCount());
   mfem::Vector f_rate(LocalContactNodeCount());
   mfem::Vector l0(LocalContactNodeCount());
   mfem::Vector l1(LocalContactNodeCount());
   mfem::Vector ldot0(LocalContactNodeCount());
   mfem::Vector ldot1(LocalContactNodeCount());
   mfem::Vector ldot2(LocalContactNodeCount());
   mfem::Vector q0(LocalContactNodeCount());
   mfem::Vector q1(LocalContactNodeCount());
   mfem::Vector qdot0(LocalContactNodeCount());
   mfem::Vector qdot1(LocalContactNodeCount());
   mfem::Vector qdot2(LocalContactNodeCount());
   mfem::Vector inverse_hessian(LocalContactNodeCount());
   mfem::Vector hessian_rate_over_square(LocalContactNodeCount());
   for (int i = 0; i < LocalContactNodeCount(); ++i)
   {
      const ContactNode &node =
         contact_nodes_[static_cast<std::size_t>(i)];
      const mfem::real_t vx = node.reduced_x_true_dof >= 0
         ? direction[node.reduced_x_true_dof] : 0.0;
      const mfem::real_t vy = node.reduced_y_true_dof >= 0
         ? direction[node.reduced_y_true_dof] : 0.0;
      const mfem::real_t a = closest_points.normal_y[i];
      const mfem::real_t slope =
         closest_points.tangent_y[i] /
         closest_points.tangent_x[i];
      const mfem::real_t second = surface_second[i];
      const mfem::real_t third = surface_third[i];
      const mfem::real_t hessian = closest_points.projection_hessian[i];
      const mfem::real_t e = -closest_points.gap[i] * a;
      const mfem::real_t xi_rate = (vx + slope * vy) / hessian;
      const mfem::real_t slope_rate = second * xi_rate;
      const mfem::real_t second_rate = third * xi_rate;
      const mfem::real_t e_rate = slope * xi_rate - vy;
      const mfem::real_t a_rate =
         -slope * second * xi_rate * a * a * a;
      const mfem::real_t hessian_rate =
         (3.0 * slope * second + e * third) * xi_rate - second * vy;
      const mfem::real_t a3 = a * a * a;
      const mfem::real_t l_value = slope * second * a3;
      const mfem::real_t l_first = -a;

      parameter_rate[i] = xi_rate;
      f[i] = slope * a3;
      f_rate[i] = slope_rate * a3 +
                  3.0 * slope * a * a * a_rate;
      l0[i] = l_value;
      l1[i] = l_first;
      ldot0[i] =
         (slope_rate * second + slope * second_rate) * a3 +
         3.0 * slope * second * a * a * a_rate;
      ldot1[i] = l_value * xi_rate - a_rate;
      ldot2[i] = l_first * xi_rate;
      q0[i] = slope;
      q1[i] = e;
      qdot0[i] = slope_rate;
      qdot1[i] = slope * xi_rate + e_rate;
      qdot2[i] = e * xi_rate;
      inverse_hessian[i] = 1.0 / hessian;
      hessian_rate_over_square[i] =
         hessian_rate / (hessian * hessian);
   }

   auto linear_combination_three = [](
      const mfem::HypreParMatrix &matrix0, const mfem::Vector &scale0,
      const mfem::HypreParMatrix &matrix1, const mfem::Vector &scale1,
      const mfem::HypreParMatrix &matrix2, const mfem::Vector &scale2)
   {
      std::unique_ptr<mfem::HypreParMatrix> term0(
         new mfem::HypreParMatrix(matrix0));
      std::unique_ptr<mfem::HypreParMatrix> term1(
         new mfem::HypreParMatrix(matrix1));
      std::unique_ptr<mfem::HypreParMatrix> term2(
         new mfem::HypreParMatrix(matrix2));
      term0->ScaleRows(scale0);
      term1->ScaleRows(scale1);
      term2->ScaleRows(scale2);
      std::unique_ptr<mfem::HypreParMatrix> partial(
         mfem::Add(1.0, *term0, 1.0, *term1));
      return std::unique_ptr<mfem::HypreParMatrix>(
         mfem::Add(1.0, *partial, 1.0, *term2));
   };
   auto weighted_product = [](
      const mfem::HypreParMatrix &left,
      const mfem::HypreParMatrix &right,
      const mfem::Vector &row_weight)
   {
      std::unique_ptr<mfem::HypreParMatrix> weighted_right(
         new mfem::HypreParMatrix(right));
      weighted_right->ScaleRows(row_weight);
      std::unique_ptr<mfem::HypreParMatrix> left_transpose(
         left.Transpose());
      return std::unique_ptr<mfem::HypreParMatrix>(mfem::ParMult(
         left_transpose.get(), weighted_right.get(), true));
   };

   mfem::Vector b0dot_scale(parameter_rate);
   mfem::Vector b1dot_scale(parameter_rate);
   std::unique_ptr<mfem::HypreParMatrix> b0dot(
      new mfem::HypreParMatrix(*b1));
   std::unique_ptr<mfem::HypreParMatrix> b1dot(
      new mfem::HypreParMatrix(*b2));
   b0dot->ScaleRows(b0dot_scale);
   b1dot->ScaleRows(b1dot_scale);

   mfem::Vector zero(LocalContactNodeCount());
   zero = 0.0;
   std::unique_ptr<mfem::HypreParMatrix> l_matrix(
      linear_combination_three(*b0, l0, *b1, l1, *b2, zero));
   std::unique_ptr<mfem::HypreParMatrix> ldot_matrix(
      linear_combination_three(*b0, ldot0, *b1, ldot1,
                               *b2, ldot2));
   std::unique_ptr<mfem::HypreParMatrix> q_matrix(
      linear_combination_three(*b0, q0, *b1, q1, *b2, zero));
   std::unique_ptr<mfem::HypreParMatrix> qdot_matrix(
      linear_combination_three(*b0, qdot0, *b1, qdot1,
                               *b2, qdot2));

   mfem::Vector weight_a(LocalContactNodeCount());
   mfem::Vector weight_b(LocalContactNodeCount());
   mfem::Vector weight_c(LocalContactNodeCount());
   mfem::Vector weight_d(LocalContactNodeCount());
   mfem::Vector weight_e(LocalContactNodeCount());
   mfem::Vector weight_f(LocalContactNodeCount());
   for (int i = 0; i < LocalContactNodeCount(); ++i)
   {
      weight_a[i] = multiplier[i] * f[i];
      weight_b[i] = multiplier[i] * f_rate[i];
      weight_c[i] = multiplier[i] * f[i];
      weight_d[i] = -multiplier[i] * inverse_hessian[i];
      weight_e[i] = -multiplier[i] * inverse_hessian[i];
      weight_f[i] = multiplier[i] * hessian_rate_over_square[i];
   }
   std::unique_ptr<mfem::HypreParMatrix> term_a(
      weighted_product(*b0dot, *b1, weight_a));
   std::unique_ptr<mfem::HypreParMatrix> term_b(
      weighted_product(*b0, *b1, weight_b));
   std::unique_ptr<mfem::HypreParMatrix> term_c(
      weighted_product(*b0, *b1dot, weight_c));
   std::unique_ptr<mfem::HypreParMatrix> term_d(
      weighted_product(*ldot_matrix, *q_matrix, weight_d));
   std::unique_ptr<mfem::HypreParMatrix> term_e(
      weighted_product(*l_matrix, *qdot_matrix, weight_e));
   std::unique_ptr<mfem::HypreParMatrix> term_f(
      weighted_product(*l_matrix, *q_matrix, weight_f));
   std::unique_ptr<mfem::HypreParMatrix> sum_ab(
      mfem::Add(1.0, *term_a, 1.0, *term_b));
   std::unique_ptr<mfem::HypreParMatrix> sum_abc(
      mfem::Add(1.0, *sum_ab, 1.0, *term_c));
   std::unique_ptr<mfem::HypreParMatrix> sum_abcd(
      mfem::Add(1.0, *sum_abc, 1.0, *term_d));
   std::unique_ptr<mfem::HypreParMatrix> sum_abcde(
      mfem::Add(1.0, *sum_abcd, 1.0, *term_e));
   gap_data_third_contraction_.reset(
      mfem::Add(1.0, *sum_abcde, 1.0, *term_f));
   return gap_data_third_contraction_.get();
}

void ParametricElasticSplineContactProblem::SetPotentialContactValues(
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
      const int true_dof = field_space->GetLocalTDofNumber(vertex_dofs[0]);
      MFEM_VERIFY(true_dof >= 0,
                  "Contact-node ownership differs between scalar and vector "
                  "P1 spaces.");
      true_values[true_dof] = contact_values[i];
   }
   field.SetFromTrueDofs(true_values);
}
