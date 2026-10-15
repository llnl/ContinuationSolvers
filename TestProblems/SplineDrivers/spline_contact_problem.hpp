#ifndef ELASTIC_SPLINE_CONTACT_PROBLEM_HPP
#define ELASTIC_SPLINE_CONTACT_PROBLEM_HPP

#include <memory>
#include <vector>

#include "mfem.hpp"

#include "../../continuationsolvers/problems/OptProblems.hpp"
#include "../Spline/quintic_smoothing_spline_1d.hpp"

/**
 * Minimize the small-strain elastic potential energy subject to nodal
 * vertical-gap constraints on one boundary attribute,
 *
 *   g_i(d) = y_i + d_{y,i} - S(x_i + d_{x,i}) >= 0.
 *
 * The optimization variable contains only non-essential displacement true
 * DOFs. Homogeneous essential values are therefore eliminated exactly rather
 * than imposed by rows in the elasticity matrix.
 *
 * There is one constraint for each globally unique P1 contact-boundary node.
 * A shared node is assigned to the MPI rank that owns its displacement true
 * DOFs, so interface constraints are never duplicated.
 */
class ElasticSplineContactProblem : public OptProblem
{
public:
   ElasticSplineContactProblem(
      mfem::ParFiniteElementSpace &displacement_space,
      mfem::ParBilinearForm &elasticity,
      mfem::ParLinearForm &load,
      const mfem::Array<int> &essential_true_dofs,
      int contact_boundary_attribute,
      const spline::QuinticSmoothingSpline1D &surface,
      mfem::real_t surface_x_min,
      mfem::real_t surface_x_max);

   double E(const mfem::Vector &d, int &eval_err) override;
   void DdE(const mfem::Vector &d, mfem::Vector &gradient) override;
   mfem::Operator *DddE(const mfem::Vector &d) override;

   void g(const mfem::Vector &d, mfem::Vector &gap,
          int &eval_err) override;
   mfem::Operator *Ddg(const mfem::Vector &d) override;
   mfem::Operator *Dddgl(const mfem::Vector &d,
                         const mfem::Vector &multiplier) override;

   /// Expand the reduced optimization vector and insert zero essential data.
   void ProlongateToFullDofs(const mfem::Vector &d,
                             mfem::Vector &full_d) const;

   /**
    * Place locally owned contact values in a scalar P1 grid function.
    * Non-contact nodes are set to zero.
    */
   void SetPotentialContactValues(const mfem::Vector &contact_values,
                                  mfem::ParGridFunction &field) const;

   int LocalContactNodeCount() const
   {
      return static_cast<int>(contact_nodes_.size());
   }

private:
   struct ContactNode
   {
      int local_vertex = -1;
      mfem::real_t x = 0.0;
      mfem::real_t y = 0.0;
      int full_x_true_dof = -1;
      int full_y_true_dof = -1;
      int reduced_x_true_dof = -1;
      int reduced_y_true_dof = -1;
      HYPRE_BigInt reduced_x_global_true_dof = -1;
      HYPRE_BigInt reduced_y_global_true_dof = -1;
   };

   mfem::ParFiniteElementSpace &displacement_space_;
   mfem::ParMesh &mesh_;
   const spline::QuinticSmoothingSpline1D &surface_;
   MPI_Comm comm_;
   int contact_boundary_attribute_;
   mfem::real_t surface_x_min_;
   mfem::real_t surface_x_max_;
   int full_local_true_dof_count_ = 0;

   std::vector<int> full_to_reduced_local_;
   std::vector<ContactNode> contact_nodes_;

   std::unique_ptr<mfem::HypreParMatrix> restriction_;
   std::unique_ptr<mfem::HypreParMatrix> prolongation_;
   std::unique_ptr<mfem::HypreParMatrix> stiffness_;
   std::unique_ptr<mfem::HypreParMatrix> gap_jacobian_;
   std::unique_ptr<mfem::HypreParMatrix> gap_hessian_;

   mfem::Vector reduced_load_;
   mfem::Vector full_displacement_;
   mfem::Vector stiffness_times_displacement_;

   void BuildContactNodes(HYPRE_BigInt first_reduced_global_dof);
   mfem::Vector DisplacedAbscissae(const mfem::Vector &d);
   bool PointsAreInSurfaceDomain(const mfem::Vector &points) const;
};

#endif // ELASTIC_SPLINE_CONTACT_PROBLEM_HPP
