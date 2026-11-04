#ifndef PARAMETRIC_SPLINE_CONTACT_PROBLEM_HPP
#define PARAMETRIC_SPLINE_CONTACT_PROBLEM_HPP

#include <memory>
#include <vector>

#include "mfem.hpp"

#include "../../continuationsolvers/problems/OptProblems.hpp"
#include "../Spline/quintic_smoothing_spline_1d.hpp"

/**
 * Linear elasticity with nodal vertical-gap constraints parameterized by the
 * data values of a parallel quintic smoothing spline,
 *
 *   g_i(d,theta) = y_i + d_yi - S(x_i + d_xi; theta) >= 0.
 *
 * The displacement vector contains only non-essential true DOFs.  The design
 * vector theta contains the locally owned true DOFs of surface_data_space.
 */
class ParametricElasticSplineContactProblem : public ParamOptProblem
{
public:
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
      const spline::QuinticSmoothingSolveOptions &surface_solve_options =
         spline::QuinticSmoothingSolveOptions());

   double E(const mfem::Vector &d, const mfem::Vector &theta,
            int &eval_err) override;
   void DdE(const mfem::Vector &d, const mfem::Vector &theta,
            mfem::Vector &gradient) override;
   mfem::Operator *DddE(const mfem::Vector &d,
                        const mfem::Vector &theta) override;
   mfem::Operator *DdthE(const mfem::Vector &d,
                         const mfem::Vector &theta) override;

   void g(const mfem::Vector &d, const mfem::Vector &theta,
          mfem::Vector &gap, int &eval_err) override;
   mfem::Operator *Ddg(const mfem::Vector &d,
                       const mfem::Vector &theta) override;
   mfem::Operator *Dthg(const mfem::Vector &d,
                        const mfem::Vector &theta) override;

   mfem::Operator *Dddgl(const mfem::Vector &d,
                         const mfem::Vector &multiplier,
                         const mfem::Vector &theta) override;
   mfem::Operator *Ddthgl(const mfem::Vector &d,
                          const mfem::Vector &multiplier,
                          const mfem::Vector &theta) override;
   mfem::Operator *Dddgl2(const mfem::Vector &d,
                          const mfem::Vector &direction,
                          const mfem::Vector &theta) override;
   mfem::Operator *Dthdgl2(const mfem::Vector &d,
                           const mfem::Vector &direction,
                           const mfem::Vector &theta) override;

   void SetSurfaceData(const mfem::Vector &theta);

   void ProlongateToFullDofs(const mfem::Vector &d,
                             mfem::Vector &full_d) const;
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
   spline::QuinticSmoothingSpline1D &surface_;
   mfem::ParFiniteElementSpace &surface_data_space_;
   MPI_Comm comm_;
   int contact_boundary_attribute_;
   mfem::real_t surface_x_min_;
   mfem::real_t surface_x_max_;
   int full_local_true_dof_count_ = 0;
   spline::QuinticSmoothingSolveOptions surface_solve_options_;

   std::vector<int> full_to_reduced_local_;
   std::vector<ContactNode> contact_nodes_;

   std::unique_ptr<mfem::HypreParMatrix> restriction_;
   std::unique_ptr<mfem::HypreParMatrix> prolongation_;
   std::unique_ptr<mfem::HypreParMatrix> stiffness_;
   std::unique_ptr<mfem::HypreParMatrix> zero_displacement_data_hessian_;
   std::unique_ptr<mfem::HypreParMatrix> contact_x_map_;
   std::unique_ptr<mfem::HypreParMatrix> contact_x_map_transpose_;

   std::unique_ptr<mfem::HypreParMatrix> gap_displacement_jacobian_;
   std::unique_ptr<mfem::HypreParMatrix> gap_data_jacobian_;
   std::unique_ptr<mfem::HypreParMatrix> gap_displacement_hessian_;
   std::unique_ptr<mfem::HypreParMatrix> gap_mixed_hessian_;
   std::unique_ptr<mfem::HypreParMatrix>
      gap_displacement_second_contraction_;
   std::unique_ptr<mfem::HypreParMatrix> gap_mixed_second_contraction_;

   mfem::Vector reduced_load_;
   mfem::Vector full_displacement_;
   mfem::Vector stiffness_times_displacement_;
   mfem::Vector cached_theta_;
   bool surface_data_are_current_ = false;

   void BuildContactNodes(HYPRE_BigInt first_reduced_global_dof);
   void BuildContactXMap();
   void UpdateSurface(const mfem::Vector &theta);
   mfem::Vector DisplacedAbscissae(const mfem::Vector &d);
   bool PointsAreInSurfaceDomain(const mfem::Vector &points) const;
};

#endif // PARAMETRIC_SPLINE_CONTACT_PROBLEM_HPP
