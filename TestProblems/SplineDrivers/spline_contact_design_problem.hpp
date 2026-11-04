#ifndef SPLINE_CONTACT_DESIGN_PROBLEM_HPP
#define SPLINE_CONTACT_DESIGN_PROBLEM_HPP

#include "mfem.hpp"

#include "../../continuationsolvers/problems/MPECProblems.hpp"

/**
 * A first upper-level objective for the elastic spline-contact MPEC:
 *
 *   J(u,p,theta) = displacement_weight/2 ||u||_2^2
 *                + design_weight/2 ||theta-theta_reference||_2^2
 *                + design_smoothing_weight/2 theta^T K theta,
 *
 * where K is the stiffness matrix for the one-dimensional surface-data
 * finite-element space.
 *
 * The lower-level equilibrium/contact equations are supplied by MPECProblem
 * through a ParamOptProblem.  This intentionally simple Euclidean objective
 * is meant as a runnable starting point; finite-element mass weighting and a
 * problem-specific design target can be added without changing the contact
 * constraint implementation.
 */
class SplineContactDesignProblem : public MPECProblem
{
public:
   SplineContactDesignProblem(
      ParamOptProblem *contact_problem,
      const mfem::Vector &theta_reference,
      const mfem::HypreParMatrix &design_stiffness,
      mfem::real_t displacement_weight,
      mfem::real_t design_weight,
      mfem::real_t design_smoothing_weight,
      mfem::real_t max_pressure_weight,
      mfem::real_t max_pressure_smoothing,
      mfem::real_t pressure_uniformity_weight,
      bool use_slacks = true);

   double E(const mfem::Vector &u, const mfem::Vector &p,
            const mfem::Vector &theta, int &eval_err) override;

   void DuE(const mfem::Vector &u, const mfem::Vector &p,
            const mfem::Vector &theta, mfem::Vector &gradient) override;
   void DpE(const mfem::Vector &u, const mfem::Vector &p,
            const mfem::Vector &theta, mfem::Vector &gradient) override;
   void DthE(const mfem::Vector &u, const mfem::Vector &p,
             const mfem::Vector &theta, mfem::Vector &gradient) override;

   mfem::Operator *DuuE(const mfem::Vector &u,
                        const mfem::Vector &p,
                        const mfem::Vector &theta) override;
   mfem::Operator *DppE(const mfem::Vector &u,
                        const mfem::Vector &p,
                        const mfem::Vector &theta) override;
   mfem::Operator *DththE(const mfem::Vector &u,
                          const mfem::Vector &p,
                          const mfem::Vector &theta) override;

private:
   mfem::Vector theta_reference_;
   mfem::Vector design_change_;
   ParamOptProblem * contact_problem_;
   // Non-owning: the assembled matrix supplied to the constructor must
   // outlive this problem.
   const mfem::HypreParMatrix &design_stiffness_;
   mfem::real_t displacement_weight_;
   mfem::real_t design_weight_;
   mfem::real_t design_smoothing_weight_;
   mfem::real_t max_pressure_weight_ = 0.0;
   mfem::real_t max_pressure_smoothing_ = 1.0;
   mfem::real_t pressure_uniformity_weight_ = 0.0;
};

#endif // SPLINE_CONTACT_DESIGN_PROBLEM_HPP
