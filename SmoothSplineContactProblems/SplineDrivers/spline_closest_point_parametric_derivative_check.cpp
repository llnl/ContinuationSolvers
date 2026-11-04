#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include "parametric_spline_contact_problem.hpp"

namespace
{

enum BoundaryAttribute
{
   BOTTOM = 1,
   RIGHT = 2,
   TOP = 3,
   LEFT = 4
};

mfem::Array<int> MakeBoundaryMarker(const mfem::ParMesh &mesh,
                                    int attribute)
{
   mfem::Array<int> marker(mesh.bdr_attributes.Max());
   marker = 0;
   marker[attribute - 1] = 1;
   return marker;
}

mfem::real_t GlobalInnerProduct(const mfem::Vector &left,
                                const mfem::Vector &right,
                                MPI_Comm comm)
{
   return mfem::InnerProduct(comm, left, right);
}

mfem::real_t ProductNorm(const mfem::Vector &first,
                         const mfem::Vector &second,
                         MPI_Comm comm)
{
   return std::sqrt(std::max<mfem::real_t>(
      0.0, GlobalInnerProduct(first, first, comm) +
              GlobalInnerProduct(second, second, comm)));
}

mfem::real_t GlobalNorm(const mfem::Vector &vector, MPI_Comm comm)
{
   return std::sqrt(std::max<mfem::real_t>(
      0.0, GlobalInnerProduct(vector, vector, comm)));
}

mfem::real_t ObservedOrder(mfem::real_t previous_error,
                           mfem::real_t current_error,
                           mfem::real_t refinement_ratio)
{
   if (previous_error <= 0.0 || current_error <= 0.0 ||
       !std::isfinite(previous_error) || !std::isfinite(current_error))
   {
      return std::numeric_limits<mfem::real_t>::quiet_NaN();
   }
   return std::log(previous_error / current_error) /
          std::log(refinement_ratio);
}

void FillDirection(mfem::Vector &direction, int rank,
                   unsigned int seed)
{
   direction.Randomize(static_cast<int>(seed + 104729u * rank));
   direction -= 0.5;
}

void Normalize(mfem::Vector &vector, MPI_Comm comm,
               const char *description)
{
   const mfem::real_t norm = GlobalNorm(vector, comm);
   MFEM_VERIFY(norm > 0.0, description);
   vector /= norm;
}

void RunTaylorTests(ParametricElasticSplineContactProblem &problem,
                    const mfem::Vector &theta,
                    mfem::real_t initial_step,
                    int number_of_steps,
                    MPI_Comm comm,
                    int rank)
{
   mfem::Vector displacement(problem.GetDimU());
   displacement = 0.0;

   mfem::Vector displacement_direction(problem.GetDimU());
   mfem::Vector data_direction(problem.GetDimTheta());
   mfem::Vector multiplier(problem.GetDimM());
   FillDirection(displacement_direction, rank, 271828u);
   FillDirection(data_direction, rank, 161803u);
   FillDirection(multiplier, rank, 314159u);
   Normalize(displacement_direction, comm,
             "Unable to construct a displacement direction.");
   Normalize(data_direction, comm,
             "Unable to construct a spline-data direction.");
   Normalize(multiplier, comm,
             "Unable to construct a constraint multiplier.");

   int evaluation_error = 0;
   mfem::Vector gap;
   problem.g(displacement, theta, gap, evaluation_error);
   MFEM_VERIFY(evaluation_error == 0,
               "The base point does not have valid closest-point "
               "projections.");

   mfem::Vector displacement_jacobian_action(problem.GetDimM());
   mfem::Vector data_jacobian_action(problem.GetDimM());
   mfem::Vector displacement_gradient(problem.GetDimU());
   mfem::Vector data_gradient(problem.GetDimTheta());
   mfem::Operator *displacement_jacobian =
      problem.Ddg(displacement, theta);
   displacement_jacobian->Mult(displacement_direction,
                               displacement_jacobian_action);
   displacement_jacobian->MultTranspose(multiplier,
                                        displacement_gradient);
   mfem::Operator *data_jacobian = problem.Dthg(displacement, theta);
   data_jacobian->Mult(data_direction, data_jacobian_action);
   data_jacobian->MultTranspose(multiplier, data_gradient);

   mfem::Vector first_gap_action(displacement_jacobian_action);
   first_gap_action.Add(1.0, data_jacobian_action);

   mfem::Vector uu_action(problem.GetDimU());
   mfem::Vector mixed_data_action(problem.GetDimU());
   mfem::Vector mixed_displacement_action(problem.GetDimTheta());
   mfem::Vector data_data_action(problem.GetDimTheta());
   mfem::Operator *uu_hessian =
      problem.Dddgl(displacement, multiplier, theta);
   uu_hessian->Mult(displacement_direction, uu_action);
   mfem::Operator *mixed_hessian =
      problem.Ddthgl(displacement, multiplier, theta);
   mixed_hessian->Mult(data_direction, mixed_data_action);
   mixed_hessian->MultTranspose(displacement_direction,
                                mixed_displacement_action);
   mfem::Operator *data_data_hessian =
      problem.Dththgl(displacement, multiplier, theta);
   data_data_hessian->Mult(data_direction, data_data_action);

   mfem::Vector predicted_displacement_gradient_change(uu_action);
   predicted_displacement_gradient_change.Add(1.0, mixed_data_action);
   mfem::Vector predicted_data_gradient_change(
      mixed_displacement_action);
   predicted_data_gradient_change.Add(1.0, data_data_action);

   const mfem::real_t phi =
      GlobalInnerProduct(multiplier, gap, comm);
   const mfem::real_t first_directional_derivative =
      GlobalInnerProduct(multiplier, first_gap_action, comm);
   const mfem::real_t second_directional_derivative =
      GlobalInnerProduct(displacement_direction, uu_action, comm) +
      2.0 * GlobalInnerProduct(displacement_direction,
                               mixed_data_action, comm) +
      GlobalInnerProduct(data_direction, data_data_action, comm);

   // Verify the two row-wise contractions used in the MPEC pressure blocks
   // against the corresponding multiplier-contracted Hessian blocks.
   mfem::Vector second_displacement_direction(problem.GetDimU());
   FillDirection(second_displacement_direction, rank, 57721u);
   Normalize(second_displacement_direction, comm,
             "Unable to construct a second displacement direction.");
   mfem::Vector row_uu_action(problem.GetDimM());
   problem.Dddgl2(displacement, displacement_direction, theta)
      ->Mult(second_displacement_direction, row_uu_action);
   const mfem::real_t row_uu_scalar =
      GlobalInnerProduct(multiplier, row_uu_action, comm);
   mfem::Vector uu_second_action(problem.GetDimU());
   uu_hessian->Mult(second_displacement_direction, uu_second_action);
   const mfem::real_t contracted_uu_scalar =
      GlobalInnerProduct(displacement_direction, uu_second_action, comm);

   mfem::Vector row_mixed_action(problem.GetDimM());
   problem.Dthdgl2(displacement, displacement_direction, theta)
      ->Mult(data_direction, row_mixed_action);
   const mfem::real_t row_mixed_scalar =
      GlobalInnerProduct(multiplier, row_mixed_action, comm);
   const mfem::real_t contracted_mixed_scalar =
      GlobalInnerProduct(displacement_direction,
                         mixed_data_action, comm);
   const mfem::real_t uu_contraction_error =
      std::abs(row_uu_scalar - contracted_uu_scalar) /
      std::max<mfem::real_t>(1.0, std::abs(contracted_uu_scalar));
   const mfem::real_t mixed_contraction_error =
      std::abs(row_mixed_scalar - contracted_mixed_scalar) /
      std::max<mfem::real_t>(1.0, std::abs(contracted_mixed_scalar));

   if (rank == 0)
   {
      std::cout
         << "\nParametric closest-point normal-gap Taylor tests\n"
         << "Both displacement and spline data are perturbed in each row.\n"
         << "Expected orders: R0_g=1, R1_g=2, "
            "R1_grad=2, R2_phi=3.\n\n"
         << std::scientific << std::setprecision(6)
         << std::setw(16) << "h"
         << std::setw(16) << "||R0_g||"
         << std::setw(16) << "order"
         << std::setw(16) << "||R1_g||"
         << std::setw(16) << "order"
         << std::setw(16) << "||R1_grad||"
         << std::setw(16) << "order"
         << std::setw(16) << "|R2_phi|"
         << std::setw(16) << "order" << '\n';
   }

   constexpr mfem::real_t refinement_ratio = 10.0;
   mfem::real_t previous_gap_zeroth = 0.0;
   mfem::real_t previous_gap_first = 0.0;
   mfem::real_t previous_gradient_first = 0.0;
   mfem::real_t previous_phi_second = 0.0;
   for (int level = 0; level < number_of_steps; ++level)
   {
      const mfem::real_t h =
         initial_step * std::pow(refinement_ratio, -level);
      mfem::Vector perturbed_displacement(displacement);
      mfem::Vector perturbed_theta(theta);
      perturbed_displacement.Add(h, displacement_direction);
      perturbed_theta.Add(h, data_direction);

      int perturbed_evaluation_error = 0;
      mfem::Vector perturbed_gap;
      problem.g(perturbed_displacement, perturbed_theta, perturbed_gap,
                perturbed_evaluation_error);
      MFEM_VERIFY(perturbed_evaluation_error == 0,
                  "A Taylor perturbation produced an invalid projection. "
                  "Reduce the initial step.");

      mfem::Vector gap_zeroth_remainder(perturbed_gap);
      gap_zeroth_remainder -= gap;
      const mfem::real_t gap_zeroth_error =
         GlobalNorm(gap_zeroth_remainder, comm);
      mfem::Vector gap_first_remainder(gap_zeroth_remainder);
      gap_first_remainder.Add(-h, first_gap_action);
      const mfem::real_t gap_first_error =
         GlobalNorm(gap_first_remainder, comm);

      mfem::Vector perturbed_displacement_gradient(problem.GetDimU());
      mfem::Vector perturbed_data_gradient(problem.GetDimTheta());
      problem.Ddg(perturbed_displacement, perturbed_theta)
         ->MultTranspose(multiplier, perturbed_displacement_gradient);
      problem.Dthg(perturbed_displacement, perturbed_theta)
         ->MultTranspose(multiplier, perturbed_data_gradient);
      perturbed_displacement_gradient -= displacement_gradient;
      perturbed_displacement_gradient.Add(
         -h, predicted_displacement_gradient_change);
      perturbed_data_gradient -= data_gradient;
      perturbed_data_gradient.Add(-h, predicted_data_gradient_change);
      const mfem::real_t gradient_first_error = ProductNorm(
         perturbed_displacement_gradient, perturbed_data_gradient, comm);

      const mfem::real_t perturbed_phi =
         GlobalInnerProduct(multiplier, perturbed_gap, comm);
      const mfem::real_t phi_second_error = std::abs(
         perturbed_phi - phi - h * first_directional_derivative -
         0.5 * h * h * second_directional_derivative);

      const mfem::real_t gap_zeroth_order = level == 0
         ? std::numeric_limits<mfem::real_t>::quiet_NaN()
         : ObservedOrder(previous_gap_zeroth, gap_zeroth_error,
                         refinement_ratio);
      const mfem::real_t gap_first_order = level == 0
         ? std::numeric_limits<mfem::real_t>::quiet_NaN()
         : ObservedOrder(previous_gap_first, gap_first_error,
                         refinement_ratio);
      const mfem::real_t gradient_first_order = level == 0
         ? std::numeric_limits<mfem::real_t>::quiet_NaN()
         : ObservedOrder(previous_gradient_first, gradient_first_error,
                         refinement_ratio);
      const mfem::real_t phi_second_order = level == 0
         ? std::numeric_limits<mfem::real_t>::quiet_NaN()
         : ObservedOrder(previous_phi_second, phi_second_error,
                         refinement_ratio);

      if (rank == 0)
      {
         std::cout << std::setw(16) << h
                   << std::setw(16) << gap_zeroth_error
                   << std::setw(16) << gap_zeroth_order
                   << std::setw(16) << gap_first_error
                   << std::setw(16) << gap_first_order
                   << std::setw(16) << gradient_first_error
                   << std::setw(16) << gradient_first_order
                   << std::setw(16) << phi_second_error
                   << std::setw(16) << phi_second_order << '\n';
      }

      previous_gap_zeroth = gap_zeroth_error;
      previous_gap_first = gap_first_error;
      previous_gradient_first = gradient_first_error;
      previous_phi_second = phi_second_error;
   }

   // The third-derivative callbacks return the directional derivative, in
   // second_displacement_direction, of each multiplier-contracted Hessian
   // block. Test the full block action and a scalar cubic Taylor expansion.
   mfem::Vector third_uu_u_action(problem.GetDimU());
   mfem::Vector third_mixed_theta_action(problem.GetDimU());
   mfem::Vector third_mixed_u_action(problem.GetDimTheta());
   mfem::Vector third_thetatheta_theta_action(problem.GetDimTheta());
   mfem::Operator *third_uu = problem.Ddddgl3(
      displacement, multiplier, second_displacement_direction, theta);
   third_uu->Mult(displacement_direction, third_uu_u_action);
   mfem::Operator *third_mixed = problem.Ddthdgl3(
      displacement, multiplier, second_displacement_direction, theta);
   third_mixed->Mult(data_direction, third_mixed_theta_action);
   third_mixed->MultTranspose(displacement_direction,
                              third_mixed_u_action);
   mfem::Operator *third_thetatheta = problem.Dththdgl3(
      displacement, multiplier, second_displacement_direction, theta);
   third_thetatheta->Mult(data_direction,
                          third_thetatheta_theta_action);

   mfem::Vector predicted_hessian_u_change(third_uu_u_action);
   predicted_hessian_u_change.Add(1.0, third_mixed_theta_action);
   mfem::Vector predicted_hessian_theta_change(third_mixed_u_action);
   predicted_hessian_theta_change.Add(
      1.0, third_thetatheta_theta_action);

   mfem::Vector scalar_first_action(problem.GetDimM());
   problem.Ddg(displacement, theta)->Mult(
      second_displacement_direction, scalar_first_action);
   const mfem::real_t scalar_first =
      GlobalInnerProduct(multiplier, scalar_first_action, comm);
   mfem::Vector scalar_second_action(problem.GetDimU());
   problem.Dddgl(displacement, multiplier, theta)->Mult(
      second_displacement_direction, scalar_second_action);
   const mfem::real_t scalar_second = GlobalInnerProduct(
      second_displacement_direction, scalar_second_action, comm);
   mfem::Vector scalar_third_action(problem.GetDimU());
   third_uu->Mult(second_displacement_direction, scalar_third_action);
   const mfem::real_t scalar_third = GlobalInnerProduct(
      second_displacement_direction, scalar_third_action, comm);

   // Recompute the base Hessian action because Dddgl above refreshes the
   // internally owned matrix returned by the problem.
   mfem::Vector base_hessian_u(problem.GetDimU());
   mfem::Vector base_hessian_theta(problem.GetDimTheta());
   mfem::Vector base_uu_action(problem.GetDimU());
   mfem::Vector base_mixed_theta_action(problem.GetDimU());
   mfem::Vector base_mixed_u_action(problem.GetDimTheta());
   mfem::Vector base_thetatheta_action(problem.GetDimTheta());
   problem.Dddgl(displacement, multiplier, theta)
      ->Mult(displacement_direction, base_uu_action);
   mfem::Operator *base_mixed =
      problem.Ddthgl(displacement, multiplier, theta);
   base_mixed->Mult(data_direction, base_mixed_theta_action);
   base_mixed->MultTranspose(displacement_direction,
                             base_mixed_u_action);
   problem.Dththgl(displacement, multiplier, theta)
      ->Mult(data_direction, base_thetatheta_action);
   base_hessian_u = base_uu_action;
   base_hessian_u.Add(1.0, base_mixed_theta_action);
   base_hessian_theta = base_mixed_u_action;
   base_hessian_theta.Add(1.0, base_thetatheta_action);

   if (rank == 0)
   {
      std::cout
         << "\nThird-gap-derivative Taylor tests\n"
         << "Only displacement is perturbed; the tested Hessian action "
            "contains uu, u-theta, and theta-theta blocks.\n"
         << "Expected orders: R0_H=1, R1_H=2, R2_phi=3, R3_phi=4.\n\n"
         << std::setw(16) << "h"
         << std::setw(16) << "||R0_H||"
         << std::setw(16) << "order"
         << std::setw(16) << "||R1_H||"
         << std::setw(16) << "order"
         << std::setw(16) << "|R2_phi|"
         << std::setw(16) << "order"
         << std::setw(16) << "|R3_phi|"
         << std::setw(16) << "order" << '\n';
   }

   mfem::real_t previous_hessian_zeroth = 0.0;
   mfem::real_t previous_hessian_first = 0.0;
   mfem::real_t previous_scalar_second = 0.0;
   mfem::real_t previous_scalar_third = 0.0;
   for (int level = 0; level < number_of_steps; ++level)
   {
      const mfem::real_t h =
         initial_step * std::pow(refinement_ratio, -level);
      mfem::Vector perturbed_displacement(displacement);
      perturbed_displacement.Add(h, second_displacement_direction);

      mfem::Vector perturbed_uu_action(problem.GetDimU());
      mfem::Vector perturbed_mixed_theta_action(problem.GetDimU());
      mfem::Vector perturbed_mixed_u_action(problem.GetDimTheta());
      mfem::Vector perturbed_thetatheta_action(problem.GetDimTheta());
      problem.Dddgl(perturbed_displacement, multiplier, theta)
         ->Mult(displacement_direction, perturbed_uu_action);
      mfem::Operator *perturbed_mixed =
         problem.Ddthgl(perturbed_displacement, multiplier, theta);
      perturbed_mixed->Mult(data_direction,
                            perturbed_mixed_theta_action);
      perturbed_mixed->MultTranspose(displacement_direction,
                                     perturbed_mixed_u_action);
      problem.Dththgl(perturbed_displacement, multiplier, theta)
         ->Mult(data_direction, perturbed_thetatheta_action);
      perturbed_uu_action.Add(1.0, perturbed_mixed_theta_action);
      perturbed_mixed_u_action.Add(1.0,
                                   perturbed_thetatheta_action);

      mfem::Vector hessian_u_remainder(perturbed_uu_action);
      mfem::Vector hessian_theta_remainder(perturbed_mixed_u_action);
      hessian_u_remainder -= base_hessian_u;
      hessian_theta_remainder -= base_hessian_theta;
      const mfem::real_t hessian_zeroth_error = ProductNorm(
         hessian_u_remainder, hessian_theta_remainder, comm);
      hessian_u_remainder.Add(-h, predicted_hessian_u_change);
      hessian_theta_remainder.Add(-h,
                                   predicted_hessian_theta_change);
      const mfem::real_t hessian_first_error = ProductNorm(
         hessian_u_remainder, hessian_theta_remainder, comm);

      int perturbed_evaluation_error = 0;
      mfem::Vector perturbed_gap;
      problem.g(perturbed_displacement, theta, perturbed_gap,
                perturbed_evaluation_error);
      MFEM_VERIFY(perturbed_evaluation_error == 0,
                  "A third-derivative perturbation produced an invalid "
                  "projection. Reduce the initial step.");
      const mfem::real_t perturbed_phi =
         GlobalInnerProduct(multiplier, perturbed_gap, comm);
      const mfem::real_t scalar_second_error = std::abs(
         perturbed_phi - phi - h * scalar_first -
         0.5 * h * h * scalar_second);
      const mfem::real_t scalar_third_error = std::abs(
         perturbed_phi - phi - h * scalar_first -
         0.5 * h * h * scalar_second -
         h * h * h * scalar_third / 6.0);

      const mfem::real_t hessian_zeroth_order = level == 0
         ? std::numeric_limits<mfem::real_t>::quiet_NaN()
         : ObservedOrder(previous_hessian_zeroth,
                         hessian_zeroth_error, refinement_ratio);
      const mfem::real_t hessian_first_order = level == 0
         ? std::numeric_limits<mfem::real_t>::quiet_NaN()
         : ObservedOrder(previous_hessian_first,
                         hessian_first_error, refinement_ratio);
      const mfem::real_t scalar_second_order = level == 0
         ? std::numeric_limits<mfem::real_t>::quiet_NaN()
         : ObservedOrder(previous_scalar_second,
                         scalar_second_error, refinement_ratio);
      const mfem::real_t scalar_third_order = level == 0
         ? std::numeric_limits<mfem::real_t>::quiet_NaN()
         : ObservedOrder(previous_scalar_third,
                         scalar_third_error, refinement_ratio);
      if (rank == 0)
      {
         std::cout << std::setw(16) << h
                   << std::setw(16) << hessian_zeroth_error
                   << std::setw(16) << hessian_zeroth_order
                   << std::setw(16) << hessian_first_error
                   << std::setw(16) << hessian_first_order
                   << std::setw(16) << scalar_second_error
                   << std::setw(16) << scalar_second_order
                   << std::setw(16) << scalar_third_error
                   << std::setw(16) << scalar_third_order << '\n';
      }
      previous_hessian_zeroth = hessian_zeroth_error;
      previous_hessian_first = hessian_first_error;
      previous_scalar_second = scalar_second_error;
      previous_scalar_third = scalar_third_error;
   }

   if (rank == 0)
   {
      std::cout
         << "\nAlgebraic contraction consistency:\n"
         << "  displacement contraction relative error = "
         << uu_contraction_error << '\n'
         << "  mixed contraction relative error        = "
         << mixed_contraction_error << '\n'
         << "\nR1_grad checks the complete (u,theta) Hessian action of "
            "lambda^T g.\n"
         << "R1_H checks the displacement-directional derivative of all "
            "three Hessian blocks.\n"
         << "Very small h eventually reaches spline-solve and floating-point "
            "accuracy limits.\n";
   }
}

} // namespace

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   int rank = 0;
   int number_of_ranks = 1;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &number_of_ranks);

   {
      const char *mesh_file = "elastic_body.mesh";
      int parallel_refinements = 0;
      int spline_intervals = 128;
      int number_of_steps = 7;
      mfem::real_t spline_data_fidelity = 0.5;
      mfem::real_t initial_step = 1.0e-1;

      mfem::OptionsParser args(argc, argv);
      args.AddOption(&mesh_file, "-m", "--mesh",
                     "Elastic-body mesh with contact boundary attribute 1.");
      args.AddOption(&parallel_refinements, "-r", "--refinements",
                     "Number of uniform parallel mesh refinements.");
      args.AddOption(&spline_intervals, "-ns", "--spline-intervals",
                     "Number of spline-data intervals on [-2,2].");
      args.AddOption(&spline_data_fidelity, "-p", "--data-fidelity",
                     "Quintic smoothing-spline data-fidelity parameter.");
      args.AddOption(&initial_step, "-dh", "--initial-step",
                     "Largest joint perturbation used in the Taylor test.");
      args.AddOption(&number_of_steps, "-n", "--number-of-steps",
                     "Number of Taylor-test step sizes.");
      args.Parse();
      if (!args.Good())
      {
         if (rank == 0) { args.PrintUsage(std::cout); }
         MPI_Finalize();
         return 1;
      }
      if (parallel_refinements < 0 || spline_intervals < 2 ||
          spline_intervals < number_of_ranks ||
          !(spline_data_fidelity > 0.0 && spline_data_fidelity < 1.0) ||
          initial_step <= 0.0 || number_of_steps < 2)
      {
         if (rank == 0)
         {
            std::cerr << "Invalid derivative-check options.\n";
         }
         MPI_Finalize();
         return 2;
      }
      if (rank == 0) { args.PrintOptions(std::cout); }

      mfem::Mesh serial_body_mesh(mesh_file, 1, 1);
      MFEM_VERIFY(serial_body_mesh.Dimension() == 2,
                  "The elastic body mesh must be two dimensional.");
      mfem::ParMesh body_mesh(MPI_COMM_WORLD, serial_body_mesh);
      for (int level = 0; level < parallel_refinements; ++level)
      {
         body_mesh.UniformRefinement();
      }

      constexpr int dimension = 2;
      constexpr int order = 1;
      mfem::H1_FECollection displacement_collection(order, dimension);
      mfem::ParFiniteElementSpace displacement_space(
         &body_mesh, &displacement_collection, dimension,
         mfem::Ordering::byVDIM);
      const mfem::Array<int> top_marker =
         MakeBoundaryMarker(body_mesh, BoundaryAttribute::TOP);
      const mfem::Array<int> left_marker =
         MakeBoundaryMarker(body_mesh, BoundaryAttribute::LEFT);
      mfem::Array<int> clamped_true_dofs;
      mfem::Array<int> left_x_true_dofs;
      displacement_space.GetEssentialTrueDofs(top_marker,
                                               clamped_true_dofs);
      displacement_space.GetEssentialTrueDofs(left_marker,
                                               left_x_true_dofs, 0);
      mfem::Array<int> essential_true_dofs;
      essential_true_dofs.Append(clamped_true_dofs);
      essential_true_dofs.Append(left_x_true_dofs);
      essential_true_dofs.Sort();
      essential_true_dofs.Unique();

      mfem::ConstantCoefficient lambda_coefficient(1.0);
      mfem::ConstantCoefficient mu_coefficient(1.0);
      mfem::ParBilinearForm elasticity(&displacement_space);
      elasticity.AddDomainIntegrator(
         new mfem::ElasticityIntegrator(lambda_coefficient,
                                        mu_coefficient));
      elasticity.Assemble();
      elasticity.Finalize();
      mfem::Vector zero_body_force(dimension);
      zero_body_force = 0.0;
      mfem::VectorConstantCoefficient zero_body_force_coefficient(
         zero_body_force);
      mfem::ParLinearForm load(&displacement_space);
      load.AddDomainIntegrator(
         new mfem::VectorDomainLFIntegrator(zero_body_force_coefficient));
      load.Assemble();

      constexpr mfem::real_t surface_x_min = -2.0;
      constexpr mfem::real_t surface_x_max = 2.0;
      mfem::Mesh serial_surface_mesh = mfem::Mesh::MakeCartesian1D(
         spline_intervals, surface_x_max - surface_x_min);
      for (int vertex = 0; vertex < serial_surface_mesh.GetNV(); ++vertex)
      {
         serial_surface_mesh.GetVertex(vertex)[0] += surface_x_min;
      }
      mfem::ParMesh surface_mesh(MPI_COMM_WORLD, serial_surface_mesh);
      mfem::H1_FECollection surface_collection(order, 1);
      mfem::ParFiniteElementSpace surface_space(&surface_mesh,
                                                &surface_collection);
      mfem::ParGridFunction surface_data(&surface_space);
      const mfem::real_t pi = std::acos(-1.0);
      mfem::FunctionCoefficient surface_coefficient(
         [pi](const mfem::Vector &x)
         {
            return 0.6 + 0.1 * std::sin(0.5 * pi * x[0]);
         });
      surface_data.ProjectCoefficient(surface_coefficient);
      mfem::Vector theta(surface_space.GetTrueVSize());
      surface_data.GetTrueDofs(theta);

      spline::QuinticSmoothingOptions smoothing_options;
      smoothing_options.data_fidelity = spline_data_fidelity;
      spline::QuinticSmoothingSpline1D surface_spline(
         surface_data, smoothing_options);
      spline::QuinticSmoothingSolveOptions spline_solve_options;
      spline_solve_options.direct_solver = true;
      const spline::QuinticSmoothingSolveResult spline_result =
         surface_spline.Solve(spline_solve_options);
      MFEM_VERIFY(spline_result.converged,
                  "The smoothing-spline solve did not converge.");

      ParametricElasticSplineContactProblem problem(
         displacement_space, elasticity, load, essential_true_dofs,
         BoundaryAttribute::BOTTOM, surface_spline, surface_space, theta,
         surface_x_min, surface_x_max, spline_solve_options,
         spline::ContactGapType::ClosestPointNormal);
      RunTaylorTests(problem, theta, initial_step, number_of_steps,
                     MPI_COMM_WORLD, rank);
   }

   MPI_Finalize();
   return 0;
}
