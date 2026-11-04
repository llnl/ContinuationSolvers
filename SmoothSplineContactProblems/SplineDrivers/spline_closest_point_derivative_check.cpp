#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include "spline_contact_problem.hpp"

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

mfem::real_t GlobalNorm(const mfem::Vector &vector, MPI_Comm comm)
{
   return std::sqrt(std::max<mfem::real_t>(
      0.0, mfem::InnerProduct(comm, vector, vector)));
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

void RunTaylorTests(ElasticSplineContactProblem &problem,
                    mfem::real_t initial_step,
                    int number_of_steps,
                    MPI_Comm comm,
                    int rank)
{
   mfem::Vector displacement(problem.GetDimU());
   displacement = 0.0;

   mfem::Vector direction(problem.GetDimU());
   FillDirection(direction, rank, 271828u);
   const mfem::real_t direction_norm = GlobalNorm(direction, comm);
   MFEM_VERIFY(direction_norm > 0.0,
               "Unable to construct a displacement direction.");
   direction /= direction_norm;

   mfem::Vector multiplier(problem.GetDimM());
   FillDirection(multiplier, rank, 314159u);
   const mfem::real_t multiplier_norm = GlobalNorm(multiplier, comm);
   MFEM_VERIFY(multiplier_norm > 0.0,
               "Unable to construct a constraint multiplier.");
   multiplier /= multiplier_norm;

   int evaluation_error = 0;
   mfem::Vector gap;
   problem.g(displacement, gap, evaluation_error);
   MFEM_VERIFY(evaluation_error == 0,
               "The base point does not have valid closest-point "
               "projections.");

   mfem::Vector jacobian_action(problem.GetDimM());
   problem.Ddg(displacement)->Mult(direction, jacobian_action);

   mfem::Vector hessian_action(problem.GetDimU());
   problem.Dddgl(displacement, multiplier)->Mult(direction,
                                                  hessian_action);

   const mfem::real_t phi = mfem::InnerProduct(comm, multiplier, gap);
   const mfem::real_t first_directional_derivative =
      mfem::InnerProduct(comm, multiplier, jacobian_action);
   const mfem::real_t second_directional_derivative =
      mfem::InnerProduct(comm, direction, hessian_action);

   if (rank == 0)
   {
      std::cout
         << "\nClosest-point normal-gap directional Taylor tests\n"
         << "The expected asymptotic orders are 1 and 2 for the vector "
            "gap test,\n"
         << "and 2 and 3 for the scalar Hessian test.\n\n"
         << std::scientific << std::setprecision(6)
         << std::setw(12) << "h"
         << std::setw(16) << "||R0_g||"
         << std::setw(10) << "order"
         << std::setw(16) << "||R1_g||"
         << std::setw(10) << "order"
         << std::setw(16) << "|R1_phi|"
         << std::setw(10) << "order"
         << std::setw(16) << "|R2_phi|"
         << std::setw(10) << "order" << '\n';
   }

   constexpr mfem::real_t refinement_ratio = 10.0;
   mfem::real_t previous_gap_zeroth = 0.0;
   mfem::real_t previous_gap_first = 0.0;
   mfem::real_t previous_phi_first = 0.0;
   mfem::real_t previous_phi_second = 0.0;

   for (int level = 0; level < number_of_steps; ++level)
   {
      const mfem::real_t h =
         initial_step * std::pow(refinement_ratio, -level);
      mfem::Vector displaced(displacement);
      displaced.Add(h, direction);

      int perturbed_evaluation_error = 0;
      mfem::Vector perturbed_gap;
      problem.g(displaced, perturbed_gap, perturbed_evaluation_error);
      MFEM_VERIFY(perturbed_evaluation_error == 0,
                  "A Taylor-test perturbation produced an invalid "
                  "closest-point projection. Reduce the initial step.");

      mfem::Vector gap_zeroth_remainder(perturbed_gap);
      gap_zeroth_remainder -= gap;
      const mfem::real_t gap_zeroth_error =
         GlobalNorm(gap_zeroth_remainder, comm);

      mfem::Vector gap_first_remainder(gap_zeroth_remainder);
      gap_first_remainder.Add(-h, jacobian_action);
      const mfem::real_t gap_first_error =
         GlobalNorm(gap_first_remainder, comm);

      const mfem::real_t perturbed_phi =
         mfem::InnerProduct(comm, multiplier, perturbed_gap);
      const mfem::real_t phi_first_error = std::abs(
         perturbed_phi - phi - h * first_directional_derivative);
      const mfem::real_t phi_second_error = std::abs(
         perturbed_phi - phi - h * first_directional_derivative -
         0.5 * h * h * second_directional_derivative);

      const mfem::real_t gap_zeroth_order =
         level == 0 ? std::numeric_limits<mfem::real_t>::quiet_NaN()
                    : ObservedOrder(previous_gap_zeroth,
                                    gap_zeroth_error,
                                    refinement_ratio);
      const mfem::real_t gap_first_order =
         level == 0 ? std::numeric_limits<mfem::real_t>::quiet_NaN()
                    : ObservedOrder(previous_gap_first,
                                    gap_first_error,
                                    refinement_ratio);
      const mfem::real_t phi_first_order =
         level == 0 ? std::numeric_limits<mfem::real_t>::quiet_NaN()
                    : ObservedOrder(previous_phi_first,
                                    phi_first_error,
                                    refinement_ratio);
      const mfem::real_t phi_second_order =
         level == 0 ? std::numeric_limits<mfem::real_t>::quiet_NaN()
                    : ObservedOrder(previous_phi_second,
                                    phi_second_error,
                                    refinement_ratio);

      if (rank == 0)
      {
         std::cout << std::setw(12) << h
                   << std::setw(16) << gap_zeroth_error
                   << std::setw(10) << gap_zeroth_order
                   << std::setw(16) << gap_first_error
                   << std::setw(10) << gap_first_order
                   << std::setw(16) << phi_first_error
                   << std::setw(10) << phi_first_order
                   << std::setw(16) << phi_second_error
                   << std::setw(10) << phi_second_order << '\n';
      }

      previous_gap_zeroth = gap_zeroth_error;
      previous_gap_first = gap_first_error;
      previous_phi_first = phi_first_error;
      previous_phi_second = phi_second_error;
   }

   if (rank == 0)
   {
      std::cout
         << "\nR0_g(h)   = g(d+h v)-g(d)\n"
         << "R1_g(h)   = g(d+h v)-g(d)-h Dg(d)v\n"
         << "R1_phi(h) = phi(d+h v)-phi(d)-h phi'(d)v\n"
         << "R2_phi(h) = R1_phi(h)-0.5 h^2 "
            "v^T D2phi(d) v\n"
         << "where phi(d)=lambda^T g(d). Very small h eventually reaches "
            "roundoff.\n";
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
                     "Largest displacement used in the Taylor test.");
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
            std::cerr << "Invalid option. Require refinements >= 0, at "
                      << "least max(2, MPI ranks) spline intervals, "
                      << "0 < p < 1, initial step > 0, and at least two "
                      << "Taylor steps.\n";
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

      ElasticSplineContactProblem problem(
         displacement_space, elasticity, load, essential_true_dofs,
         BoundaryAttribute::BOTTOM, surface_spline, surface_x_min,
         surface_x_max, spline::ContactGapType::ClosestPointNormal);

      RunTaylorTests(problem, initial_step, number_of_steps,
                     MPI_COMM_WORLD, rank);
   }

   MPI_Finalize();
   return 0;
}
