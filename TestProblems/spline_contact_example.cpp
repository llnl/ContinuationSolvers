#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "spline_contact_problem.hpp"
#include "../continuationsolvers/solvers/IPSolver.hpp"


enum BoundaryAttribute
{
   BOTTOM = 1,
   RIGHT = 2,
   TOP = 3,
   LEFT = 4
};

namespace
{

enum RigidBlockBoundaryAttribute
{
   RIGID_TOP = 1,
   RIGID_RIGHT = 2,
   RIGID_BOTTOM = 3,
   RIGID_LEFT = 4
};

mfem::Mesh MakeRigidSurfaceBlock(
   int horizontal_elements,
   int vertical_elements,
   mfem::real_t x_min,
   mfem::real_t x_max,
   mfem::real_t minimum_thickness,
   const mfem::Vector &top_values)
{
   MFEM_VERIFY(horizontal_elements >= 2 && vertical_elements >= 1,
               "The rigid block requires at least two horizontal elements "
               "and one vertical element.");
   MFEM_VERIFY(top_values.Size() == horizontal_elements + 1,
               "Expected one rigid-surface value per horizontal mesh node.");
   MFEM_VERIFY(std::isfinite(x_min) && std::isfinite(x_max) &&
                  x_min < x_max && std::isfinite(minimum_thickness) &&
                  minimum_thickness > 0.0,
               "The rigid-block dimensions must be finite and positive.");

   mfem::real_t minimum_top =
      std::numeric_limits<mfem::real_t>::infinity();
   for (int column = 0; column <= horizontal_elements; ++column)
   {
      MFEM_VERIFY(std::isfinite(top_values[column]),
                  "The rigid surface contains a non-finite value.");
      minimum_top = std::min(minimum_top, top_values[column]);
   }
   const mfem::real_t bottom_y = minimum_top - minimum_thickness;

   constexpr int dimension = 2;
   constexpr int element_attribute = 1;
   const int number_of_vertices =
      (horizontal_elements + 1) * (vertical_elements + 1);
   const int number_of_elements =
      2 * horizontal_elements * vertical_elements;
   const int number_of_boundary_elements =
      2 * horizontal_elements + 2 * vertical_elements;
   mfem::Mesh mesh(dimension, number_of_vertices, number_of_elements,
                   number_of_boundary_elements, dimension);

   std::vector<std::vector<int>> vertex(
      static_cast<std::size_t>(vertical_elements + 1),
      std::vector<int>(static_cast<std::size_t>(horizontal_elements + 1),
                       -1));
   for (int row = 0; row <= vertical_elements; ++row)
   {
      const mfem::real_t eta =
         static_cast<mfem::real_t>(row) / vertical_elements;
      for (int column = 0; column <= horizontal_elements; ++column)
      {
         const mfem::real_t x =
            x_min + (x_max - x_min) *
                       static_cast<mfem::real_t>(column) /
                       horizontal_elements;
         const mfem::real_t y =
            bottom_y + eta * (top_values[column] - bottom_y);
         vertex[static_cast<std::size_t>(row)]
               [static_cast<std::size_t>(column)] = mesh.AddVertex(x, y);
      }
   }

   for (int row = 0; row < vertical_elements; ++row)
   {
      for (int column = 0; column < horizontal_elements; ++column)
      {
         const int lower_left =
            vertex[static_cast<std::size_t>(row)]
                  [static_cast<std::size_t>(column)];
         const int lower_right =
            vertex[static_cast<std::size_t>(row)]
                  [static_cast<std::size_t>(column + 1)];
         const int upper_right =
            vertex[static_cast<std::size_t>(row + 1)]
                  [static_cast<std::size_t>(column + 1)];
         const int upper_left =
            vertex[static_cast<std::size_t>(row + 1)]
                  [static_cast<std::size_t>(column)];
         mesh.AddTriangle(lower_left, lower_right, upper_right,
                          element_attribute);
         mesh.AddTriangle(lower_left, upper_right, upper_left,
                          element_attribute);
      }
   }

   for (int column = 0; column < horizontal_elements; ++column)
   {
      mesh.AddBdrSegment(
         vertex[0][static_cast<std::size_t>(column)],
         vertex[0][static_cast<std::size_t>(column + 1)],
         RIGID_BOTTOM);
   }
   for (int row = 0; row < vertical_elements; ++row)
   {
      mesh.AddBdrSegment(
         vertex[static_cast<std::size_t>(row)]
               [static_cast<std::size_t>(horizontal_elements)],
         vertex[static_cast<std::size_t>(row + 1)]
               [static_cast<std::size_t>(horizontal_elements)],
         RIGID_RIGHT);
   }
   for (int column = horizontal_elements; column > 0; --column)
   {
      mesh.AddBdrSegment(
         vertex[static_cast<std::size_t>(vertical_elements)]
               [static_cast<std::size_t>(column)],
         vertex[static_cast<std::size_t>(vertical_elements)]
               [static_cast<std::size_t>(column - 1)],
         RIGID_TOP);
   }
   for (int row = vertical_elements; row > 0; --row)
   {
      mesh.AddBdrSegment(
         vertex[static_cast<std::size_t>(row)][0],
         vertex[static_cast<std::size_t>(row - 1)][0],
         RIGID_LEFT);
   }

   mesh.FinalizeTriMesh(/*generate_edges=*/1,
                        /*refine=*/1,
                        /*fix_orientation=*/true);
   return mesh;
}

mfem::Array<int> MakeBoundaryMarker(const mfem::ParMesh &mesh,
                                    int attribute)
{
   mfem::Array<int> marker(mesh.bdr_attributes.Max());
   marker = 0;
   marker[attribute - 1] = 1;
   return marker;
}

class FullHessianInteriorPointSolver : public InteriorPointSolver
{
public:
   explicit FullHessianInteriorPointSolver(GeneralOptProblem *problem)
      : InteriorPointSolver(problem)
   {
      fullLagrangianHessian = true;
   }
};

mfem::real_t Global2Norm(const mfem::Vector &v, MPI_Comm comm)
{
   return mfem::GlobalLpNorm(2, v.Norml2(), comm);
}

mfem::real_t GlobalMinimum(const mfem::Vector &v, MPI_Comm comm)
{
   mfem::real_t local_minimum = std::numeric_limits<mfem::real_t>::infinity();
   for (int i = 0; i < v.Size(); ++i)
   {
      local_minimum = std::min(local_minimum, v[i]);
   }
   mfem::real_t global_minimum = 0.0;
   MPI_Allreduce(&local_minimum, &global_minimum, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MIN, comm);
   return global_minimum;
}

void CheckContactDerivatives(ElasticSplineContactProblem &problem,
                             const mfem::Vector &d,
                             mfem::real_t epsilon,
                             MPI_Comm comm,
                             int rank)
{
   mfem::Vector direction(d.Size());
   direction.Randomize(271828);
   direction -= 0.5;
   const mfem::real_t direction_norm = Global2Norm(direction, comm);
   MFEM_VERIFY(direction_norm > 0.0,
               "Unable to construct a derivative-check direction.");
   direction /= direction_norm;

   mfem::Vector d_plus(d);
   mfem::Vector d_minus(d);
   d_plus.Add(epsilon, direction);
   d_minus.Add(-epsilon, direction);

   int eval_err_plus = 0;
   int eval_err_minus = 0;
   mfem::Vector gap_plus, gap_minus;
   problem.g(d_plus, gap_plus, eval_err_plus);
   problem.g(d_minus, gap_minus, eval_err_minus);
   MFEM_VERIFY(eval_err_plus == 0 && eval_err_minus == 0,
               "The finite-difference perturbation left the spline domain.");

   mfem::Vector finite_difference(gap_plus);
   finite_difference -= gap_minus;
   finite_difference /= (2.0 * epsilon);

   mfem::Vector jacobian_action(problem.GetDimM());
   problem.Ddg(d)->Mult(direction, jacobian_action);
   mfem::Vector jacobian_error(jacobian_action);
   jacobian_error -= finite_difference;
   const mfem::real_t jacobian_relative_error =
      Global2Norm(jacobian_error, comm) /
      std::max<mfem::real_t>(1.0, Global2Norm(finite_difference, comm));

   mfem::Vector multiplier(problem.GetDimM());
   multiplier = 1.0;
   mfem::Vector gradient_plus(d.Size());
   mfem::Vector gradient_minus(d.Size());
   problem.Ddg(d_plus)->MultTranspose(multiplier, gradient_plus);
   problem.Ddg(d_minus)->MultTranspose(multiplier, gradient_minus);
   mfem::Vector gradient_finite_difference(gradient_plus);
   gradient_finite_difference -= gradient_minus;
   gradient_finite_difference /= (2.0 * epsilon);

   mfem::Vector hessian_action(d.Size());
   problem.Dddgl(d, multiplier)->Mult(direction, hessian_action);
   mfem::Vector hessian_error(hessian_action);
   hessian_error -= gradient_finite_difference;
   const mfem::real_t hessian_relative_error =
      Global2Norm(hessian_error, comm) /
      std::max<mfem::real_t>(1.0,
                             Global2Norm(gradient_finite_difference, comm));

   if (rank == 0)
   {
      std::cout << "Contact derivative checks (central difference h="
                << epsilon << "):\n"
                << "  ||Dg v - FD(g)|| / max(1,||FD(g)||) = "
                << jacobian_relative_error << '\n'
                << "  ||D2(g.l) v - FD(Dg^T l)|| / "
                   "max(1,||FD(Dg^T l)||) = "
                << hessian_relative_error << '\n';
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
      const char *paraview_prefix = "ParaView";
      int parallel_refinements = 0;
      int spline_intervals = 128;
      int rigid_vertical_elements = 4;
      int maximum_iterations = 40;
      mfem::real_t spline_data_fidelity = 0.5;
      mfem::real_t rigid_minimum_thickness = 0.25;
      mfem::real_t youngs_modulus = 1.0;
      mfem::real_t poisson_ratio = 0.3;
      mfem::real_t body_force_y = -5.0e-2;
      mfem::real_t traction_x = -2.0e-2;
      mfem::real_t traction_y = -8.0e-2;
      mfem::real_t optimization_tolerance = 1.0e-8;
      mfem::real_t derivative_check_step = 1.0e-6;
      bool check_derivatives = true;

      mfem::OptionsParser args(argc, argv);
      args.AddOption(&mesh_file, "-m", "--mesh",
                     "Elastic-body mesh with contact boundary attribute 1.");
      args.AddOption(&parallel_refinements, "-r", "--refinements",
                     "Number of uniform parallel mesh refinements.");
      args.AddOption(&spline_intervals, "-ns", "--spline-intervals",
                     "Number of spline-data intervals on [-2,2].");
      args.AddOption(&spline_data_fidelity, "-p", "--data-fidelity",
                     "Quintic smoothing-spline data-fidelity parameter.");
      args.AddOption(&rigid_vertical_elements, "-nr", "--rigid-layers",
                     "Number of vertical element layers in the rigid block.");
      args.AddOption(&rigid_minimum_thickness,
                     "-rt", "--rigid-thickness",
                     "Minimum vertical thickness of the rigid block.");
      args.AddOption(&youngs_modulus, "-E", "--youngs-modulus",
                     "Young's modulus.");
      args.AddOption(&poisson_ratio, "-nu", "--poisson-ratio",
                     "Poisson ratio for plane strain.");
      args.AddOption(&body_force_y, "-fy", "--body-force-y",
                     "Constant vertical body-force density.");
      args.AddOption(&traction_x, "-tx", "--traction-x",
                     "Horizontal traction on boundary attribute 2.");
      args.AddOption(&traction_y, "-ty", "--traction-y",
                     "Vertical traction on boundary attribute 2.");
      args.AddOption(&maximum_iterations, "-mi", "--maximum-iterations",
                     "Maximum number of interior-point iterations.");
      args.AddOption(&optimization_tolerance, "-tol", "--tolerance",
                     "Interior-point optimality tolerance.");
      args.AddOption(&check_derivatives,
                     "-check", "--check-derivatives",
                     "-no-check", "--no-derivative-checks",
                     "Run gap Jacobian and Hessian finite-difference checks.");
      args.AddOption(&derivative_check_step, "-dh", "--derivative-step",
                     "Central finite-difference step for derivative checks.");
      args.AddOption(&paraview_prefix, "-pv", "--paraview-prefix",
                     "Directory for ParaView output.");
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
          rigid_vertical_elements < 1 || rigid_minimum_thickness <= 0.0 ||
          youngs_modulus <= 0.0 || poisson_ratio <= -1.0 ||
          poisson_ratio >= 0.5 || maximum_iterations <= 0 ||
          optimization_tolerance <= 0.0 || derivative_check_step <= 0.0)
      {
         if (rank == 0)
         {
            std::cerr << "Invalid option. Require refinements >= 0, at least "
                      << "max(2, MPI ranks) spline intervals, 0 < p < 1, "
                      << "positive rigid-block dimensions, "
                      << "E > 0, -1 < nu < 0.5, and positive solver "
                      << "parameters.\n";
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

      // setup forcing terms, body + traction
      mfem::Vector body_force(dimension);
      body_force = 0.0;
      body_force[1] = body_force_y;
      mfem::VectorConstantCoefficient body_force_coefficient(body_force);
      mfem::Vector traction(dimension);
      traction[0] = traction_x;
      traction[1] = traction_y;
      mfem::VectorConstantCoefficient traction_coefficient(traction);
      mfem::Array<int> right_marker =
         MakeBoundaryMarker(body_mesh, BoundaryAttribute::RIGHT);

      mfem::ParLinearForm load(&displacement_space);
      load.AddDomainIntegrator(
         new mfem::VectorDomainLFIntegrator(body_force_coefficient));
      load.AddBoundaryIntegrator(
         new mfem::VectorBoundaryLFIntegrator(traction_coefficient),
         right_marker);
      load.Assemble();

      const mfem::real_t mu =
         youngs_modulus / (2.0 * (1.0 + poisson_ratio));
      const mfem::real_t lambda =
         youngs_modulus * poisson_ratio /
         ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
      mfem::ConstantCoefficient lambda_coefficient(lambda);
      mfem::ConstantCoefficient mu_coefficient(mu);
      mfem::ParBilinearForm elasticity(&displacement_space);
      elasticity.AddDomainIntegrator(
         new mfem::ElasticityIntegrator(lambda_coefficient,
                                        mu_coefficient));
      elasticity.Assemble();
      elasticity.Finalize();

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
      const mfem::real_t pi = M_PI;
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

      // Build a separate visualization mesh for the rigid obstacle.  The top
      // vertices use the solved smoothing spline, so the displayed geometry
      // is exactly the surface used by the contact constraints.
      mfem::Vector rigid_surface_x(spline_intervals + 1);
      for (int column = 0; column <= spline_intervals; ++column)
      {
         rigid_surface_x[column] =
            surface_x_min + (surface_x_max - surface_x_min) *
                               static_cast<mfem::real_t>(column) /
                               spline_intervals;
      }
      const mfem::Vector rigid_surface_y =
         surface_spline.Evaluate(rigid_surface_x);
      mfem::Mesh serial_rigid_block = MakeRigidSurfaceBlock(
         spline_intervals, rigid_vertical_elements,
         surface_x_min, surface_x_max, rigid_minimum_thickness,
         rigid_surface_y);
      mfem::ParMesh rigid_block_mesh(MPI_COMM_WORLD, serial_rigid_block);

      ElasticSplineContactProblem problem(
         displacement_space, elasticity, load, essential_true_dofs,
         BoundaryAttribute::BOTTOM, surface_spline, surface_x_min, surface_x_max);

      mfem::Vector initial_displacement(problem.GetDimU());
      initial_displacement = 0.0;
      int initial_gap_error = 0;
      mfem::Vector initial_gap;
      problem.g(initial_displacement, initial_gap, initial_gap_error);
      MFEM_VERIFY(initial_gap_error == 0,
                  "The undeformed contact nodes are outside the spline domain.");

      if (check_derivatives)
      {
         CheckContactDerivatives(problem, initial_displacement,
                                 derivative_check_step,
                                 MPI_COMM_WORLD, rank);
      }

      mfem::Vector final_displacement(problem.GetDimU());
      final_displacement = 0.0;
      FullHessianInteriorPointSolver optimizer(&problem);
      optimizer.SetTol(optimization_tolerance);
      optimizer.SetMaxIter(maximum_iterations);
      optimizer.SetPrintLevel(2);
      //optimizer.CheckLinearSystemResiduals();
      optimizer.Mult(initial_displacement, final_displacement);

      int final_gap_error = 0;
      mfem::Vector final_gap;
      problem.g(final_displacement, final_gap, final_gap_error);
      MFEM_VERIFY(final_gap_error == 0,
                  "The final contact nodes are outside the spline domain.");

      mfem::Vector full_displacement;
      problem.ProlongateToFullDofs(final_displacement, full_displacement);
      mfem::ParGridFunction displacement(&displacement_space);
      displacement.SetFromTrueDofs(full_displacement);

      mfem::H1_FECollection scalar_body_collection(order, dimension);
      mfem::ParFiniteElementSpace scalar_body_space(
         &body_mesh, &scalar_body_collection);
      mfem::ParGridFunction contact_gap(&scalar_body_space);
      problem.SetPotentialContactValues(final_gap, contact_gap);
      mfem::Vector potential_contact_indicator_values(problem.GetDimM());
      potential_contact_indicator_values = 1.0;
      mfem::ParGridFunction potential_contact_indicator(&scalar_body_space);
      problem.SetPotentialContactValues(potential_contact_indicator_values, potential_contact_indicator);

      mfem::ParaViewDataCollection body_output(
         "ElasticSplineContact", &body_mesh);
      body_output.SetPrefixPath(paraview_prefix);
      body_output.SetLevelsOfDetail(order);
      body_output.SetDataFormat(mfem::VTKFormat::BINARY);
      body_output.SetHighOrderOutput(false);
      body_output.RegisterField("displacement", &displacement);
      body_output.RegisterField("contact_gap", &contact_gap);
      body_output.RegisterField("potential_contact_indicator", &potential_contact_indicator);
      body_output.SetCycle(0);
      body_output.SetTime(0.0);
      body_output.Save();

      body_mesh.SetNodalFESpace(&displacement_space);
      mfem::GridFunction *body_nodes = body_mesh.GetNodes();
      body_nodes->Add(1.0, displacement);
      body_output.SetCycle(1);
      body_output.SetTime(1.0);
      body_output.Save();

      mfem::ParGridFunction surface_coordinate(&surface_space);
      mfem::FunctionCoefficient x_coefficient(
         [](const mfem::Vector &x) { return x[0]; });
      surface_coordinate.ProjectCoefficient(x_coefficient);
      mfem::Vector surface_true_x(surface_space.GetTrueVSize());
      surface_coordinate.GetTrueDofs(surface_true_x);
      const mfem::Vector smoothed_surface_values =
         surface_spline.Evaluate(surface_true_x);
      mfem::ParGridFunction smoothed_surface(&surface_space);
      smoothed_surface.SetFromTrueDofs(smoothed_surface_values);

      mfem::ParaViewDataCollection surface_output(
         "ElasticSplineContactSurface", &surface_mesh);
      surface_output.SetPrefixPath(paraview_prefix);
      surface_output.SetLevelsOfDetail(order);
      surface_output.SetDataFormat(mfem::VTKFormat::BINARY);
      surface_output.SetHighOrderOutput(false);
      surface_output.RegisterField("surface_data", &surface_data);
      surface_output.RegisterField("smoothed_surface", &smoothed_surface);
      surface_output.SetCycle(0);
      surface_output.SetTime(0.0);
      surface_output.Save();

      mfem::H1_FECollection rigid_collection(order, dimension);
      mfem::ParFiniteElementSpace rigid_space(&rigid_block_mesh,
                                              &rigid_collection);
      mfem::ParGridFunction rigid_indicator(&rigid_space);
      rigid_indicator = 1.0;
      mfem::ParaViewDataCollection rigid_output(
         "ElasticSplineContactRigidBlock", &rigid_block_mesh);
      rigid_output.SetPrefixPath(paraview_prefix);
      rigid_output.SetLevelsOfDetail(order);
      rigid_output.SetDataFormat(mfem::VTKFormat::BINARY);
      rigid_output.SetHighOrderOutput(false);
      rigid_output.RegisterField("rigid_block", &rigid_indicator);
      rigid_output.SetCycle(0);
      rigid_output.SetTime(0.0);
      rigid_output.Save();
      rigid_output.SetCycle(1);
      rigid_output.SetTime(1.0);
      rigid_output.Save();

      int objective_error = 0;
      const mfem::real_t final_energy =
         problem.E(final_displacement, objective_error);
      const mfem::real_t initial_minimum_gap =
         GlobalMinimum(initial_gap, MPI_COMM_WORLD);
      const mfem::real_t final_minimum_gap =
         GlobalMinimum(final_gap, MPI_COMM_WORLD);
      if (rank == 0)
      {
         std::cout << "\nFixed quintic-spline contact problem:\n"
                   << "  global free displacement dofs = "
                   << problem.GetDimUGlb() << '\n'
                   << "  global contact constraints = "
                   << problem.GetDimMGlb() << '\n'
                   << "  initial minimum gap = "
                   << initial_minimum_gap << '\n'
                   << "  final minimum gap = "
                   << final_minimum_gap << '\n'
                   << "  final potential energy = " << final_energy << '\n'
                   << "  spline linear residual = "
                   << spline_result.final_norm << '\n';
         if (optimizer.GetConverged())
         {
           std::cout << "interior-point solver converged\n";
         } 
         else
         {
           std::cout << "interior-point solver did not converge\n";
         }
           std::cout << "ParaView output: " << paraview_prefix
                     << "/ElasticSplineContact and " << paraview_prefix
                     << "/ElasticSplineContactRigidBlock\n"
                     << "Spline diagnostic output: " << paraview_prefix
                     << "/ElasticSplineContactSurface\n";
      }
   }

   MPI_Finalize();
   return 0;
}
