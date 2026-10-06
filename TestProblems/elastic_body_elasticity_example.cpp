#include "mfem.hpp"

#include <cmath>
#include <iostream>

namespace
{

enum BoundaryAttribute
{
   FREE_ARC = 1,
   RIGHT = 2,
   CLAMPED_TOP = 3,
   LEFT = 4
};

mfem::Array<int> MakeBoundaryMarker(const mfem::ParMesh &mesh,
                                    int first_attribute,
                                    int second_attribute = -1)
{
   mfem::Array<int> marker(mesh.bdr_attributes.Max());
   marker = 0;
   marker[first_attribute - 1] = 1;
   if (second_attribute > 0) { marker[second_attribute - 1] = 1; }
   return marker;
}

bool HasBoundaryAttribute(const mfem::Mesh &mesh, int attribute)
{
   return mesh.bdr_attributes.Find(attribute) >= 0;
}

} // namespace

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   {
   const char *mesh_file = "elastic_body.mesh";
   const char *paraview_prefix = "ParaView";
   mfem::real_t youngs_modulus = 1.0;
   mfem::real_t poisson_ratio = 0.3;
   mfem::real_t body_force_y = -5.0e-2;
   mfem::real_t traction_x = -2.0e-2;
   mfem::real_t traction_y = -8.0e-2;
   int parallel_refinements = 0;

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Input mesh produced by elastic_body_mesh_example.");
   args.AddOption(&youngs_modulus, "-E", "--youngs-modulus",
                  "Young's modulus.");
   args.AddOption(&poisson_ratio, "-nu", "--poisson-ratio",
                  "Poisson ratio for the plane-strain material model.");
   args.AddOption(&body_force_y, "-fy", "--body-force-y",
                  "Constant vertical body-force density.");
   args.AddOption(&traction_x, "-tx", "--traction-x",
                  "Horizontal traction on boundary attribute 2.");
   args.AddOption(&traction_y, "-ty", "--traction-y",
                  "Vertical traction on boundary attribute 2.");
   args.AddOption(&parallel_refinements, "-r", "--refinements",
                  "Number of uniform parallel mesh refinements.");
   args.AddOption(&paraview_prefix, "-p", "--paraview-prefix",
                  "Directory for parallel ParaView output.");
   args.Parse();

   if (!args.Good())
   {
      if (rank == 0) { args.PrintUsage(std::cout); }
      MPI_Finalize();
      return 1;
   }

   if (youngs_modulus <= 0.0 || poisson_ratio <= -1.0 ||
       poisson_ratio >= 0.5 || parallel_refinements < 0)
   {
      if (rank == 0)
      {
         std::cerr << "Expected E > 0, -1 < nu < 0.5, and "
                   << "refinements >= 0.\n";
      }
      MPI_Finalize();
      return 2;
   }

   if (rank == 0) { args.PrintOptions(std::cout); }

   mfem::Mesh serial_mesh(mesh_file, /*generate_edges=*/1,
                          /*refine=*/1);
   if (serial_mesh.Dimension() != 2)
   {
      if (rank == 0)
      {
         std::cerr << "This example requires a two-dimensional mesh.\n";
      }
      MPI_Finalize();
      return 3;
   }

   for (int attribute = FREE_ARC; attribute <= LEFT; ++attribute)
   {
      if (!HasBoundaryAttribute(serial_mesh, attribute))
      {
         if (rank == 0)
         {
            std::cerr << "The input mesh is missing boundary attribute "
                      << attribute << ".\n";
         }
         MPI_Finalize();
         return 4;
      }
   }

   mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   for (int level = 0; level < parallel_refinements; ++level)
   {
      mesh.UniformRefinement();
   }

   constexpr int dimension = 2;
   constexpr int order = 1;
   mfem::H1_FECollection collection(order, dimension);
   mfem::ParFiniteElementSpace displacement_space(
      &mesh, &collection, dimension, mfem::Ordering::byVDIM);

   // Attribute 3: u_x = u_y = 0 (clamped).
   const mfem::Array<int> clamped_marker =
      MakeBoundaryMarker(mesh, CLAMPED_TOP);
   mfem::Array<int> clamped_true_dofs;
   displacement_space.GetEssentialTrueDofs(clamped_marker,
                                           clamped_true_dofs);

   // Attribute 4: u_x = 0; u_y remains free.  Attribute 2 is not included
   // because both of its displacement components are subject to traction.
   const mfem::Array<int> left_marker = MakeBoundaryMarker(mesh, LEFT);
   mfem::Array<int> left_x_true_dofs;
   displacement_space.GetEssentialTrueDofs(left_marker,
                                           left_x_true_dofs,
                                           /*component=*/0);

   mfem::Array<int> essential_true_dofs;
   essential_true_dofs.Append(clamped_true_dofs);
   essential_true_dofs.Append(left_x_true_dofs);
   essential_true_dofs.Sort();
   essential_true_dofs.Unique();

   // Optional constant body force.  Its default value is zero so the load in
   // this example comes from the prescribed traction below.
   mfem::Vector force(dimension);
   force = 0.0;
   force[1] = body_force_y;
   mfem::VectorConstantCoefficient force_coefficient(force);

   // Attribute 2 has t = (traction_x, traction_y).  Negative components push
   // the right boundary to the left and downward.  Attribute 1 remains a
   // homogeneous natural boundary because no integrator is added on it.
   mfem::Vector traction(dimension);
   traction[0] = traction_x;
   traction[1] = traction_y;
   mfem::VectorConstantCoefficient traction_coefficient(traction);
   mfem::Array<int> traction_marker = MakeBoundaryMarker(mesh, RIGHT);

   mfem::ParLinearForm load(&displacement_space);
   load.AddDomainIntegrator(
      new mfem::VectorDomainLFIntegrator(force_coefficient));
   load.AddBoundaryIntegrator(
      new mfem::VectorBoundaryLFIntegrator(traction_coefficient),
      traction_marker);
   load.Assemble();

   // Plane-strain Lamé parameters.
   const mfem::real_t mu =
      youngs_modulus / (2.0 * (1.0 + poisson_ratio));
   const mfem::real_t lambda =
      youngs_modulus * poisson_ratio /
      ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
   mfem::ConstantCoefficient lambda_coefficient(lambda);
   mfem::ConstantCoefficient mu_coefficient(mu);

   mfem::ParBilinearForm elasticity(&displacement_space);
   elasticity.AddDomainIntegrator(
      new mfem::ElasticityIntegrator(lambda_coefficient, mu_coefficient));
   elasticity.Assemble();

   mfem::ParGridFunction displacement(&displacement_space);
   displacement = 0.0;

   mfem::HypreParMatrix system_matrix;
   mfem::Vector solution, right_hand_side;
   elasticity.FormLinearSystem(essential_true_dofs, displacement, load,
                               system_matrix, solution, right_hand_side);

   if (rank == 0)
   {
      std::cout << "Global displacement true dofs: "
                << displacement_space.GlobalTrueVSize() << '\n'
                << "Linear-system size: "
                << system_matrix.GetGlobalNumRows() << '\n'
                << "Boundary conditions:\n"
                << "  attribute 1: traction-free\n"
                << "  attribute 2: traction = (" << traction_x << ", "
                << traction_y << ")\n"
                << "  attribute 3: u_x = u_y = 0\n";
      std::cout << "  attribute 4: u_x = 0, u_y free\n";
   }

   mfem::HypreBoomerAMG preconditioner(system_matrix);
   preconditioner.SetSystemsOptions(dimension, /*order_bynodes=*/false);
   preconditioner.SetPrintLevel(0);

   mfem::HyprePCG solver(system_matrix);
   solver.SetTol(1.0e-12);
   solver.SetMaxIter(500);
   solver.SetPrintLevel(rank == 0 ? 2 : 0);
   solver.SetPreconditioner(preconditioner);
   solver.Mult(right_hand_side, solution);

   elasticity.RecoverFEMSolution(solution, load, displacement);

   mfem::ParaViewDataCollection output("ElasticBodyElasticity", &mesh);
   output.SetPrefixPath(paraview_prefix);
   output.SetCycle(0);
   output.SetTime(0.0);
   output.SetLevelsOfDetail(order);
   output.SetDataFormat(mfem::VTKFormat::BINARY);
   output.SetHighOrderOutput(false);
   output.RegisterField("displacement", &displacement);
   output.Save();


   mesh.SetNodalFESpace(&displacement_space);
   mfem::GridFunction *mesh_nodes = mesh.GetNodes();
   mesh_nodes->Add(1.0, displacement);
   

   output.SetCycle(1);
   output.SetTime(1.0);
   output.Save();
   
   if (rank == 0)
   {
      std::cout << "Wrote parallel ParaView collection under "
                << paraview_prefix << "/ElasticBodyElasticity.\n";
   }

   }

   // All MFEM/HYPRE objects have gone out of scope before MPI is finalized.
   MPI_Finalize();
   return 0;
}
