#include "mfem.hpp"
#include "quartic_spline_1d.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   {
      int number_of_intervals = 32;
      mfem::OptionsParser args(argc, argv);
      args.AddOption(&number_of_intervals, "-n", "--number-of-intervals",
                     "Number of intervals in [0,1].");
      args.Parse();
      if (!args.Good())
      {
         if (rank == 0) { args.PrintUsage(std::cout); }
         MPI_Finalize();
         return 1;
      }
      if (number_of_intervals < 2)
      {
         if (rank == 0)
         {
            std::cerr << "At least two intervals are needed for the "
                         "three-point boundary differences.\n";
         }
         MPI_Finalize();
         return 2;
      }
      if (rank == 0) { args.PrintOptions(std::cout); }

      mfem::Mesh serial_mesh =
         mfem::Mesh::MakeCartesian1D(number_of_intervals, 1.0);
      mfem::ParMesh parallel_mesh(MPI_COMM_WORLD, serial_mesh);

      mfem::H1_FECollection fec(1, 1);
      mfem::ParFiniteElementSpace fes(&parallel_mesh, &fec);
      mfem::ParGridFunction u(&fes);
      mfem::FunctionCoefficient sine(
         [](const mfem::Vector &x) { return std::sin(x[0]); });
      u.ProjectCoefficient(sine);

      // The spline obtains its three-point finite-difference boundary data
      // directly from the distributed nodal values in u.
      spline::ParQuarticSpline1D spline(u);
      const spline::QuarticSolveResult solve = spline.Solve();
      const mfem::real_t sample_x = 0.37;
      const mfem::real_t sample_value = spline.Evaluate(sample_x);

      const std::vector<mfem::real_t> &local_nodes =
         spline.LocalNodeCoordinates();
      mfem::Vector displacements(static_cast<int>(local_nodes.size()));
      for (int i = 0; i < displacements.Size(); ++i)
      {
         const mfem::real_t x = local_nodes[static_cast<std::size_t>(i)];
         displacements[i] = 0.02 * x * (1.0 - x);
      }
      const mfem::Vector displaced_values =
         spline.EvaluateDisplacedNodes(displacements);

      mfem::real_t local_max_error = 0.0;
      for (int e = 0; e < parallel_mesh.GetNE(); ++e)
      {
         mfem::Array<int> vertices;
         parallel_mesh.GetElementVertices(e, vertices);
         const mfem::real_t a = parallel_mesh.GetVertex(vertices[0])[0];
         const mfem::real_t b = parallel_mesh.GetVertex(vertices[1])[0];
         for (int q = 0; q <= 8; ++q)
         {
            const mfem::real_t x = a + (b - a) * q / 8.0;
            local_max_error =
               std::max(local_max_error,
                        std::abs(spline.EvaluateLocalElement(e, x) -
                                 std::sin(x)));
         }
      }

      for (int i = 0; i < displaced_values.Size(); ++i)
      {
         const mfem::real_t x =
            local_nodes[static_cast<std::size_t>(i)] + displacements[i];
         local_max_error =
            std::max(local_max_error,
                     std::abs(displaced_values[i] - std::sin(x)));
      }

      mfem::real_t global_max_error = 0.0;
      MPI_Reduce(&local_max_error, &global_max_error, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MAX, 0,
                 MPI_COMM_WORLD);
      if (rank == 0)
      {
         if (!solve.direct_solver)
	 {
	    std::cout << "GMRES iterations: " << solve.iterations << '\n';
	 }
	 std::cout << "spline(" << sample_x << ") = " << sample_value << '\n';
         std::cout << "sampled max error: " << global_max_error << '\n';
      }
   }

   MPI_Finalize();
   return 0;
}
