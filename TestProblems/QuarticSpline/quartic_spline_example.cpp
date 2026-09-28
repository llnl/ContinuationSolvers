#include "mfem.hpp"
#include "quartic_spline_1d.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
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
      auto true_function = [](const double x) { return std::sin(3.14*x); };      
      mfem::FunctionCoefficient sine(
         [true_function](const mfem::Vector &x) { return true_function(x[0]); });
      u.ProjectCoefficient(sine);
      



      // The spline obtains its three-point finite-difference boundary data
      // directly from the distributed nodal values in u.
      spline::ParQuarticSpline1D spline(u);
      const spline::QuarticSolveResult solve = spline.Solve();
      const mfem::real_t sample_x = 0.3713;
      const mfem::real_t sample_value = spline.Evaluate(sample_x);

      // The callback returns the locally owned true-DOF portion of
      // grad_y S(sample_x). Because the spline is linear in y, its pairing
      // with the original interpolation data must reproduce S(sample_x).
      const auto data_gradient_callback =
         spline.MakeDataGradientCallback();
      const mfem::Vector sample_data_gradient =
         data_gradient_callback(sample_x);
      mfem::Vector true_values(fes.GetTrueVSize());
      u.GetTrueDofs(true_values);
      mfem::real_t local_gradient_pairing = 0.0;
      for (int i = 0; i < true_values.Size(); ++i)
      {
         local_gradient_pairing +=
            sample_data_gradient[i] * true_values[i];
      }
      mfem::real_t gradient_pairing = 0.0;
      MPI_Allreduce(&local_gradient_pairing, &gradient_pairing, 1,
                    mfem::MPITypeMap<mfem::real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);

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
      int n_samples_per_element = 8;
      // for each element we will generate n_samples_per_element 
      // for which we will evaluate the difference between the spline
      // and the true function
      // determine the max discrepancy
      for (int e = 0; e < parallel_mesh.GetNE(); ++e)
      {
         mfem::Array<int> vertices;
         parallel_mesh.GetElementVertices(e, vertices);
         const mfem::real_t a = parallel_mesh.GetVertex(vertices[0])[0];
         const mfem::real_t b = parallel_mesh.GetVertex(vertices[1])[0];
         std::default_random_engine generator;
         std::uniform_real_distribution<double> distribution(a, b);

         for (int q = 0; q <= n_samples_per_element; ++q)
         {
            const mfem::real_t element_sample_pt = distribution(generator);
            local_max_error =
               std::max(local_max_error,
                        std::abs(spline.EvaluateLocalElement(e, element_sample_pt) -
                                 true_function(element_sample_pt)));
         }
      }

      for (int i = 0; i < displaced_values.Size(); ++i)
      {
         const mfem::real_t x =
            local_nodes[static_cast<std::size_t>(i)] + displacements[i];
         local_max_error =
            std::max(local_max_error,
                     std::abs(displaced_values[i] - true_function(x)));
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
         std::cout << "sin(" << sample_x << ") = " << true_function(sample_x) << "\n";
         std::cout << "|grad_y S . y - S| = "
                   << std::abs(gradient_pairing - sample_value) << '\n';
         std::cout << "sampled max error: " << global_max_error << '\n';
      }


     // continuity check using Evaluate method
     {
        mfem::real_t x = 1. / number_of_intervals;
        for (int d = 0; d <= 3; d++)
        {
           std::cout << "derivative order " << d << " continuity check about a knot\n";
           mfem::real_t eps = 0.5 * x;
           for (int i = 0; i < 20; i++)
           {
              mfem::real_t ffwd = spline.Evaluate(x + eps, d);
              mfem::real_t fbkwd = spline.Evaluate(x - eps, d);
              double err = std::abs(ffwd - fbkwd);
              if (rank == 0)
              {
                 std::cout << "|f^(" << d <<")(" << x + eps << ") - f^(" << d << ")(" << x - eps << ")| = " << err << ", dx = " << 2.0 * eps << std::endl;
              }
              eps *= 0.5;
           }
           std::cout << "\n\n";
                
        }



     }

     // do some plotting
     {
        mfem::Mesh serial_fine_mesh =
           mfem::Mesh::MakeCartesian1D(10 * number_of_intervals, 1.0);
        mfem::ParMesh parallel_fine_mesh(MPI_COMM_WORLD, serial_fine_mesh);
        mfem::ParFiniteElementSpace fes_fine(&parallel_fine_mesh, &fec);
        mfem::ParGridFunction ufine(&fes_fine);
        mfem::FunctionCoefficient nodal_values( [](const mfem::Vector & x) {return x[0]; });
        ufine.ProjectCoefficient(nodal_values);
        
        mfem::Vector Xnodes(fes_fine.GetTrueVSize());
        ufine.GetTrueDofs(Xnodes);

        auto Unodes = spline.Evaluate(Xnodes);
        ufine.SetFromTrueDofs(Unodes);
        mfem::ParaViewDataCollection paraview_dc("SplineExample", &parallel_fine_mesh);
        paraview_dc.SetPrefixPath("ParaView");
        paraview_dc.SetLevelsOfDetail(1);
        paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(false);
        paraview_dc.RegisterField("spline", &ufine);
        paraview_dc.SetCycle(0);
        paraview_dc.SetTime(0.0);
        paraview_dc.Save();

        ufine.ProjectCoefficient(sine);
        paraview_dc.SetCycle(1);
        paraview_dc.SetTime(1.0);
        paraview_dc.Save();

        for (int d = 1; d <= 3; d++)
        {
           auto dUnodes = spline.Evaluate(Xnodes, d);
           ufine.SetFromTrueDofs(dUnodes);
           paraview_dc.SetCycle(d+1);
           paraview_dc.SetTime((double) (d+1));
           paraview_dc.Save();
        }



     }


   }




   MPI_Finalize();
   return 0;
}
