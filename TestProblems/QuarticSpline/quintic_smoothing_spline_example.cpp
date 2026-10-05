#include "mfem.hpp"
#include "quintic_smoothing_spline_1d.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   {
      int number_of_intervals = 100;
      mfem::real_t data_fidelity = 0.5;

      mfem::OptionsParser args(argc, argv);
      args.AddOption(&number_of_intervals,
                     "-n", "--number-of-intervals",
                     "Number of intervals in [0,1].");
      args.AddOption(&data_fidelity,
                     "-p", "--data-fidelity",
                     "Smoothing parameter: larger values follow the data "
                     "more closely; 0 < p < 1.");
      args.Parse();
      if (!args.Good())
      {
         if (rank == 0) { args.PrintUsage(std::cout); }
         MPI_Finalize();
         return 1;
      }
      if (number_of_intervals < 2 ||
          !(data_fidelity > 0.0 && data_fidelity < 1.0))
      {
         if (rank == 0)
         {
            std::cerr << "At least two intervals and 0 < p < 1 are "
                         "required.\n";
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

      const mfem::real_t pi = std::acos(-1.0);
      const auto true_function =
         [pi](mfem::real_t x) { return std::sin(2.1 * pi * x); };
      mfem::FunctionCoefficient function_coefficient(
         [true_function](const mfem::Vector &x)
         {
            return true_function(x[0]);
         });
      u.ProjectCoefficient(function_coefficient);

      spline::QuinticSmoothingOptions smoothing_options;
      smoothing_options.data_fidelity = data_fidelity;
      spline::ParQuinticSmoothingSpline1D spline(
         u, smoothing_options);
      const spline::QuinticSmoothingSolveResult solve = spline.Solve();

      mfem::real_t local_max_error = 0.0;
      int pts_per_element = 8;
      mfem::real_t element_dx = 1. / static_cast<mfem::real_t>(pts_per_element); 
      for (int e = 0; e < parallel_mesh.GetNE(); ++e)
      {
         mfem::Array<int> vertices;
         parallel_mesh.GetElementVertices(e, vertices);
         const mfem::real_t a =
            parallel_mesh.GetVertex(vertices[0])[0];
         const mfem::real_t b =
            parallel_mesh.GetVertex(vertices[1])[0];
         for (int q = 0; q <= pts_per_element; ++q)
         {
            const mfem::real_t x =
               a + (b - a) * static_cast<mfem::real_t>(q) * element_dx;
            local_max_error =
               std::max(local_max_error,
                        std::abs(spline.EvaluateLocalElement(e, x) -
                                 true_function(x)));
         }
      }

      mfem::real_t global_max_error = 0.0;
      MPI_Reduce(&local_max_error, &global_max_error, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type,
                 MPI_MAX, 0, MPI_COMM_WORLD);

      const mfem::real_t sample_pt = 0.3713;
      const mfem::real_t sample_value = spline.Evaluate(sample_pt);
      const mfem::real_t left_third_derivative =
         spline.Evaluate(0.0, 3);
      const mfem::real_t right_third_derivative =
         spline.Evaluate(1.0, 3);

      const mfem::Vector data_gradient = spline.EvaluateDataGradient(sample_pt);
      mfem::Vector true_values(fes.GetTrueVSize());
      u.GetTrueDofs(true_values);
      mfem::real_t local_gradient_pairing = 0.0;
      for (int i = 0; i < true_values.Size(); ++i)
      {
         local_gradient_pairing +=
            data_gradient[i] * true_values[i];
      }
      mfem::real_t gradient_pairing = 0.0;
      MPI_Allreduce(&local_gradient_pairing, &gradient_pairing, 1,
                    mfem::MPITypeMap<mfem::real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);

      if (rank == 0)
      {
         std::cout << "spline(" << sample_pt << ") = "
                   << sample_value << '\n'
                   << "sampled max error = " << global_max_error << '\n'
                   << "S'''(0) = " << left_third_derivative << '\n'
                   << "S'''(1) = " << right_third_derivative << '\n'
                   << "|grad_y S . y - S| = "
                   << std::abs(gradient_pairing - sample_value) << '\n'
                   << "linear system residual = " << solve.final_norm << '\n';
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
        ufine.GetTrueDofs(Xnodes); // get spatial coordinates of nodes

        auto Unodes = spline.Evaluate(Xnodes);  // evaluate the spline at the nodes
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

        ufine.ProjectCoefficient(function_coefficient);
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
