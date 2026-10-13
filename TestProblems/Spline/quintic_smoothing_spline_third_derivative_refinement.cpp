#include "mfem.hpp"
#include "quintic_smoothing_spline_1d.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   int rank = 0;
   int number_of_ranks = 1;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &number_of_ranks);

   {
      int initial_number_of_intervals = 8;
      int number_of_refinement_levels = 7;
      int fine_number_of_intervals = 8192;
      mfem::real_t data_fidelity = 0.5;
      mfem::real_t interior_boundary_fraction = 0.05;

      mfem::OptionsParser args(argc, argv);
      args.AddOption(&initial_number_of_intervals,
                     "-n0", "--initial-number-of-intervals",
                     "Number of intervals on the coarsest data mesh.");
      args.AddOption(&number_of_refinement_levels,
                     "-l", "--number-of-refinement-levels",
                     "Number of meshes in the refinement sequence.");
      args.AddOption(&fine_number_of_intervals,
                     "-fn", "--fine-number-of-intervals",
                     "Number of intervals on the common visualization "
                     "mesh.");
      args.AddOption(&data_fidelity,
                     "-p", "--data-fidelity",
                     "Quintic smoothing parameter; 0 < p < 1.");
      args.AddOption(
         &interior_boundary_fraction,
         "-ibf", "--interior-boundary-fraction",
         "Fraction of the interval omitted at each endpoint when reporting "
         "the interior maximum third-derivative error.");
      args.Parse();
      if (!args.Good())
      {
         if (rank == 0) { args.PrintUsage(std::cout); }
         MPI_Finalize();
         return 1;
      }

      bool refinement_sequence_is_valid =
         initial_number_of_intervals >= std::max(2, number_of_ranks) &&
         number_of_refinement_levels >= 1;
      int largest_number_of_intervals = initial_number_of_intervals;
      for (int level = 1;
           level < number_of_refinement_levels &&
           refinement_sequence_is_valid;
           ++level)
      {
         if (largest_number_of_intervals >
             std::numeric_limits<int>::max() / 2)
         {
            refinement_sequence_is_valid = false;
         }
         else
         {
            largest_number_of_intervals *= 2;
         }
      }

      const bool options_are_valid =
         refinement_sequence_is_valid &&
         fine_number_of_intervals >= largest_number_of_intervals &&
         fine_number_of_intervals >= number_of_ranks &&
         data_fidelity > 0.0 && data_fidelity < 1.0 &&
         interior_boundary_fraction >= 0.0 &&
         interior_boundary_fraction < 0.5;
      if (!options_are_valid)
      {
         if (rank == 0)
         {
            std::cerr
               << "Invalid refinement options. The coarsest data mesh "
                  "must have at least two intervals and at least one per "
                  "MPI rank; the fine mesh must be at least as refined as "
                  "the last data mesh; 0 < p < 1; and "
                  "0 <= interior-boundary-fraction < 0.5.\n";
         }
         MPI_Finalize();
         return 2;
      }
      if (rank == 0) { args.PrintOptions(std::cout); }

      spline::QuinticSmoothingOptions smoothing_options;
      smoothing_options.data_fidelity = data_fidelity;

      const mfem::real_t wave_number = 3.14;
      const auto exact_function =
         [wave_number](mfem::real_t x)
         {
            return std::sin(wave_number * x);
         };
      const auto exact_third_derivative =
         [wave_number](mfem::real_t x)
         {
            return -wave_number * wave_number * wave_number *
                   std::cos(wave_number * x);
         };

      mfem::FunctionCoefficient function_coefficient(
         [exact_function](const mfem::Vector &x)
         {
            return exact_function(x[0]);
         });
      mfem::FunctionCoefficient coordinate_coefficient(
         [](const mfem::Vector &x) { return x[0]; });
      mfem::FunctionCoefficient third_derivative_coefficient(
         [exact_third_derivative](const mfem::Vector &x)
         {
            return exact_third_derivative(x[0]);
         });

      // Use one common mesh for every field so ParaView comparisons do not
      // include differences in visualization resolution.
      mfem::Mesh serial_fine_mesh =
         mfem::Mesh::MakeCartesian1D(fine_number_of_intervals, 1.0);
      mfem::ParMesh fine_mesh(MPI_COMM_WORLD, serial_fine_mesh);
      mfem::H1_FECollection fine_fec(1, 1);
      mfem::ParFiniteElementSpace fine_fes(&fine_mesh, &fine_fec);

      mfem::ParGridFunction fine_coordinates(&fine_fes);
      fine_coordinates.ProjectCoefficient(coordinate_coefficient);
      mfem::Vector fine_true_coordinates(fine_fes.GetTrueVSize());
      fine_coordinates.GetTrueDofs(fine_true_coordinates);

      mfem::ParGridFunction exact_function_field(&fine_fes);
      exact_function_field.ProjectCoefficient(function_coefficient);
      mfem::ParGridFunction exact_third_derivative_field(&fine_fes);
      exact_third_derivative_field.ProjectCoefficient(
         third_derivative_coefficient);
      mfem::Vector exact_third_derivative_true(fine_fes.GetTrueVSize());
      exact_third_derivative_field.GetTrueDofs(
         exact_third_derivative_true);

      mfem::ParaViewDataCollection paraview(
         "QuinticSmoothingSplineThirdDerivativeRefinement", &fine_mesh);
      paraview.SetPrefixPath("ParaView");
      paraview.SetLevelsOfDetail(1);
      paraview.SetDataFormat(mfem::VTKFormat::BINARY);
      paraview.SetHighOrderOutput(false);
      paraview.RegisterField("exact_function", &exact_function_field);
      paraview.RegisterField("exact_third_derivative",
                             &exact_third_derivative_field);

      // ParaViewDataCollection stores non-owning field pointers. Keep all
      // fields alive until the collection has been saved.
      std::vector<std::unique_ptr<mfem::ParGridFunction>> output_fields;
      output_fields.reserve(
         static_cast<std::size_t>(3 * number_of_refinement_levels));

      int number_of_intervals = initial_number_of_intervals;
      if (rank == 0)
      {
         std::cout
            << std::setprecision(16)
            << "\nQuintic smoothing-spline third-derivative refinement "
               "study (p = "
            << data_fidelity << ")\n"
            << "intervals  global max error in S'''"
               "   interior max error in S'''"
               "        S'''(0)                 S'''(1)"
               "                 residual\n";
      }

      for (int level = 0;
           level < number_of_refinement_levels;
           ++level)
      {
         mfem::Mesh serial_mesh =
            mfem::Mesh::MakeCartesian1D(number_of_intervals, 1.0);
         mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
         mfem::H1_FECollection fec(1, 1);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);
         mfem::ParGridFunction nodal_data(&fes);
         nodal_data.ProjectCoefficient(function_coefficient);

         spline::ParQuinticSmoothingSpline1D smoothing_spline(
            nodal_data, smoothing_options);
         const spline::QuinticSmoothingSolveResult solve =
            smoothing_spline.Solve();

         const mfem::Vector spline_true =
            smoothing_spline.Evaluate(fine_true_coordinates);
         const mfem::Vector third_derivative_true =
            smoothing_spline.Evaluate(fine_true_coordinates, 3);
         mfem::Vector third_derivative_error_true(
            third_derivative_true);
         for (int i = 0; i < third_derivative_error_true.Size(); ++i)
         {
            third_derivative_error_true[i] -=
               exact_third_derivative_true[i];
         }

         mfem::real_t local_global_max_error = 0.0;
         mfem::real_t local_interior_max_error = 0.0;
         for (int i = 0; i < third_derivative_error_true.Size(); ++i)
         {
            const mfem::real_t error =
               std::abs(third_derivative_error_true[i]);
            local_global_max_error =
               std::max(local_global_max_error, error);

            const mfem::real_t x = fine_true_coordinates[i];
            if (x >= interior_boundary_fraction &&
                x <= 1.0 - interior_boundary_fraction)
            {
               local_interior_max_error =
                  std::max(local_interior_max_error, error);
            }
         }

         mfem::real_t global_max_error = 0.0;
         mfem::real_t interior_max_error = 0.0;
         MPI_Reduce(&local_global_max_error, &global_max_error, 1,
                    mfem::MPITypeMap<mfem::real_t>::mpi_type,
                    MPI_MAX, 0, MPI_COMM_WORLD);
         MPI_Reduce(&local_interior_max_error, &interior_max_error, 1,
                    mfem::MPITypeMap<mfem::real_t>::mpi_type,
                    MPI_MAX, 0, MPI_COMM_WORLD);

         const mfem::real_t left_third_derivative =
            smoothing_spline.Evaluate(0.0, 3);
         const mfem::real_t right_third_derivative =
            smoothing_spline.Evaluate(1.0, 3);

         auto spline_field =
            std::make_unique<mfem::ParGridFunction>(&fine_fes);
         spline_field->SetFromTrueDofs(spline_true);
         auto third_derivative_field =
            std::make_unique<mfem::ParGridFunction>(&fine_fes);
         third_derivative_field->SetFromTrueDofs(third_derivative_true);
         auto third_derivative_error_field =
            std::make_unique<mfem::ParGridFunction>(&fine_fes);
         third_derivative_error_field->SetFromTrueDofs(
            third_derivative_error_true);

         std::ostringstream spline_field_name;
         spline_field_name << "smoothing_spline_n"
                           << number_of_intervals;
         std::ostringstream derivative_field_name;
         derivative_field_name << "third_derivative_n"
                               << number_of_intervals;
         std::ostringstream error_field_name;
         error_field_name << "third_derivative_error_n"
                          << number_of_intervals;

         paraview.RegisterField(spline_field_name.str().c_str(),
                                spline_field.get());
         paraview.RegisterField(derivative_field_name.str().c_str(),
                                third_derivative_field.get());
         paraview.RegisterField(error_field_name.str().c_str(),
                                third_derivative_error_field.get());
         output_fields.push_back(std::move(spline_field));
         output_fields.push_back(std::move(third_derivative_field));
         output_fields.push_back(std::move(third_derivative_error_field));

         if (rank == 0)
         {
            std::cout << std::setw(9) << number_of_intervals << "  "
                      << std::setw(25) << global_max_error << "  "
                      << std::setw(27) << interior_max_error << "  "
                      << std::setw(23) << left_third_derivative << "  "
                      << std::setw(23) << right_third_derivative << "  "
                      << std::setw(23) << solve.final_norm << '\n';
         }

         if (level + 1 < number_of_refinement_levels)
         {
            number_of_intervals *= 2;
         }
      }

      paraview.SetCycle(0);
      paraview.SetTime(0.0);
      paraview.Save();

      if (rank == 0)
      {
         std::cout
            << "\nParaView output written under ParaView/"
               "QuinticSmoothingSplineThirdDerivativeRefinement. "
               "Compare exact_third_derivative with the "
               "third_derivative_n* and third_derivative_error_n* "
               "fields.\n";
      }
   }

   MPI_Finalize();
   return 0;
}
