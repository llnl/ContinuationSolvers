#include "mfem.hpp"
#include "quartic_spline_1d.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
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
      bool minimum_curvature = true;
      bool minimum_curvature_and_variation = false;
      mfem::real_t curvature_variation_length_fraction = 1.0e-2;

      mfem::OptionsParser args(argc, argv);
      args.AddOption(&initial_number_of_intervals,
                     "-n0", "--initial-number-of-intervals",
                     "Number of intervals on the coarsest mesh.");
      args.AddOption(&number_of_refinement_levels,
                     "-l", "--number-of-refinement-levels",
                     "Number of meshes in the refinement sequence.");
      args.AddOption(&fine_number_of_intervals,
                     "-fn", "--fine-number-of-intervals",
                     "Number of intervals on the common visualization "
                     "mesh.");
      args.AddOption(&minimum_curvature,
                     "-mc", "--minimum-curvature",
                     "-fd", "--finite-difference-boundary-conditions",
                     "Use minimum-curvature interpolation instead of "
                     "finite-difference endpoint conditions.");
      args.AddOption(
         &minimum_curvature_and_variation,
         "-mcv", "--minimum-curvature-and-variation",
         "-no-mcv", "--no-minimum-curvature-and-variation",
         "Minimize curvature and curvature variation while interpolating.");
      args.AddOption(
         &curvature_variation_length_fraction,
         "-cvlf", "--curvature-variation-length-fraction",
         "Length scale for the curvature-variation penalty as a fraction "
         "of the global interpolation interval.");
      args.Parse();
      if (!args.Good())
      {
         if (rank == 0) { args.PrintUsage(std::cout); }
         MPI_Finalize();
         return 1;
      }

      bool refinement_sequence_is_valid =
         initial_number_of_intervals >= number_of_ranks &&
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

      const bool finite_difference_mode =
         !minimum_curvature && !minimum_curvature_and_variation;
      const bool options_are_valid =
         refinement_sequence_is_valid &&
         (!finite_difference_mode || initial_number_of_intervals >= 2) &&
         fine_number_of_intervals >= largest_number_of_intervals &&
         fine_number_of_intervals >= number_of_ranks &&
         curvature_variation_length_fraction >= 0.0 &&
         (!minimum_curvature_and_variation ||
          curvature_variation_length_fraction > 0.0);
      if (!options_are_valid)
      {
         if (rank == 0)
         {
            std::cerr
               << "Invalid refinement options. The coarsest and fine "
                  "meshes must have at least one interval per MPI rank; "
                  "the finite-difference mode requires at least two "
                  "intervals; the fine mesh must be at least as refined as "
                  "the last comparison mesh; and -cvlf must be positive "
                  "when -mcv is selected.\n";
         }
         MPI_Finalize();
         return 2;
      }
      if (rank == 0) { args.PrintOptions(std::cout); }

      using InterpolationMode = spline::QuarticInterpolationMode;
      spline::QuarticInterpolationOptions interpolation_options;
      interpolation_options.mode =
         minimum_curvature_and_variation
            ? InterpolationMode::MinimumCurvatureAndVariation
            : minimum_curvature
                 ? InterpolationMode::MinimumCurvature
                 : InterpolationMode::FiniteDifferenceBoundaryConditions;
      interpolation_options.curvature_variation_length_fraction =
         curvature_variation_length_fraction;

      const char *mode_name =
         minimum_curvature_and_variation
            ? "minimum_curvature_and_variation"
            : minimum_curvature
                 ? "minimum_curvature"
                 : "finite_difference";

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

      // Every refinement level is sampled on this one fixed mesh, so the
      // resulting ParaView fields can be compared point by point.
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

      std::ostringstream collection_name;
      collection_name << "QuarticSplineThirdDerivativeRefinement_"
                      << mode_name;
      mfem::ParaViewDataCollection paraview(
         collection_name.str().c_str(), &fine_mesh);
      paraview.SetPrefixPath("ParaView");
      paraview.SetLevelsOfDetail(1);
      paraview.SetDataFormat(mfem::VTKFormat::BINARY);
      paraview.SetHighOrderOutput(false);
      paraview.RegisterField("exact_function", &exact_function_field);
      paraview.RegisterField("exact_third_derivative",
                             &exact_third_derivative_field);

      // ParaViewDataCollection stores non-owning field pointers. Keep every
      // refinement field alive until after Save().
      std::vector<std::unique_ptr<mfem::ParGridFunction>> output_fields;
      output_fields.reserve(
         static_cast<std::size_t>(3 * number_of_refinement_levels));

      int number_of_intervals = initial_number_of_intervals;
      if (rank == 0)
      {
         std::cout << std::setprecision(16)
                   << "\nThird-derivative refinement study ("
                   << mode_name << ")\n"
                   << "intervals  max nodal error in S'''"
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

         spline::ParQuarticSpline1D quartic_spline(
            nodal_data, interpolation_options);
         const spline::QuarticSolveResult solve = quartic_spline.Solve();

         const mfem::Vector spline_true =
            quartic_spline.Evaluate(fine_true_coordinates);
         const mfem::Vector third_derivative_true =
            quartic_spline.Evaluate(fine_true_coordinates, 3);
         mfem::Vector third_derivative_error_true(
            third_derivative_true);
         for (int i = 0; i < third_derivative_error_true.Size(); ++i)
         {
            third_derivative_error_true[i] -=
               exact_third_derivative_true[i];
         }

         mfem::real_t local_max_third_derivative_error = 0.0;
         for (int i = 0; i < third_derivative_error_true.Size(); ++i)
         {
            local_max_third_derivative_error =
               std::max(local_max_third_derivative_error,
                        std::abs(third_derivative_error_true[i]));
         }
         mfem::real_t global_max_third_derivative_error = 0.0;
         MPI_Reduce(&local_max_third_derivative_error,
                    &global_max_third_derivative_error, 1,
                    mfem::MPITypeMap<mfem::real_t>::mpi_type,
                    MPI_MAX, 0, MPI_COMM_WORLD);

         const mfem::real_t left_third_derivative =
            quartic_spline.Evaluate(0.0, 3);
         const mfem::real_t right_third_derivative =
            quartic_spline.Evaluate(1.0, 3);

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
         spline_field_name << "spline_n" << number_of_intervals;
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
                      << std::setw(25)
                      << global_max_third_derivative_error << "  "
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
            << collection_name.str()
            << ". Compare exact_third_derivative with the "
               "third_derivative_n* and third_derivative_error_n* "
               "fields.\n";
      }
   }

   MPI_Finalize();
   return 0;
}
