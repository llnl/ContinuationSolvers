#include "spline_closest_point_1d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace spline
{

void ClosestPointBatchResult::SetSize(int size)
{
   parameter.SetSize(size);
   gap.SetSize(size);
   normal_x.SetSize(size);
   normal_y.SetSize(size);
   tangent_x.SetSize(size);
   tangent_y.SetSize(size);
   projection_hessian.SetSize(size);
   curvature_factor.SetSize(size);
   distance_squared.SetSize(size);
   valid = mfem::Array<int>(size);
   valid = 0;
   iterations = 0;
}

SplineClosestPoint1D::SplineClosestPoint1D(
   const QuinticSmoothingSpline1D &surface,
   mfem::real_t parameter_min,
   mfem::real_t parameter_max,
   const ClosestPointOptions &options)
   : surface_(surface),
     comm_(surface.DataSpace().GetParMesh()->GetComm()),
     parameter_min_(parameter_min),
     parameter_max_(parameter_max),
     options_(options)
{
   MFEM_VERIFY(std::isfinite(parameter_min_) &&
                  std::isfinite(parameter_max_) &&
                  parameter_min_ < parameter_max_,
               "Closest-point projection requires a finite, nonempty "
               "parameter interval.");
   MFEM_VERIFY(options_.absolute_stationarity_tolerance > 0.0 &&
                  options_.relative_stationarity_tolerance >= 0.0 &&
                  options_.minimum_projection_hessian > 0.0 &&
                  options_.maximum_step_fraction > 0.0 &&
                  options_.maximum_step_fraction <= 1.0 &&
                  options_.endpoint_tolerance >= 0.0 &&
                  options_.maximum_iterations > 0,
               "Invalid closest-point projection options.");
}

bool SplineClosestPoint1D::Project(
   const mfem::Vector &point_x,
   const mfem::Vector &point_y,
   ClosestPointBatchResult &result) const
{
   MFEM_VERIFY(point_x.Size() == point_y.Size(),
               "Closest-point coordinate vectors must have equal sizes.");

   const int local_size = point_x.Size();
   result.SetSize(local_size);

   int local_coordinates_are_finite = 1;
   for (int i = 0; i < local_size; ++i)
   {
      if (!std::isfinite(point_x[i]) || !std::isfinite(point_y[i]))
      {
         local_coordinates_are_finite = 0;
      }
      result.parameter[i] =
         std::max(parameter_min_, std::min(parameter_max_, point_x[i]));
   }
   int global_coordinates_are_finite = 0;
   MPI_Allreduce(&local_coordinates_are_finite,
                 &global_coordinates_are_finite, 1, MPI_INT, MPI_MIN, comm_);
   if (global_coordinates_are_finite == 0) { return false; }

   const mfem::real_t interval_length = parameter_max_ - parameter_min_;
   const mfem::real_t maximum_step =
      options_.maximum_step_fraction * interval_length;

   mfem::Array<int> converged(local_size);
   converged = 0;

   for (int iteration = 0;
        iteration < options_.maximum_iterations; ++iteration)
   {
      const mfem::Vector surface_value =
         surface_.Evaluate(result.parameter, 0);
      const mfem::Vector surface_first =
         surface_.Evaluate(result.parameter, 1);
      const mfem::Vector surface_second =
         surface_.Evaluate(result.parameter, 2);

      for (int i = 0; i < local_size; ++i)
      {
         if (converged[i] != 0) { continue; }

         const mfem::real_t slope = surface_first[i];
         const mfem::real_t stationarity =
            result.parameter[i] - point_x[i] +
            (surface_value[i] - point_y[i]) * slope;
         const mfem::real_t projection_hessian =
            1.0 + slope * slope +
            (surface_value[i] - point_y[i]) * surface_second[i];
         const mfem::real_t tolerance =
            options_.absolute_stationarity_tolerance +
            options_.relative_stationarity_tolerance *
               (1.0 + std::abs(point_x[i]) + std::abs(point_y[i]) +
                std::abs(result.parameter[i]));

         if (std::abs(stationarity) <= tolerance &&
             projection_hessian > options_.minimum_projection_hessian)
         {
            converged[i] = 1;
            continue;
         }

         // Newton is used whenever the squared-distance Hessian is
         // sufficiently positive.  Otherwise use the positive metric term
         // 1+S'^2 as a descent denominator.
         const mfem::real_t denominator =
            projection_hessian > options_.minimum_projection_hessian
               ? projection_hessian
               : 1.0 + slope * slope;
         mfem::real_t step = -stationarity / denominator;
         step = std::max(-maximum_step, std::min(maximum_step, step));
         result.parameter[i] = std::max(
            parameter_min_,
            std::min(parameter_max_, result.parameter[i] + step));
      }

      int local_all_converged = 1;
      for (int i = 0; i < local_size; ++i)
      {
         if (converged[i] == 0) { local_all_converged = 0; }
      }
      int global_all_converged = 0;
      MPI_Allreduce(&local_all_converged, &global_all_converged,
                    1, MPI_INT, MPI_MIN, comm_);
      result.iterations = iteration + 1;
      if (global_all_converged != 0) { break; }
   }

   const mfem::Vector surface_value =
      surface_.Evaluate(result.parameter, 0);
   const mfem::Vector surface_first =
      surface_.Evaluate(result.parameter, 1);
   const mfem::Vector surface_second =
      surface_.Evaluate(result.parameter, 2);
   const mfem::real_t left_surface_value =
      surface_.Evaluate(parameter_min_, 0);
   const mfem::real_t right_surface_value =
      surface_.Evaluate(parameter_max_, 0);

   const mfem::real_t endpoint_parameter_tolerance =
      options_.endpoint_tolerance * (1.0 + interval_length);

   int local_success = 1;
   for (int i = 0; i < local_size; ++i)
   {
      const mfem::real_t slope = surface_first[i];
      const mfem::real_t metric = std::sqrt(1.0 + slope * slope);
      const mfem::real_t dx = point_x[i] - result.parameter[i];
      const mfem::real_t dy = point_y[i] - surface_value[i];
      const mfem::real_t stationarity =
         result.parameter[i] - point_x[i] +
         (surface_value[i] - point_y[i]) * slope;
      const mfem::real_t projection_hessian =
         1.0 + slope * slope +
         (surface_value[i] - point_y[i]) * surface_second[i];
      const mfem::real_t tolerance =
         options_.absolute_stationarity_tolerance +
         options_.relative_stationarity_tolerance *
            (1.0 + std::abs(point_x[i]) + std::abs(point_y[i]) +
             std::abs(result.parameter[i]));

      result.tangent_x[i] = 1.0 / metric;
      result.tangent_y[i] = slope / metric;
      result.normal_x[i] = -slope / metric;
      result.normal_y[i] = 1.0 / metric;
      result.gap[i] =
         dx * result.normal_x[i] + dy * result.normal_y[i];
      result.projection_hessian[i] = projection_hessian;
      result.curvature_factor[i] =
         -surface_second[i] / (metric * projection_hessian);
      result.distance_squared[i] = dx * dx + dy * dy;

      const mfem::real_t left_dx = point_x[i] - parameter_min_;
      const mfem::real_t left_dy = point_y[i] - left_surface_value;
      const mfem::real_t left_distance_squared =
         left_dx * left_dx + left_dy * left_dy;
      const mfem::real_t right_dx = point_x[i] - parameter_max_;
      const mfem::real_t right_dy = point_y[i] - right_surface_value;
      const mfem::real_t right_distance_squared =
         right_dx * right_dx + right_dy * right_dy;
      const mfem::real_t distance_tolerance =
         options_.endpoint_tolerance *
         (1.0 + result.distance_squared[i]);

      const bool is_interior =
         result.parameter[i] > parameter_min_ + endpoint_parameter_tolerance &&
         result.parameter[i] < parameter_max_ - endpoint_parameter_tolerance;
      const bool endpoint_is_not_closer =
         left_distance_squared > result.distance_squared[i] +
                                    distance_tolerance &&
         right_distance_squared > result.distance_squared[i] +
                                     distance_tolerance;
      const bool point_is_valid =
         converged[i] != 0 && std::abs(stationarity) <= tolerance &&
         projection_hessian > options_.minimum_projection_hessian &&
         is_interior && endpoint_is_not_closer &&
         std::isfinite(result.curvature_factor[i]);
      result.valid[i] = point_is_valid ? 1 : 0;
      if (!point_is_valid) { local_success = 0; }
   }

   int global_success = 0;
   MPI_Allreduce(&local_success, &global_success,
                 1, MPI_INT, MPI_MIN, comm_);
   return global_success != 0;
}

} // namespace spline
