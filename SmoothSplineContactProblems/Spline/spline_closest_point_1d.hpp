#ifndef SPLINE_CLOSEST_POINT_1D_HPP
#define SPLINE_CLOSEST_POINT_1D_HPP

#include "mfem.hpp"
#include "quintic_smoothing_spline_1d.hpp"

namespace spline
{

enum class ContactGapType
{
   Vertical,
   ClosestPointNormal
};

struct ClosestPointOptions
{
   mfem::real_t absolute_stationarity_tolerance = 1.0e-12;
   mfem::real_t relative_stationarity_tolerance = 1.0e-10;
   mfem::real_t minimum_projection_hessian = 1.0e-12;
   mfem::real_t maximum_step_fraction = 0.25;
   mfem::real_t endpoint_tolerance = 1.0e-10;
   int maximum_iterations = 40;
};

/**
 * Batched closest-point data for points projected onto the graph
 *
 *   r(xi) = (xi, S(xi)).
 *
 * The normal is the upward normal (-S',1)/sqrt(1+S'^2), so the signed gap is
 * positive above the graph.  curvature_factor is the nonzero eigenvalue of
 * the signed-distance Hessian,
 *
 *   Hess_z(g) = curvature_factor * tangent * tangent^T.
 */
struct ClosestPointBatchResult
{
   mfem::Vector parameter;
   mfem::Vector gap;
   mfem::Vector normal_x;
   mfem::Vector normal_y;
   mfem::Vector tangent_x;
   mfem::Vector tangent_y;
   mfem::Vector projection_hessian;
   mfem::Vector curvature_factor;
   mfem::Vector distance_squared;
   mfem::Array<int> valid;
   int iterations = 0;

   void SetSize(int size);
};

/**
 * Collective batched closest-point projection onto a solved parallel
 * quintic smoothing spline.
 *
 * The implementation assumes the query points remain in a tubular
 * neighborhood with a unique interior closest point.  It verifies
 * stationarity, positive local distance curvature, and that neither global
 * spline endpoint is closer.  Failure of any of these checks is reported by
 * returning false and by setting the corresponding valid entry to zero.
 *
 * Every MPI rank must call Project(), but ranks may supply different local
 * numbers of query points.
 */
class SplineClosestPoint1D
{
public:
   SplineClosestPoint1D(
      const QuinticSmoothingSpline1D &surface,
      mfem::real_t parameter_min,
      mfem::real_t parameter_max,
      const ClosestPointOptions &options = ClosestPointOptions());

   bool Project(const mfem::Vector &point_x,
                const mfem::Vector &point_y,
                ClosestPointBatchResult &result) const;

private:
   const QuinticSmoothingSpline1D &surface_;
   MPI_Comm comm_ = MPI_COMM_NULL;
   mfem::real_t parameter_min_ = 0.0;
   mfem::real_t parameter_max_ = 0.0;
   ClosestPointOptions options_;
};

} // namespace spline

#endif // SPLINE_CLOSEST_POINT_1D_HPP
