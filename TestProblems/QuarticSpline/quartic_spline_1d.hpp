#ifndef QUARTIC_SPLINE_1D_HPP
#define QUARTIC_SPLINE_1D_HPP

#include "mfem.hpp"
#include "linear_1d_node_data.hpp"

#include <array>
#include <memory>
#include <vector>

namespace spline
{

struct QuarticSolveOptions
{
   mfem::real_t relative_tolerance = 1.0e-12;
   mfem::real_t absolute_tolerance = 0.0;
   int maximum_iterations = 1000;
   int krylov_dimension = 100;
   int print_level = 0;
   bool require_convergence = true;
   bool direct_solver = true;
};

struct QuarticSolveResult
{
   bool converged = false;
   int iterations = 0;
   mfem::real_t initial_norm = -1.0;
   mfem::real_t final_norm = -1.0;
   bool direct_solver = true;
};

/**
 * Parallel C3 quartic interpolating spline for a scalar P1 H1
 * mfem::ParGridFunction on a conforming one-dimensional mesh.
 *
 * On each interval [a,b], with h_e = b-a, the polynomial is
 *
 *   S_e(x) = sum_{j=0}^4 a_{e,j} ((x-b)/h_e)^j.
 *
 * Thus a_{e,j} = h_e^j c_{e,j}, where c_{e,j} denotes the corresponding
 * coefficient in the unscaled physical-coordinate basis.
 *
 * There are five unknowns and five rows per mesh interval. Both rows and
 * columns are owned by the MPI rank that owns the interval. No auxiliary
 * finite element space is constructed: ParMesh supplies rank-contiguous
 * global element numbers and Hypre-compatible partition arrays directly.
 * Rows imposing an order-d derivative condition are scaled by h_e^d, where
 * h_e is the width of the interval that owns the row. This scaling reduces
 * dimensional imbalance without changing the represented spline.
 *
 * Construction and Solve() are collective on the ParMesh communicator.
 * The input ParGridFunction and its mesh/space must outlive this object.
 */
class ParQuarticSpline1D
{
public:
   static constexpr int order = 4;
   static constexpr int coefficients_per_interval = order + 1;

   explicit ParQuarticSpline1D(
      const mfem::ParGridFunction &input,
      mfem::real_t coordinate_tolerance = 0.0);

   ParQuarticSpline1D(const ParQuarticSpline1D &) = delete;
   ParQuarticSpline1D &operator=(const ParQuarticSpline1D &) = delete;
   ParQuarticSpline1D(ParQuarticSpline1D &&) = delete;
   ParQuarticSpline1D &operator=(ParQuarticSpline1D &&) = delete;
   ~ParQuarticSpline1D() = default;

   /// Assemble A and b. The constructor calls this once automatically.
   void Assemble();

   /// Solve A a = b for the normalized coefficients with restarted GMRES.
   QuarticSolveResult Solve(
      const QuarticSolveOptions &options = QuarticSolveOptions());

   const mfem::HypreParMatrix &SystemMatrix() const;
   /**
    * Matrix B in b = B y, where y is the input GridFunction's distributed
    * true-DOF vector. B includes the endpoint interpolation rows and all
    * three finite-difference boundary conditions.
    *
    * For an adjoint lambda satisfying A^T lambda = dF/da, the derivative
    * with respect to the nodal data is B^T lambda.
    */
   const mfem::HypreParMatrix &RightHandSideDataJacobian() const;
   const mfem::HypreParVector &RightHandSide() const;
   /// Distributed vector of normalized coefficients a_{e,j}.
   const mfem::HypreParVector &CoefficientVector() const;

   HYPRE_BigInt GlobalNumberOfIntervals() const
   {
      return global_number_of_intervals_;
   }

   int LocalNumberOfIntervals() const
   {
      return static_cast<int>(intervals_.size());
   }

   /// Normalized coefficients a_0,...,a_4 for a locally owned interval.
   std::array<mfem::real_t, coefficients_per_interval>
   LocalCoefficients(int local_element) const;

   /// Evaluate on a specified locally owned mesh interval.
   mfem::real_t EvaluateLocalElement(int local_element,
                                     mfem::real_t x) const;

   /**
    * Evaluate if x lies in at least one locally owned interval. At a shared
    * knot either adjacent rank can return the value. Returns false when x is
    * outside this rank's locally owned intervals.
    */
   bool TryEvaluateLocal(mfem::real_t x, mfem::real_t &value) const;

   /**
    * Collectively evaluate the spline at x and return the value on every
    * rank. All ranks in the spline communicator must call this method with
    * the same x. If x is a shared knot, the lowest candidate rank performs
    * the evaluation and becomes the MPI broadcast root.
    */
   mfem::real_t Evaluate(mfem::real_t x) const;

   /// Coordinates corresponding to the local displacement-vector ordering.
   const std::vector<mfem::real_t> &LocalNodeCoordinates() const
   {
      return node_data_.Coordinates();
   }

   /**
    * Collectively evaluate S(x_i + displacement_i) for every node x_i in
    * LocalNodeCoordinates(). The returned Vector has the same local ordering
    * and includes locally present copies of shared nodes. A displaced point
    * may lie on an interval owned by another rank. Every displaced point must
    * remain in the global spline interval.
    */
   mfem::Vector EvaluateDisplacedNodes(
      const mfem::Vector &displacements) const;

private:
   struct Neighbor
   {
      HYPRE_BigInt global_interval = -1;
      mfem::real_t x_left = 0.0;
      mfem::real_t x_right = 0.0;

      bool IsSet() const { return global_interval >= 0; }
   };

   struct Interval
   {
      int local_element = -1;
      int left_vertex = -1;
      int right_vertex = -1;
      HYPRE_BigInt global_interval = -1;

      mfem::real_t x_left = 0.0;
      mfem::real_t x_right = 0.0;
      HYPRE_BigInt left_global_true_dof = -1;
      HYPRE_BigInt right_global_true_dof = -1;

      Neighbor left_neighbor;
      Neighbor right_neighbor;
      bool physical_left_boundary = false;
      bool physical_right_boundary = false;
   };

   struct RowEntry
   {
      HYPRE_BigInt column = -1;
      mfem::real_t value = 0.0;
   };

   struct BoundaryNode
   {
      mfem::real_t x = 0.0;
      HYPRE_BigInt global_true_dof = -1;
   };

   struct BoundaryStencil
   {
      std::array<BoundaryNode, 3> nodes;
      std::array<mfem::real_t, 3> first_derivative_weights;
      std::array<mfem::real_t, 3> second_derivative_weights;
   };

   const mfem::ParGridFunction *input_ = nullptr; // not owned
   mfem::ParFiniteElementSpace *input_space_ = nullptr; // not owned
   mfem::ParMesh *mesh_ = nullptr; // not owned
   MPI_Comm comm_ = MPI_COMM_NULL;
   int rank_ = -1;

   mfem::real_t coordinate_tolerance_ = 0.0;
   ParLinear1DNodeData node_data_;

   std::vector<Interval> intervals_;
   mfem::Array<HYPRE_BigInt> coefficient_offsets_;
   HYPRE_BigInt global_number_of_intervals_ = 0;
   HYPRE_BigInt global_system_size_ = 0;
   mfem::real_t global_x_left_ = 0.0;
   mfem::real_t global_x_right_ = 0.0;
   BoundaryStencil left_boundary_stencil_;
   BoundaryStencil right_boundary_stencil_;

   // Declaration order ensures the vectors are destroyed before the matrices.
   std::unique_ptr<mfem::HypreParMatrix> A_;
   std::unique_ptr<mfem::HypreParMatrix> rhs_data_jacobian_;
   std::unique_ptr<mfem::HypreParVector> rhs_;
   std::unique_ptr<mfem::HypreParVector> coefficients_;
   bool solved_ = false;

   void BuildIntervals();
   void CompleteOffRankNeighbors();
   void BuildBoundaryStencils();

   static BoundaryStencil MakeBoundaryStencil(
      const std::array<BoundaryNode, 3> &nodes);

   static void BuildCSR(std::vector<std::vector<RowEntry>> &rows,
                        std::vector<int> &I,
                        std::vector<HYPRE_BigInt> &J,
                        std::vector<mfem::real_t> &data);

   bool Near(mfem::real_t a, mfem::real_t b) const;
   static mfem::real_t IntegerPower(mfem::real_t x, int exponent);
   static mfem::real_t FallingFactorial(int j, int derivative);
   static HYPRE_BigInt CoefficientColumn(HYPRE_BigInt global_interval,
                                         int j);

   void AddEntry(std::vector<RowEntry> &row,
                 HYPRE_BigInt global_interval,
                 int coefficient,
                 mfem::real_t value) const;

   void AddDerivative(std::vector<RowEntry> &row,
                      HYPRE_BigInt global_interval,
                      mfem::real_t x_right,
                      mfem::real_t interval_length,
                      mfem::real_t x,
                      int derivative,
                      mfem::real_t row_scale) const;
};

} // namespace spline

#endif // QUARTIC_SPLINE_1D_HPP
