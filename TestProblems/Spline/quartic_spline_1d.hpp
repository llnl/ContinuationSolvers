#ifndef QUARTIC_SPLINE_1D_HPP
#define QUARTIC_SPLINE_1D_HPP

#include "mfem.hpp"
#include "linear_1d_node_data.hpp"

#include <array>
#include <functional>
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

enum class QuarticInterpolationMode
{
   FiniteDifferenceBoundaryConditions,
   MinimumCurvature,
   MinimumCurvatureAndVariation
};

struct QuarticInterpolationOptions
{
   QuarticInterpolationMode mode =
      QuarticInterpolationMode::FiniteDifferenceBoundaryConditions;
   mfem::real_t coordinate_tolerance = 0.0;

   /**
    * Dimensionless length eta used by MinimumCurvatureAndVariation:
    *
    *   integral [(S'')^2 + (eta L)^2 (S''')^2] dx,
    *
    * where L is the global interpolation-interval length. Thus eta=0.01
    * penalizes curvature variation over a length scale equal to one percent
    * of the global interval. This member is ignored by the other modes.
    */
   mfem::real_t curvature_variation_length_fraction = 1.0e-2;
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
 * FiniteDifferenceBoundaryConditions reproduces the original square system
 * with three one-sided finite-difference endpoint conditions.
 * MinimumCurvature instead minimizes integral |S''|^2 subject to exact nodal
 * interpolation and C1/C2/C3 continuity. It introduces Lagrange multipliers
 * and solves the corresponding symmetric-indefinite KKT system; no endpoint
 * derivative values are estimated in this mode.
 * MinimumCurvatureAndVariation uses the same constraints and additionally
 * penalizes |S'''|^2, which suppresses thin endpoint curvature layers.
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

   ParQuarticSpline1D(
      const mfem::ParGridFunction &input,
      QuarticInterpolationMode interpolation_mode,
      mfem::real_t coordinate_tolerance = 0.0);

   ParQuarticSpline1D(
      const mfem::ParGridFunction &input,
      const QuarticInterpolationOptions &interpolation_options);

   ParQuarticSpline1D(const ParQuarticSpline1D &) = delete;
   ParQuarticSpline1D &operator=(const ParQuarticSpline1D &) = delete;
   ParQuarticSpline1D(ParQuarticSpline1D &&) = delete;
   ParQuarticSpline1D &operator=(ParQuarticSpline1D &&) = delete;
   ~ParQuarticSpline1D() = default;

   /// Assemble A and b. The constructor calls this once automatically.
   void Assemble();

   /// Solve the assembled finite-difference or KKT system.
   QuarticSolveResult Solve(
      const QuarticSolveOptions &options = QuarticSolveOptions());

   /// The coefficient system, or the full KKT matrix in either variational mode.
   const mfem::HypreParMatrix &SystemMatrix() const;
   /**
    * Matrix B in b = B y, where y is the input GridFunction's distributed
    * true-DOF vector. In finite-difference mode B also includes the three
    * endpoint derivative formulas. In either variational mode it is the
    * block map [0; F] into the KKT right-hand side.
    *
    * For an adjoint lambda satisfying A^T lambda = q, where q contains the
    * coefficient evaluation gradient and zeros in any multiplier block, the
    * derivative with respect to the nodal data is B^T lambda.
    */
   const mfem::HypreParMatrix &RightHandSideDataJacobian() const;
   /// Full system right-hand side, including the KKT blocks when present.
   const mfem::HypreParVector &RightHandSide() const;
   /// Distributed vector of normalized coefficients a_{e,j}.
   const mfem::HypreParVector &CoefficientVector() const;

   QuarticInterpolationMode InterpolationMode() const
   {
      return interpolation_mode_;
   }

   mfem::real_t CurvatureVariationLengthFraction() const
   {
      return curvature_variation_length_fraction_;
   }

   /**
    * Collectively replace y while preserving the mesh coordinates and all
    * assembled matrices. local_true_y uses the input finite element space's
    * locally owned true-DOF ordering. A retained MUMPS factorization can be
    * reused by the next Solve() call.
    */
   void SetInterpolationValues(const mfem::Vector &local_true_y);

   /// Reload y from the ParGridFunction supplied to the constructor.
   void RefreshInterpolationValues();

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
                                     mfem::real_t x, int d=0) const;

   /**
    * Evaluate if x lies in at least one locally owned interval. At a shared
    * knot either adjacent rank can return the value. Returns false when x is
    * outside this rank's locally owned intervals.
    */
   bool TryEvaluateLocal(mfem::real_t x, mfem::real_t &value, int d = 0) const;

   /**
    * Collectively evaluate the spline at x and return the value on every
    * rank. All ranks in the spline communicator must call this method with
    * the same x. If x is a shared knot, the lowest candidate rank performs
    * the evaluation and becomes the MPI broadcast root.
    */
   mfem::real_t Evaluate(mfem::real_t x, int d = 0) const;

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
   /**
    * Collectively evaluate S(eval_pts_i) at all distributed eval_pts_i.
    * The returned Vector has the same local ordering
    * and includes locally present copies of shared nodes. Each evaluation point
    * may lie on an interval owned by some rank, i.e., every point must
    * remain in the global spline interpolation interval.
    */
   mfem::Vector Evaluate(
      const mfem::Vector &evaluation_pts, int d = 0) const;

   /**
    * Collectively return the gradient of S(x) with respect to the input
    * interpolation values y. The result is the locally owned portion of the
    * distributed true-DOF vector
    *
    *   grad_y S(x) = B^T A^{-T} q(x),
    *
    * where q is grad_a S(x), extended by zeros for KKT multipliers.
    *
    * All ranks must call this method with the same x. Solve() must previously
    * have been called with direct_solver=true so the MUMPS factorization is
    * available for the transpose solve.
    */
   mfem::Vector EvaluateDataGradient(mfem::real_t x) const;

   /**
    * Collectively assemble the Jacobian of the spline values at a
    * distributed set of evaluation points with respect to the input data.
    * Rank r supplies its locally owned rows in evaluation_points; the global
    * row ordering is the concatenation of those vectors in MPI-rank order.
    * The columns use the input finite element space's true-DOF partitioning:
    *
    *   J(i,j) = d S(evaluation_points_i) / d y_j.
    *
    * Thus each row is the distributed gradient returned by
    * EvaluateDataGradient for the corresponding point. Every point must lie
    * in the global spline interval. Solve() must previously have been called
    * with direct_solver=true.
    */
   mfem::HypreParMatrix EvaluateDataJacobian(
      const mfem::Vector &evaluation_points) const;

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
   QuarticInterpolationMode interpolation_mode_ =
      QuarticInterpolationMode::FiniteDifferenceBoundaryConditions;
   mfem::real_t curvature_variation_length_fraction_ = 1.0e-2;
   ParLinear1DNodeData node_data_;

   std::vector<Interval> intervals_;
   mfem::Array<HYPRE_BigInt> coefficient_offsets_;
   mfem::Array<HYPRE_BigInt> constraint_offsets_;
   mfem::Array<HYPRE_BigInt> system_offsets_;
   HYPRE_BigInt global_number_of_intervals_ = 0;
   HYPRE_BigInt global_coefficient_size_ = 0;
   mfem::real_t global_x_left_ = 0.0;
   mfem::real_t global_x_right_ = 0.0;
   BoundaryStencil left_boundary_stencil_;
   BoundaryStencil right_boundary_stencil_;

   // Declaration order ensures the vectors are destroyed before the matrices.
   std::unique_ptr<mfem::HypreParMatrix> A_;
   std::unique_ptr<mfem::HypreParMatrix> rhs_data_jacobian_;
   std::unique_ptr<mfem::HypreParVector> rhs_;
   std::unique_ptr<mfem::HypreParVector> system_solution_;
   std::unique_ptr<mfem::HypreParVector> coefficients_;
   // Declared after A_ so it is destroyed before the matrix it factorizes.
   std::unique_ptr<mfem::Solver> direct_solver_;
   bool solved_ = false;

   void BuildIntervals();
   void CompleteOffRankNeighbors();
   void BuildBoundaryStencils();
   void AssembleFiniteDifferenceSystem();
   void AssembleMinimumCurvatureSystem();

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
