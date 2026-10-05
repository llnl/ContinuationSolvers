#include "quintic_smoothing_spline_1d.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace spline
{
namespace
{

constexpr int neighbor_count_tag = 23241;
constexpr int neighbor_data_tag = 23242;

struct SharedIntervalRecord
{
   mfem::real_t shared_x;
   mfem::real_t x_left;
   mfem::real_t x_right;
   HYPRE_BigInt global_interval;
};

static_assert(std::is_trivially_copyable<SharedIntervalRecord>::value,
              "SharedIntervalRecord must be suitable for MPI_BYTE transfer.");

struct VertexIncidence
{
   int interval = -1;
   bool is_left = false;
};

} // namespace

ParQuinticSmoothingSpline1D::ParQuinticSmoothingSpline1D(
   const mfem::ParGridFunction &input,
   const QuinticSmoothingOptions &smoothing_options)
   : input_(&input),
     input_space_(input.ParFESpace()),
     mesh_(input_space_ ? input_space_->GetParMesh() : nullptr),
     comm_(mesh_ ? mesh_->GetComm() : MPI_COMM_NULL),
     rank_(mesh_ ? mesh_->GetMyRank() : -1),
     data_fidelity_(smoothing_options.data_fidelity),
     coordinate_tolerance_(smoothing_options.coordinate_tolerance),
     node_data_(input)
{
   MFEM_VERIFY(mesh_ != nullptr, "A valid ParMesh is required.");
   MFEM_VERIFY(std::isfinite(data_fidelity_) &&
                  data_fidelity_ > 0.0 && data_fidelity_ < 1.0,
               "data_fidelity must satisfy 0 < p < 1.");
   MFEM_VERIFY(std::isfinite(coordinate_tolerance_) &&
                  coordinate_tolerance_ >= 0.0,
               "coordinate_tolerance must be finite and nonnegative.");

   BuildIntervals();
   MFEM_VERIFY(global_number_of_intervals_ >= 2,
               "A quintic smoothing spline requires at least three nodes.");
   CompleteOffRankNeighbors();
   Assemble();
}

bool ParQuinticSmoothingSpline1D::Near(mfem::real_t a, mfem::real_t b) const
{
   const mfem::real_t scale =
      std::max<mfem::real_t>({1.0, std::abs(a), std::abs(b)});
   const mfem::real_t automatic_tolerance =
      64.0 * std::numeric_limits<mfem::real_t>::epsilon() * scale;
   return std::abs(a - b) <=
          std::max(coordinate_tolerance_, automatic_tolerance);
}

mfem::real_t ParQuinticSmoothingSpline1D::IntegerPower(mfem::real_t x, int exponent)
{
   MFEM_ASSERT(exponent >= 0, "A nonnegative exponent is required.");
   mfem::real_t result = 1.0;
   for (int i = 0; i < exponent; ++i) { result *= x; }
   return result;
}

mfem::real_t ParQuinticSmoothingSpline1D::FallingFactorial(int j, int derivative)
{
   MFEM_ASSERT(j >= derivative && derivative >= 0,
               "Invalid monomial derivative.");
   mfem::real_t result = 1.0;
   for (int k = 0; k < derivative; ++k) { result *= (j - k); }
   return result;
}

HYPRE_BigInt ParQuinticSmoothingSpline1D::CoefficientColumn(
   HYPRE_BigInt global_interval, int j)
{
   MFEM_ASSERT(global_interval >= 0, "Invalid global interval number.");
   MFEM_ASSERT(j >= 0 && j < coefficients_per_interval,
               "Invalid quintic coefficient number.");
   return coefficients_per_interval * global_interval + j;
}

void ParQuinticSmoothingSpline1D::BuildIntervals()
{
   const int local_ne = mesh_->GetNE();
   MFEM_VERIFY(local_ne > 0,
               "ParQuinticSmoothingSpline1D currently requires every MPI rank to own "
               "at least one interval.");

   global_number_of_intervals_ =
      static_cast<HYPRE_BigInt>(mesh_->GetGlobalNE());
   MFEM_VERIFY(global_number_of_intervals_ > 0,
               "At least one mesh interval is required.");
   MFEM_VERIFY(global_number_of_intervals_ <=
                  std::numeric_limits<HYPRE_BigInt>::max() /
                     coefficients_per_interval,
               "The quintic coefficient numbering overflows HYPRE_BigInt.");
   global_coefficient_size_ =
      coefficients_per_interval * global_number_of_intervals_;

   const HYPRE_BigInt local_system_size =
      coefficients_per_interval * static_cast<HYPRE_BigInt>(local_ne);
   HYPRE_BigInt local_sizes[1] = {local_system_size};
   mfem::Array<HYPRE_BigInt> *offset_arrays[1] = {&coefficient_offsets_};
   mesh_->GenerateOffsets(1, local_sizes, offset_arrays);

   MFEM_VERIFY(coefficient_offsets_.Last() == global_coefficient_size_,
               "Inconsistent global interval and coefficient counts.");

   mfem::Array<HYPRE_BigInt> global_elements;
   mesh_->GetGlobalElementIndices(global_elements);
   MFEM_VERIFY(global_elements.Size() == local_ne,
               "ParMesh returned an invalid global element map.");

   std::vector<HYPRE_BigInt> vertex_global_true_dofs(
      static_cast<std::size_t>(mesh_->GetNV()), -1);
   std::vector<bool> vertex_is_shared(
      static_cast<std::size_t>(mesh_->GetNV()), false);
   for (const NodeSample &sample : node_data_.Nodes())
   {
      vertex_global_true_dofs[static_cast<std::size_t>(sample.local_vertex)] =
         sample.global_true_dof;
      vertex_is_shared[static_cast<std::size_t>(sample.local_vertex)] =
         sample.is_shared;
   }

   intervals_.resize(static_cast<std::size_t>(local_ne));
   std::vector<std::vector<VertexIncidence>> incidence(
      static_cast<std::size_t>(mesh_->GetNV()));

   mfem::Array<int> vertices;
   for (int e = 0; e < local_ne; ++e)
   {
      mesh_->GetElementVertices(e, vertices);
      MFEM_VERIFY(vertices.Size() == 2,
                  "Every element of a one-dimensional mesh must have two "
                  "vertices.");

      int left_vertex = vertices[0];
      int right_vertex = vertices[1];
      mfem::real_t x_left = mesh_->GetVertex(left_vertex)[0];
      mfem::real_t x_right = mesh_->GetVertex(right_vertex)[0];
      if (x_right < x_left)
      {
         std::swap(left_vertex, right_vertex);
         std::swap(x_left, x_right);
      }
      MFEM_VERIFY(!Near(x_left, x_right),
                  "Degenerate mesh interval detected.");

      Interval &interval = intervals_[static_cast<std::size_t>(e)];
      interval.local_element = e;
      interval.left_vertex = left_vertex;
      interval.right_vertex = right_vertex;
      interval.global_interval = global_elements[e];
      interval.x_left = x_left;
      interval.x_right = x_right;
      interval.left_global_true_dof =
         vertex_global_true_dofs[static_cast<std::size_t>(left_vertex)];
      interval.right_global_true_dof =
         vertex_global_true_dofs[static_cast<std::size_t>(right_vertex)];
      MFEM_VERIFY(interval.left_global_true_dof >= 0 &&
                     interval.right_global_true_dof >= 0,
                  "Could not number an interval endpoint true DOF.");

      incidence[static_cast<std::size_t>(left_vertex)].push_back({e, true});
      incidence[static_cast<std::size_t>(right_vertex)].push_back({e, false});
   }

   mfem::real_t local_x_left = intervals_.front().x_left;
   mfem::real_t local_x_right = intervals_.front().x_right;
   for (const Interval &interval : intervals_)
   {
      local_x_left = std::min(local_x_left, interval.x_left);
      local_x_right = std::max(local_x_right, interval.x_right);
   }
   MPI_Allreduce(&local_x_left, &global_x_left_, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MIN, comm_);
   MPI_Allreduce(&local_x_right, &global_x_right_, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MAX, comm_);

   int local_physical_left = 0;
   int local_physical_right = 0;
   for (int e = 0; e < local_ne; ++e)
   {
      Interval &interval = intervals_[static_cast<std::size_t>(e)];
      for (int side_index = 0; side_index < 2; ++side_index)
      {
         const bool is_left = side_index == 0;
         const int vertex = is_left ? interval.left_vertex
                                    : interval.right_vertex;
         const auto &at_vertex = incidence[static_cast<std::size_t>(vertex)];
         MFEM_VERIFY(!at_vertex.empty() && at_vertex.size() <= 2,
                     "The mesh is not a one-dimensional manifold.");

         Neighbor &neighbor = is_left ? interval.left_neighbor
                                      : interval.right_neighbor;
         if (at_vertex.size() == 2)
         {
            const VertexIncidence &other =
               at_vertex[0].interval == e ? at_vertex[1] : at_vertex[0];
            const Interval &other_interval =
               intervals_[static_cast<std::size_t>(other.interval)];
            neighbor.global_interval = other_interval.global_interval;
            neighbor.x_left = other_interval.x_left;
            neighbor.x_right = other_interval.x_right;
         }
         else if (!vertex_is_shared[static_cast<std::size_t>(vertex)])
         {
            if (is_left)
            {
               interval.physical_left_boundary = true;
               ++local_physical_left;
            }
            else
            {
               interval.physical_right_boundary = true;
               ++local_physical_right;
            }
         }
      }
   }

   int global_physical_left = 0;
   int global_physical_right = 0;
   MPI_Allreduce(&local_physical_left, &global_physical_left, 1, MPI_INT,
                 MPI_SUM, comm_);
   MPI_Allreduce(&local_physical_right, &global_physical_right, 1, MPI_INT,
                 MPI_SUM, comm_);
   MFEM_VERIFY(global_physical_left == 1 && global_physical_right == 1,
               "The mesh must describe one connected, non-periodic interval.");
}

void ParQuinticSmoothingSpline1D::CompleteOffRankNeighbors()
{
   std::vector<SharedIntervalRecord> send_records;
   for (const Interval &interval : intervals_)
   {
      if (!interval.left_neighbor.IsSet() &&
          !interval.physical_left_boundary)
      {
         send_records.push_back({interval.x_left, interval.x_left,
                                 interval.x_right,
                                 interval.global_interval});
      }
      if (!interval.right_neighbor.IsSet() &&
          !interval.physical_right_boundary)
      {
         send_records.push_back({interval.x_right, interval.x_left,
                                 interval.x_right,
                                 interval.global_interval});
      }
   }

   if (send_records.empty()) { return; }

   // This obtains only neighboring-rank identities and mesh topology. It does
   // not exchange ParGridFunction values or construct a ghost field.
   mesh_->ExchangeFaceNbrData();
   std::vector<int> neighbor_ranks;
   neighbor_ranks.reserve(static_cast<std::size_t>(mesh_->GetNFaceNeighbors()));
   for (int fn = 0; fn < mesh_->GetNFaceNeighbors(); ++fn)
   {
      neighbor_ranks.push_back(mesh_->GetFaceNbrRank(fn));
   }
   std::sort(neighbor_ranks.begin(), neighbor_ranks.end());
   neighbor_ranks.erase(
      std::unique(neighbor_ranks.begin(), neighbor_ranks.end()),
      neighbor_ranks.end());
   MFEM_VERIFY(!neighbor_ranks.empty(),
               "A shared endpoint has no face-neighbor rank.");

   MFEM_VERIFY(send_records.size() <=
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
               "Too many shared interval records for MPI.");
   const int send_count = static_cast<int>(send_records.size());
   const int number_of_neighbors = static_cast<int>(neighbor_ranks.size());

   std::vector<int> receive_counts(
      static_cast<std::size_t>(number_of_neighbors), 0);
   std::vector<MPI_Request> requests(
      static_cast<std::size_t>(2 * number_of_neighbors), MPI_REQUEST_NULL);
   for (int i = 0; i < number_of_neighbors; ++i)
   {
      MPI_Irecv(&receive_counts[static_cast<std::size_t>(i)], 1, MPI_INT,
                neighbor_ranks[static_cast<std::size_t>(i)],
                neighbor_count_tag, comm_,
                &requests[static_cast<std::size_t>(2 * i)]);
      MPI_Isend(&send_count, 1, MPI_INT,
                neighbor_ranks[static_cast<std::size_t>(i)],
                neighbor_count_tag, comm_,
                &requests[static_cast<std::size_t>(2 * i + 1)]);
   }
   MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
               MPI_STATUSES_IGNORE);

   std::vector<std::vector<SharedIntervalRecord>> receive_records(
      static_cast<std::size_t>(number_of_neighbors));
   std::fill(requests.begin(), requests.end(), MPI_REQUEST_NULL);
   for (int i = 0; i < number_of_neighbors; ++i)
   {
      const int receive_count = receive_counts[static_cast<std::size_t>(i)];
      MFEM_VERIFY(receive_count >= 0,
                  "Received an invalid shared interval record count.");
      receive_records[static_cast<std::size_t>(i)].resize(
         static_cast<std::size_t>(receive_count));

      const std::size_t receive_bytes =
         static_cast<std::size_t>(receive_count) *
         sizeof(SharedIntervalRecord);
      const std::size_t send_bytes =
         send_records.size() * sizeof(SharedIntervalRecord);
      MFEM_VERIFY(receive_bytes <=
                     static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
                     send_bytes <=
                     static_cast<std::size_t>(std::numeric_limits<int>::max()),
                  "Shared interval metadata exceeds an MPI message count.");

      MPI_Irecv(receive_records[static_cast<std::size_t>(i)].data(),
                static_cast<int>(receive_bytes), MPI_BYTE,
                neighbor_ranks[static_cast<std::size_t>(i)],
                neighbor_data_tag, comm_,
                &requests[static_cast<std::size_t>(2 * i)]);
      MPI_Isend(send_records.data(), static_cast<int>(send_bytes), MPI_BYTE,
                neighbor_ranks[static_cast<std::size_t>(i)],
                neighbor_data_tag, comm_,
                &requests[static_cast<std::size_t>(2 * i + 1)]);
   }
   MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
               MPI_STATUSES_IGNORE);

   for (Interval &interval : intervals_)
   {
      for (int side_index = 0; side_index < 2; ++side_index)
      {
         const bool is_left = side_index == 0;
         Neighbor &neighbor = is_left ? interval.left_neighbor
                                      : interval.right_neighbor;
         const bool is_physical = is_left
                                     ? interval.physical_left_boundary
                                     : interval.physical_right_boundary;
         if (neighbor.IsSet() || is_physical) { continue; }

         const mfem::real_t shared_x =
            is_left ? interval.x_left : interval.x_right;
         int matches = 0;
         for (const auto &from_rank : receive_records)
         {
            for (const SharedIntervalRecord &record : from_rank)
            {
               const bool correct_endpoint =
                  is_left ? Near(record.x_right, shared_x)
                          : Near(record.x_left, shared_x);
               if (Near(record.shared_x, shared_x) && correct_endpoint)
               {
                  neighbor.global_interval = record.global_interval;
                  neighbor.x_left = record.x_left;
                  neighbor.x_right = record.x_right;
                  ++matches;
               }
            }
         }
         MFEM_VERIFY(matches == 1,
                     "Could not identify exactly one off-rank interval at a "
                     "shared mesh node.");
      }
   }
}


void ParQuinticSmoothingSpline1D::AddEntry(std::vector<RowEntry> &row,
                                  HYPRE_BigInt global_interval,
                                  int coefficient,
                                  mfem::real_t value) const
{
   if (value == 0.0) { return; }
   row.push_back(
      {CoefficientColumn(global_interval, coefficient), value});
}

void ParQuinticSmoothingSpline1D::AddDerivative(
   std::vector<RowEntry> &row,
   HYPRE_BigInt global_interval,
   mfem::real_t x_right,
   mfem::real_t interval_length,
   mfem::real_t x,
   int derivative,
   mfem::real_t row_scale) const
{
   MFEM_ASSERT(derivative >= 0 && derivative <= order,
               "Invalid derivative order.");
   MFEM_ASSERT(interval_length > 0.0,
               "A positive interval length is required.");

   // If t = (x-x_right)/h, then
   //
   //   d^d(t^j)/dx^d = (j!/(j-d)!) t^(j-d) / h^d.
   //
   // The coefficients assembled here are therefore the normalized
   // coefficients a_j = h^j c_j rather than the physical-power
   // coefficients c_j.
   const mfem::real_t t = (x - x_right) / interval_length;
   const mfem::real_t derivative_scale =
      row_scale / IntegerPower(interval_length, derivative);
   for (int j = derivative; j <= order; ++j)
   {
      const mfem::real_t value =
         derivative_scale * FallingFactorial(j, derivative) *
         IntegerPower(t, j - derivative);
      AddEntry(row, global_interval, j, value);
   }
}

void ParQuinticSmoothingSpline1D::BuildCSR(
   std::vector<std::vector<RowEntry>> &rows,
   std::vector<int> &I,
   std::vector<HYPRE_BigInt> &J,
   std::vector<mfem::real_t> &data)
{
   MFEM_VERIFY(rows.size() <
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
               "The local CSR row count exceeds INT_MAX.");
   I.assign(rows.size() + 1, 0);
   J.clear();
   data.clear();

   for (std::size_t row_number = 0; row_number < rows.size(); ++row_number)
   {
      std::vector<RowEntry> &row = rows[row_number];
      std::sort(row.begin(), row.end(),
                [](const RowEntry &a, const RowEntry &b)
                {
                   return a.column < b.column;
                });

      for (std::size_t k = 0; k < row.size();)
      {
         const HYPRE_BigInt column = row[k].column;
         mfem::real_t value = 0.0;
         do
         {
            value += row[k].value;
            ++k;
         }
         while (k < row.size() && row[k].column == column);

         if (value != 0.0)
         {
            J.push_back(column);
            data.push_back(value);
         }
      }
      MFEM_VERIFY(J.size() <=
                     static_cast<std::size_t>(std::numeric_limits<int>::max()),
                  "The local CSR nonzero count exceeds INT_MAX.");
      I[row_number + 1] = static_cast<int>(J.size());
   }
}


void ParQuinticSmoothingSpline1D::Assemble()
{
   direct_solver_.reset();
   coefficients_.reset();
   system_solution_.reset();
   rhs_.reset();
   rhs_data_jacobian_.reset();
   A_.reset();

   AssembleSmoothingSystem();
   MFEM_VERIFY(A_ != nullptr && rhs_data_jacobian_ != nullptr,
               "Smoothing-spline matrix assembly failed.");

   const int local_coefficient_size =
      coefficients_per_interval * static_cast<int>(intervals_.size());
   rhs_ = std::make_unique<mfem::HypreParVector>(*A_);
   system_solution_ = std::make_unique<mfem::HypreParVector>(*A_);
   coefficients_ = std::make_unique<mfem::HypreParVector>(
      comm_, global_coefficient_size_, coefficient_offsets_.GetData());
   MFEM_VERIFY(coefficients_->Size() == local_coefficient_size,
               "Unexpected local coefficient-vector size.");
   MFEM_VERIFY(rhs_->Size() == system_solution_->Size() &&
                  rhs_->Size() == A_->GetNumRows(),
               "Unexpected local smoothing-system vector size.");

   mfem::Vector true_values(input_space_->GetTrueVSize());
   input_->GetTrueDofs(true_values);
   SetDataValues(true_values);
}

void ParQuinticSmoothingSpline1D::AssembleSmoothingSystem()
{
   const int local_ne = static_cast<int>(intervals_.size());
   const int local_coefficient_size =
      coefficients_per_interval * local_ne;

   int local_constraint_count = 0;
   for (const Interval &interval : intervals_)
   {
      if (!interval.physical_right_boundary)
      {
         local_constraint_count += order;
      }
   }

   const HYPRE_BigInt global_constraint_count =
      order * (global_number_of_intervals_ - 1);
   MFEM_VERIFY(global_coefficient_size_ <=
                  std::numeric_limits<HYPRE_BigInt>::max() -
                     global_constraint_count,
               "The quintic smoothing KKT size overflows HYPRE_BigInt.");
   const HYPRE_BigInt global_kkt_size =
      global_coefficient_size_ + global_constraint_count;
   const int local_kkt_size =
      local_coefficient_size + local_constraint_count;

   HYPRE_BigInt local_sizes[2] =
   {
      static_cast<HYPRE_BigInt>(local_constraint_count),
      static_cast<HYPRE_BigInt>(local_kkt_size)
   };
   mfem::Array<HYPRE_BigInt> *offset_arrays[2] =
   {
      &constraint_offsets_, &system_offsets_
   };
   mesh_->GenerateOffsets(2, local_sizes, offset_arrays);
   MFEM_VERIFY(constraint_offsets_.Last() == global_constraint_count,
               "Inconsistent smoothing constraint count.");
   MFEM_VERIFY(system_offsets_.Last() == global_kkt_size,
               "Inconsistent smoothing KKT size.");

   std::vector<std::vector<RowEntry>> objective_rows(
      static_cast<std::size_t>(local_coefficient_size));
   std::vector<std::vector<RowEntry>> constraint_rows(
      static_cast<std::size_t>(local_constraint_count));
   std::vector<std::vector<RowEntry>> coefficient_data_rows(
      static_cast<std::size_t>(local_coefficient_size));

   const auto add_data_entry =
      [&coefficient_data_rows](int local_row,
                               HYPRE_BigInt global_true_dof,
                               mfem::real_t value)
      {
         MFEM_ASSERT(local_row >= 0 &&
                        local_row <
                           static_cast<int>(coefficient_data_rows.size()),
                     "Invalid smoothing-data row.");
         MFEM_ASSERT(global_true_dof >= 0,
                     "Invalid global true-DOF column.");
         if (value != 0.0)
         {
            coefficient_data_rows[static_cast<std::size_t>(local_row)]
               .push_back({global_true_dof, value});
         }
      };

   const mfem::real_t average_h =
      (global_x_right_ - global_x_left_) /
      static_cast<mfem::real_t>(global_number_of_intervals_);
   const mfem::real_t penalty_weight =
      (1.0 - data_fidelity_) *
      IntegerPower(average_h, 2 * 3 - 1);

   int constraint_row = 0;
   for (const Interval &interval : intervals_)
   {
      const int local_element = interval.local_element;
      const int local_base =
         coefficients_per_interval * local_element;
      const mfem::real_t h = interval.x_right - interval.x_left;
      MFEM_ASSERT(h > 0.0, "A positive interval length is required.");

      // Each interior data node occurs as the endpoint of two intervals.
      // Giving each occurrence half weight counts every global datum once
      // after C0 continuity is imposed.
      for (int endpoint = 0; endpoint < 2; ++endpoint)
      {
         const bool is_left = endpoint == 0;
         const bool physical_boundary =
            is_left ? interval.physical_left_boundary
                    : interval.physical_right_boundary;
         const mfem::real_t occurrence_weight =
            data_fidelity_ * (physical_boundary ? 1.0 : 0.5);
         const HYPRE_BigInt data_dof =
            is_left ? interval.left_global_true_dof
                    : interval.right_global_true_dof;
         const mfem::real_t t = is_left ? -1.0 : 0.0;

         mfem::real_t basis[coefficients_per_interval];
         for (int j = 0; j <= order; ++j)
         {
            basis[j] = IntegerPower(t, j);
         }

         for (int j = 0; j <= order; ++j)
         {
            const int local_row = local_base + j;
            add_data_entry(local_row, data_dof,
                           occurrence_weight * basis[j]);
            for (int k = 0; k <= order; ++k)
            {
               AddEntry(
                  objective_rows[static_cast<std::size_t>(local_row)],
                  interval.global_interval, k,
                  occurrence_weight * basis[j] * basis[k]);
            }
         }
      }

      // h_bar^5 integral (S''')^2 dx. For normalized monomials,
      // integral on this element contributes h_bar^5 / h^5.
      const int derivative = 3;
      const mfem::real_t element_scale =
         penalty_weight / IntegerPower(h, 2 * derivative - 1);
      for (int j = derivative; j <= order; ++j)
      {
         const int local_row = local_base + j;
         for (int k = derivative; k <= order; ++k)
         {
            const int exponent = j + k - 2 * derivative;
            const mfem::real_t monomial_integral =
               (exponent % 2 == 0 ? 1.0 : -1.0) /
               static_cast<mfem::real_t>(exponent + 1);
            AddEntry(
               objective_rows[static_cast<std::size_t>(local_row)],
               interval.global_interval, k,
               element_scale * FallingFactorial(j, derivative) *
                  FallingFactorial(k, derivative) *
                  monomial_integral);
         }
      }

      // Assign all C0,...,C4 equations at an interior knot to the interval
      // on its left. Scaling an order-d equation by h^d keeps its row
      // dimensionless and does not change the feasible spline space.
      if (!interval.physical_right_boundary)
      {
         MFEM_VERIFY(interval.right_neighbor.IsSet(),
                     "Missing the interval to the right.");
         for (int derivative_order = 0;
              derivative_order <= order - 1;
              ++derivative_order)
         {
            const mfem::real_t row_scale =
               IntegerPower(h, derivative_order);
            AddDerivative(
               constraint_rows[static_cast<std::size_t>(constraint_row)],
               interval.right_neighbor.global_interval,
               interval.right_neighbor.x_right,
               interval.right_neighbor.x_right -
                  interval.right_neighbor.x_left,
               interval.x_right, derivative_order, row_scale);
            AddDerivative(
               constraint_rows[static_cast<std::size_t>(constraint_row)],
               interval.global_interval, interval.x_right, h,
               interval.x_right, derivative_order, -row_scale);
            ++constraint_row;
         }
      }
   }
   MFEM_VERIFY(constraint_row == local_constraint_count,
               "Incorrect local smoothing constraint count.");

   std::vector<int> objective_I;
   std::vector<HYPRE_BigInt> objective_J;
   std::vector<mfem::real_t> objective_data;
   BuildCSR(objective_rows, objective_I, objective_J, objective_data);

   std::vector<int> constraint_I;
   std::vector<HYPRE_BigInt> constraint_J;
   std::vector<mfem::real_t> constraint_data;
   BuildCSR(constraint_rows, constraint_I, constraint_J, constraint_data);

   std::vector<std::vector<RowEntry>> system_data_rows(
      static_cast<std::size_t>(local_kkt_size));
   for (int row = 0; row < local_coefficient_size; ++row)
   {
      system_data_rows[static_cast<std::size_t>(row)] =
         std::move(coefficient_data_rows[static_cast<std::size_t>(row)]);
   }
   std::vector<int> system_rhs_I;
   std::vector<HYPRE_BigInt> system_rhs_J;
   std::vector<mfem::real_t> system_rhs_data;
   BuildCSR(system_data_rows, system_rhs_I,
            system_rhs_J, system_rhs_data);

   auto objective = std::make_unique<mfem::HypreParMatrix>(
      comm_, local_coefficient_size, global_coefficient_size_,
      global_coefficient_size_, objective_I.data(), objective_J.data(),
      objective_data.data(), coefficient_offsets_.GetData(),
      coefficient_offsets_.GetData());
   auto constraints = std::make_unique<mfem::HypreParMatrix>(
      comm_, local_constraint_count, global_constraint_count,
      global_coefficient_size_, constraint_I.data(), constraint_J.data(),
      constraint_data.data(), constraint_offsets_.GetData(),
      coefficient_offsets_.GetData());
   std::unique_ptr<mfem::HypreParMatrix> constraints_transpose(
      constraints->Transpose());

   mfem::Array2D<mfem::HypreParMatrix *> system_blocks(2, 2);
   system_blocks(0, 0) = objective.get();
   system_blocks(0, 1) = constraints_transpose.get();
   system_blocks(1, 0) = constraints.get();
   system_blocks(1, 1) = nullptr;
   A_.reset(mfem::HypreParMatrixFromBlocks(system_blocks));

   rhs_data_jacobian_ = std::make_unique<mfem::HypreParMatrix>(
      comm_, local_kkt_size, global_kkt_size,
      input_space_->GlobalTrueVSize(), system_rhs_I.data(),
      system_rhs_J.data(), system_rhs_data.data(),
      system_offsets_.GetData(), input_space_->GetTrueDofOffsets());
   MFEM_VERIFY(A_->GetNumRows() == local_kkt_size &&
                  rhs_data_jacobian_->GetNumRows() == local_kkt_size,
               "Inconsistent local smoothing block ordering.");
}

void ParQuinticSmoothingSpline1D::SetDataValues(
   const mfem::Vector &local_true_y)
{
   MFEM_VERIFY(rhs_data_jacobian_ != nullptr && rhs_ != nullptr &&
                  system_solution_ != nullptr && coefficients_ != nullptr,
               "Assemble the spline before updating interpolation values.");
   MFEM_VERIFY(local_true_y.Size() == input_space_->GetTrueVSize(),
               "Interpolation values must use the local true-DOF ordering.");

   rhs_data_jacobian_->Mult(local_true_y, *rhs_);
   *system_solution_ = 0.0;
   *coefficients_ = 0.0;
   solved_ = false;
}

void ParQuinticSmoothingSpline1D::RefreshDataValues()
{
   mfem::Vector true_values(input_space_->GetTrueVSize());
   input_->GetTrueDofs(true_values);
   SetDataValues(true_values);
}


QuinticSmoothingSolveResult ParQuinticSmoothingSpline1D::Solve(
   const QuinticSmoothingSolveOptions &options)
{
   MFEM_VERIFY(A_ && rhs_ && system_solution_ && coefficients_,
               "The spline system has not been assembled.");
   MFEM_VERIFY(options.relative_tolerance >= 0.0 &&
                  options.absolute_tolerance >= 0.0,
               "Solver tolerances must be nonnegative.");
   MFEM_VERIFY(options.maximum_iterations > 0 &&
                  options.krylov_dimension > 0,
               "GMRES iteration limits must be positive.");

   QuinticSmoothingSolveResult result;
   result.direct_solver = options.direct_solver;
   *system_solution_ = 0.0;
   *coefficients_ = 0.0;
   const mfem::real_t rnorm_0 =
      mfem::GlobalLpNorm(2, rhs_->Norml2(), comm_);
   mfem::real_t rnorm_f = -1.0;
   if (options.direct_solver)
   {
      if (direct_solver_ == nullptr)
      {
         auto solver = std::make_unique<mfem::MUMPSSolver>(comm_);
         solver->SetPrintLevel(options.print_level);
         solver->SetMatrixSymType(
            mfem::MUMPSSolver::SYMMETRIC_INDEFINITE);
         solver->SetOperator(*A_);
         direct_solver_ = std::move(solver);
      }
      direct_solver_->Mult(*rhs_, *system_solution_);
      mfem::Vector residual(rhs_->Size());
      A_->Mult(*system_solution_, residual);
      residual.Add(-1.0, *rhs_);
      rnorm_f = mfem::GlobalLpNorm(2, residual.Norml2(), comm_);
      const mfem::real_t tolerance =
         std::max(options.absolute_tolerance,
                  options.relative_tolerance * rnorm_0);
      result.converged = rnorm_f <= tolerance;
      result.initial_norm = rnorm_0;
      result.final_norm = rnorm_f;
   }
   else
   {
      direct_solver_.reset();
      mfem::GMRESSolver gmres(comm_);
      gmres.SetRelTol(options.relative_tolerance);
      gmres.SetAbsTol(options.absolute_tolerance);
      gmres.SetMaxIter(options.maximum_iterations);
      gmres.SetKDim(options.krylov_dimension);
      gmres.SetPrintLevel(options.print_level);
      gmres.SetOperator(*A_);
      gmres.Mult(*rhs_, *system_solution_);
      result.converged = gmres.GetConverged();
      result.iterations = gmres.GetNumIterations();
      result.initial_norm = gmres.GetInitialNorm();
      result.final_norm = gmres.GetFinalNorm();
   }

   const int local_coefficient_size = coefficients_->Size();
   const mfem::real_t *system_data = system_solution_->HostRead();
   mfem::real_t *coefficient_data = coefficients_->HostWrite();
   std::copy(system_data, system_data + local_coefficient_size,
             coefficient_data);
   solved_ = result.converged;

   if (options.require_convergence)
   {
      MFEM_VERIFY(result.converged,
                  "Linear solver did not converge for the quintic smoothing spline system.");
   }
   return result;
}

const mfem::HypreParMatrix &ParQuinticSmoothingSpline1D::SystemMatrix() const
{
   MFEM_VERIFY(A_ != nullptr, "The spline matrix has not been assembled.");
   return *A_;
}

const mfem::HypreParMatrix &
ParQuinticSmoothingSpline1D::RightHandSideDataJacobian() const
{
   MFEM_VERIFY(rhs_data_jacobian_ != nullptr,
               "The right-hand-side data Jacobian has not been assembled.");
   return *rhs_data_jacobian_;
}

const mfem::HypreParVector &ParQuinticSmoothingSpline1D::RightHandSide() const
{
   MFEM_VERIFY(rhs_ != nullptr, "The spline right-hand side is unavailable.");
   return *rhs_;
}

const mfem::HypreParVector &ParQuinticSmoothingSpline1D::CoefficientVector() const
{
   MFEM_VERIFY(coefficients_ != nullptr,
               "The spline coefficient vector is unavailable.");
   return *coefficients_;
}

std::array<mfem::real_t, ParQuinticSmoothingSpline1D::coefficients_per_interval>
ParQuinticSmoothingSpline1D::LocalCoefficients(int local_element) const
{
   MFEM_VERIFY(solved_, "Solve the spline system before reading coefficients.");
   MFEM_VERIFY(local_element >= 0 &&
                  local_element < static_cast<int>(intervals_.size()),
               "Invalid local element number.");

   const mfem::real_t *coefficient_data = coefficients_->HostRead();
   std::array<mfem::real_t, coefficients_per_interval> result;
   const int offset = coefficients_per_interval * local_element;
   std::copy(coefficient_data + offset,
             coefficient_data + offset + coefficients_per_interval,
             result.begin());
   return result;
}


//
// y(x) = (a[n] * t + a[n-1]) *t + a[n-2] + ...
// y'(x) = n * a[n] * t^(n-1) + ...
mfem::real_t ParQuinticSmoothingSpline1D::EvaluateLocalElement(
   int local_element, mfem::real_t x, int d) const
{
   MFEM_VERIFY(local_element >= 0 &&
                  local_element < static_cast<int>(intervals_.size()),
               "Invalid local element number.");
   const Interval &interval =
      intervals_[static_cast<std::size_t>(local_element)];
   MFEM_VERIFY((x > interval.x_left || Near(x, interval.x_left)) &&
                  (x < interval.x_right || Near(x, interval.x_right)),
               "Evaluation point is outside the requested local interval.");
   MFEM_VERIFY(d >= 0 && d <= order,
               "Derivative order must be between zero and five.");

   const auto a = LocalCoefficients(local_element);
   const mfem::real_t h = interval.x_right - interval.x_left;
   const mfem::real_t t = (x - interval.x_right) / h;
   mfem::real_t value = a[order] * FallingFactorial(order, d);
   for (int j = order - 1; j >= d; --j) { value = value * t + a[j] * FallingFactorial(j, d); }
   value /= std::pow(h, d);
   return value;
}

bool ParQuinticSmoothingSpline1D::TryEvaluateLocal(mfem::real_t x,
                                          mfem::real_t &value, int d) const
{
   MFEM_VERIFY(solved_, "Solve the spline system before evaluation.");
   for (const Interval &interval : intervals_)
   {
      if ((x > interval.x_left || Near(x, interval.x_left)) &&
          (x < interval.x_right || Near(x, interval.x_right)))
      {
         value = EvaluateLocalElement(interval.local_element, x, d);
         return true;
      }
   }
   return false;
}

mfem::real_t ParQuinticSmoothingSpline1D::Evaluate(mfem::real_t x, int d) const
{
   MFEM_VERIFY(solved_, "Solve the spline system before evaluation.");

   // A collective consistency check prevents one rank from silently
   // evaluating a different point than the other ranks.
   const int local_x_is_finite = std::isfinite(x) ? 1 : 0;
   int every_x_is_finite = 0;
   MPI_Allreduce(&local_x_is_finite, &every_x_is_finite, 1, MPI_INT,
                 MPI_MIN, comm_);
   MFEM_VERIFY(every_x_is_finite == 1,
               "Evaluate requires a finite coordinate on every rank.");

   mfem::real_t minimum_requested_x = 0.0;
   mfem::real_t maximum_requested_x = 0.0;
   MPI_Allreduce(&x, &minimum_requested_x, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MIN, comm_);
   MPI_Allreduce(&x, &maximum_requested_x, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MAX, comm_);
   MFEM_VERIFY(minimum_requested_x == maximum_requested_x,
               "Every rank must call Evaluate with the same coordinate.");

   MFEM_VERIFY(x >= global_x_left_ && x <= global_x_right_,
               "Evaluation point is outside the global spline interval.");

   mfem::real_t value = 0.0;
   const bool can_evaluate = TryEvaluateLocal(x, value, d);
   const int no_candidate = std::numeric_limits<int>::max();
   const int local_candidate_rank = can_evaluate ? rank_ : no_candidate;
   int evaluator_rank = no_candidate;
   MPI_Allreduce(&local_candidate_rank, &evaluator_rank, 1, MPI_INT,
                 MPI_MIN, comm_);
   MFEM_VERIFY(evaluator_rank != no_candidate,
               "No MPI rank owns an interval containing the requested point.");

   // More than one rank can contain a partition-interface knot. Selecting the
   // lowest candidate rank makes the result and broadcast root deterministic.
   if (rank_ != evaluator_rank) { value = 0.0; }
   MPI_Bcast(&value, 1, mfem::MPITypeMap<mfem::real_t>::mpi_type,
             evaluator_rank, comm_);
   return value;
}

mfem::Vector ParQuinticSmoothingSpline1D::EvaluateDataGradient(
   mfem::real_t x) const
{
   MFEM_VERIFY(solved_,
               "Solve the spline system before evaluating its gradient.");
   MFEM_VERIFY(direct_solver_ != nullptr,
               "EvaluateDataGradient requires Solve() with "
               "direct_solver=true.");
   MFEM_VERIFY(A_ != nullptr && rhs_data_jacobian_ != nullptr,
               "The spline derivative operators are unavailable.");

   // As in Evaluate(), require every rank to participate with the same point.
   const int local_x_is_finite = std::isfinite(x) ? 1 : 0;
   int every_x_is_finite = 0;
   MPI_Allreduce(&local_x_is_finite, &every_x_is_finite, 1, MPI_INT,
                 MPI_MIN, comm_);
   MFEM_VERIFY(every_x_is_finite == 1,
               "EvaluateDataGradient requires a finite coordinate on every "
               "rank.");

   mfem::real_t minimum_requested_x = 0.0;
   mfem::real_t maximum_requested_x = 0.0;
   MPI_Allreduce(&x, &minimum_requested_x, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MIN, comm_);
   MPI_Allreduce(&x, &maximum_requested_x, 1,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MAX, comm_);
   MFEM_VERIFY(minimum_requested_x == maximum_requested_x,
               "Every rank must call EvaluateDataGradient with the same "
               "coordinate.");
   MFEM_VERIFY(x >= global_x_left_ && x <= global_x_right_,
               "Gradient evaluation point is outside the global spline "
               "interval.");

   // Match Evaluate() at partition interfaces: the lowest candidate rank is
   // selected, and that rank uses its first containing local interval.
   int local_element = -1;
   for (const Interval &interval : intervals_)
   {
      if ((x > interval.x_left || Near(x, interval.x_left)) &&
          (x < interval.x_right || Near(x, interval.x_right)))
      {
         local_element = interval.local_element;
         break;
      }
   }

   const int no_candidate = std::numeric_limits<int>::max();
   const int local_candidate_rank =
      local_element >= 0 ? rank_ : no_candidate;
   int evaluator_rank = no_candidate;
   MPI_Allreduce(&local_candidate_rank, &evaluator_rank, 1, MPI_INT,
                 MPI_MIN, comm_);
   MFEM_VERIFY(evaluator_rank != no_candidate,
               "No MPI rank owns an interval containing the gradient "
               "evaluation point.");

   // grad_a S(x) is zero except for the six normalized basis functions on
   // the selected interval: [1, t, t^2, t^3, t^4, t^5].
   mfem::HypreParVector coefficient_gradient(*A_);
   coefficient_gradient = 0.0;
   if (rank_ == evaluator_rank)
   {
      MFEM_VERIFY(local_element >= 0,
                  "The selected evaluator rank has no containing interval.");
      const Interval &interval =
         intervals_[static_cast<std::size_t>(local_element)];
      const mfem::real_t h = interval.x_right - interval.x_left;
      const mfem::real_t t = (x - interval.x_right) / h;
      const int offset = coefficients_per_interval * local_element;
      mfem::real_t *gradient_data = coefficient_gradient.HostWrite();
      mfem::real_t basis_value = 1.0;
      for (int j = 0; j <= order; ++j)
      {
         gradient_data[offset + j] = basis_value;
         basis_value *= t;
      }
   }

   // A^T lambda = grad_a S(x), followed by grad_y S(x) = B^T lambda.
   mfem::HypreParVector adjoint(*A_);
   adjoint = 0.0;
   direct_solver_->MultTranspose(coefficient_gradient, adjoint);

   mfem::Vector data_gradient(input_space_->GetTrueVSize());
   rhs_data_jacobian_->MultTranspose(adjoint, data_gradient);
   return data_gradient;
}


mfem::Vector ParQuinticSmoothingSpline1D::Evaluate(const mfem::Vector &evaluation_pts, int d) const
{
   MFEM_VERIFY(solved_, "Solve the spline system before evaluation.");

   // local_count --> number of points given rank wants to query
   const int local_count = static_cast<int>(evaluation_pts.Size());
   // collect all local query points and check that they are all valid points
   // that is they are within the valid spline interpolation bounds
   const mfem::real_t *evaluation_pts_data = evaluation_pts.HostRead();
   std::vector<mfem::real_t> local_queries(
      static_cast<std::size_t>(local_count));
   int local_queries_are_valid = 1;
   for (int i = 0; i < local_count; ++i)
   {
      const mfem::real_t query = evaluation_pts_data[i];
      local_queries[static_cast<std::size_t>(i)] = query;
      if (!std::isfinite(query) || query < global_x_left_ ||
          query > global_x_right_)
      {
         local_queries_are_valid = 0;
      }
   }

   int every_query_is_valid = 0;
   MPI_Allreduce(&local_queries_are_valid, &every_query_is_valid, 1,
                 MPI_INT, MPI_MIN, comm_);
   MFEM_VERIFY(every_query_is_valid == 1,
               "Every evaluation point must be finite and remain in the "
               "global spline interval.");
   
   // how many queries per process, expose that info to each rank
   int number_of_ranks = 0;
   MPI_Comm_size(comm_, &number_of_ranks);
   std::vector<int> query_counts(static_cast<std::size_t>(number_of_ranks), 0);
   MPI_Allgather(&local_count, 1, MPI_INT, query_counts.data(), 1, MPI_INT,
                 comm_);

   std::vector<int> query_displacements(
      static_cast<std::size_t>(number_of_ranks), 0);
   int global_query_count = 0;
   for (int r = 0; r < number_of_ranks; ++r)
   {
      MFEM_VERIFY(query_counts[static_cast<std::size_t>(r)] >= 0 &&
                     query_counts[static_cast<std::size_t>(r)] <=
                        std::numeric_limits<int>::max() - global_query_count,
                  "The displaced-node batch exceeds MPI_Allgatherv limits.");
      query_displacements[static_cast<std::size_t>(r)] = global_query_count;
      global_query_count += query_counts[static_cast<std::size_t>(r)];
   }

   std::vector<mfem::real_t> global_queries(
      static_cast<std::size_t>(global_query_count));
   MPI_Allgatherv(local_queries.data(), local_count,
                  mfem::MPITypeMap<mfem::real_t>::mpi_type,
                  global_queries.data(), query_counts.data(),
                  query_displacements.data(),
                  mfem::MPITypeMap<mfem::real_t>::mpi_type, comm_);

   const int no_candidate = std::numeric_limits<int>::max();
   std::vector<int> local_candidate_ranks(
      static_cast<std::size_t>(global_query_count), no_candidate);
   std::vector<mfem::real_t> local_values(
      static_cast<std::size_t>(global_query_count), 0.0);
   for (int i = 0; i < global_query_count; ++i)
   {
      mfem::real_t value = 0.0;
      if (TryEvaluateLocal(global_queries[static_cast<std::size_t>(i)],
                           value, d))
      {
         local_candidate_ranks[static_cast<std::size_t>(i)] = rank_;
         local_values[static_cast<std::size_t>(i)] = value;
      }
   }

   std::vector<int> evaluator_ranks(
      static_cast<std::size_t>(global_query_count), no_candidate);
   MPI_Allreduce(local_candidate_ranks.data(), evaluator_ranks.data(),
                 global_query_count, MPI_INT, MPI_MIN, comm_);
   for (int i = 0; i < global_query_count; ++i)
   {
      MFEM_VERIFY(evaluator_ranks[static_cast<std::size_t>(i)] != no_candidate,
                  "No MPI rank owns an interval containing a displaced node.");
      if (evaluator_ranks[static_cast<std::size_t>(i)] != rank_)
      {
         local_values[static_cast<std::size_t>(i)] = 0.0;
      }
   }

   std::vector<mfem::real_t> global_values(
      static_cast<std::size_t>(global_query_count), 0.0);
   MPI_Allreduce(local_values.data(), global_values.data(),
                 global_query_count,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_SUM, comm_);

   mfem::Vector result(local_count);
   mfem::real_t *result_data = result.HostWrite();
   const int local_offset =
      query_displacements[static_cast<std::size_t>(rank_)];
   for (int i = 0; i < local_count; ++i)
   {
      result_data[i] =
         global_values[static_cast<std::size_t>(local_offset + i)];
   }
   return result;
}

mfem::Vector ParQuinticSmoothingSpline1D::EvaluateDisplacedNodes(
   const mfem::Vector &displacements) const
{
   MFEM_VERIFY(solved_, "Solve the spline system before evaluation.");

   const std::vector<mfem::real_t> &coordinates = node_data_.Coordinates();
   const bool local_size_is_valid =
      coordinates.size() <=
         static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
      displacements.Size() == static_cast<int>(coordinates.size());
   const int local_size_flag = local_size_is_valid ? 1 : 0;
   int every_size_is_valid = 0;
   MPI_Allreduce(&local_size_flag, &every_size_is_valid, 1, MPI_INT,
                 MPI_MIN, comm_);
   MFEM_VERIFY(every_size_is_valid == 1,
               "EvaluateDisplacedNodes requires one displacement for every "
               "coordinate in LocalNodeCoordinates() on each rank.");

   // local_count --> number of points given rank wants to query
   const int local_count = static_cast<int>(coordinates.size());
   // collect all local query points and check that they are all valid points
   // that is they are within the valid spline interpolation bounds
   const mfem::real_t *displacement_data = displacements.HostRead();
   std::vector<mfem::real_t> local_queries(
      static_cast<std::size_t>(local_count));
   int local_queries_are_valid = 1;
   for (int i = 0; i < local_count; ++i)
   {
      const mfem::real_t query =
         coordinates[static_cast<std::size_t>(i)] + displacement_data[i];
      local_queries[static_cast<std::size_t>(i)] = query;
      if (!std::isfinite(query) || query < global_x_left_ ||
          query > global_x_right_)
      {
         local_queries_are_valid = 0;
      }
   }

   int every_query_is_valid = 0;
   MPI_Allreduce(&local_queries_are_valid, &every_query_is_valid, 1,
                 MPI_INT, MPI_MIN, comm_);
   MFEM_VERIFY(every_query_is_valid == 1,
               "Every displaced node must be finite and remain in the "
               "global spline interval.");
   
   // how many queries per process, expose that info to each rank
   int number_of_ranks = 0;
   MPI_Comm_size(comm_, &number_of_ranks);
   std::vector<int> query_counts(static_cast<std::size_t>(number_of_ranks), 0);
   MPI_Allgather(&local_count, 1, MPI_INT, query_counts.data(), 1, MPI_INT,
                 comm_);

   std::vector<int> query_displacements(
      static_cast<std::size_t>(number_of_ranks), 0);
   int global_query_count = 0;
   for (int r = 0; r < number_of_ranks; ++r)
   {
      MFEM_VERIFY(query_counts[static_cast<std::size_t>(r)] >= 0 &&
                     query_counts[static_cast<std::size_t>(r)] <=
                        std::numeric_limits<int>::max() - global_query_count,
                  "The displaced-node batch exceeds MPI_Allgatherv limits.");
      query_displacements[static_cast<std::size_t>(r)] = global_query_count;
      global_query_count += query_counts[static_cast<std::size_t>(r)];
   }

   std::vector<mfem::real_t> global_queries(
      static_cast<std::size_t>(global_query_count));
   MPI_Allgatherv(local_queries.data(), local_count,
                  mfem::MPITypeMap<mfem::real_t>::mpi_type,
                  global_queries.data(), query_counts.data(),
                  query_displacements.data(),
                  mfem::MPITypeMap<mfem::real_t>::mpi_type, comm_);

   const int no_candidate = std::numeric_limits<int>::max();
   std::vector<int> local_candidate_ranks(
      static_cast<std::size_t>(global_query_count), no_candidate);
   std::vector<mfem::real_t> local_values(
      static_cast<std::size_t>(global_query_count), 0.0);
   for (int i = 0; i < global_query_count; ++i)
   {
      mfem::real_t value = 0.0;
      if (TryEvaluateLocal(global_queries[static_cast<std::size_t>(i)], value))
      {
         local_candidate_ranks[static_cast<std::size_t>(i)] = rank_;
         local_values[static_cast<std::size_t>(i)] = value;
      }
   }

   std::vector<int> evaluator_ranks(
      static_cast<std::size_t>(global_query_count), no_candidate);
   MPI_Allreduce(local_candidate_ranks.data(), evaluator_ranks.data(),
                 global_query_count, MPI_INT, MPI_MIN, comm_);
   for (int i = 0; i < global_query_count; ++i)
   {
      MFEM_VERIFY(evaluator_ranks[static_cast<std::size_t>(i)] != no_candidate,
                  "No MPI rank owns an interval containing a displaced node.");
      if (evaluator_ranks[static_cast<std::size_t>(i)] != rank_)
      {
         local_values[static_cast<std::size_t>(i)] = 0.0;
      }
   }

   std::vector<mfem::real_t> global_values(
      static_cast<std::size_t>(global_query_count), 0.0);
   MPI_Allreduce(local_values.data(), global_values.data(),
                 global_query_count,
                 mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_SUM, comm_);

   mfem::Vector result(local_count);
   mfem::real_t *result_data = result.HostWrite();
   const int local_offset =
      query_displacements[static_cast<std::size_t>(rank_)];
   for (int i = 0; i < local_count; ++i)
   {
      result_data[i] =
         global_values[static_cast<std::size_t>(local_offset + i)];
   }
   return result;
}

} // namespace spline
