#include "quartic_spline_1d.hpp"

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

constexpr int neighbor_count_tag = 23141;
constexpr int neighbor_data_tag = 23142;

struct SharedIntervalRecord
{
   mfem::real_t shared_x;
   mfem::real_t x_left;
   mfem::real_t x_right;
   HYPRE_BigInt global_interval;
};

static_assert(std::is_trivially_copyable<SharedIntervalRecord>::value,
              "SharedIntervalRecord must be suitable for MPI_BYTE transfer.");

struct GlobalNodeRecord
{
   mfem::real_t x;
   HYPRE_BigInt global_true_dof;
};

static_assert(std::is_trivially_copyable<GlobalNodeRecord>::value,
              "GlobalNodeRecord must be suitable for MPI_BYTE transfer.");

struct VertexIncidence
{
   int interval = -1;
   bool is_left = false;
};

} // namespace

ParQuarticSpline1D::ParQuarticSpline1D(
   const mfem::ParGridFunction &input,
   mfem::real_t coordinate_tolerance)
   : input_(&input),
     input_space_(input.ParFESpace()),
     mesh_(input_space_ ? input_space_->GetParMesh() : nullptr),
     comm_(mesh_ ? mesh_->GetComm() : MPI_COMM_NULL),
     rank_(mesh_ ? mesh_->GetMyRank() : -1),
     coordinate_tolerance_(coordinate_tolerance),
     node_data_(input)
{
   MFEM_VERIFY(mesh_ != nullptr, "A valid ParMesh is required.");
   MFEM_VERIFY(coordinate_tolerance_ >= 0.0,
               "coordinate_tolerance must be nonnegative.");

   BuildIntervals();
   CompleteOffRankNeighbors();
   BuildBoundaryStencils();
   Assemble();
}

bool ParQuarticSpline1D::Near(mfem::real_t a, mfem::real_t b) const
{
   const mfem::real_t scale =
      std::max<mfem::real_t>({1.0, std::abs(a), std::abs(b)});
   const mfem::real_t automatic_tolerance =
      64.0 * std::numeric_limits<mfem::real_t>::epsilon() * scale;
   return std::abs(a - b) <=
          std::max(coordinate_tolerance_, automatic_tolerance);
}

mfem::real_t ParQuarticSpline1D::IntegerPower(mfem::real_t x, int exponent)
{
   MFEM_ASSERT(exponent >= 0, "A nonnegative exponent is required.");
   mfem::real_t result = 1.0;
   for (int i = 0; i < exponent; ++i) { result *= x; }
   return result;
}

mfem::real_t ParQuarticSpline1D::FallingFactorial(int j, int derivative)
{
   MFEM_ASSERT(j >= derivative && derivative >= 0,
               "Invalid monomial derivative.");
   mfem::real_t result = 1.0;
   for (int k = 0; k < derivative; ++k) { result *= (j - k); }
   return result;
}

HYPRE_BigInt ParQuarticSpline1D::CoefficientColumn(
   HYPRE_BigInt global_interval, int j)
{
   MFEM_ASSERT(global_interval >= 0, "Invalid global interval number.");
   MFEM_ASSERT(j >= 0 && j < coefficients_per_interval,
               "Invalid quartic coefficient number.");
   return coefficients_per_interval * global_interval + j;
}

void ParQuarticSpline1D::BuildIntervals()
{
   const int local_ne = mesh_->GetNE();
   MFEM_VERIFY(local_ne > 0,
               "ParQuarticSpline1D currently requires every MPI rank to own "
               "at least one interval.");

   global_number_of_intervals_ =
      static_cast<HYPRE_BigInt>(mesh_->GetGlobalNE());
   MFEM_VERIFY(global_number_of_intervals_ > 0,
               "At least one mesh interval is required.");
   MFEM_VERIFY(global_number_of_intervals_ <=
                  std::numeric_limits<HYPRE_BigInt>::max() /
                     coefficients_per_interval,
               "The quartic coefficient numbering overflows HYPRE_BigInt.");
   global_system_size_ =
      coefficients_per_interval * global_number_of_intervals_;

   const HYPRE_BigInt local_system_size =
      coefficients_per_interval * static_cast<HYPRE_BigInt>(local_ne);
   HYPRE_BigInt local_sizes[1] = {local_system_size};
   mfem::Array<HYPRE_BigInt> *offset_arrays[1] = {&coefficient_offsets_};
   mesh_->GenerateOffsets(1, local_sizes, offset_arrays);

   MFEM_VERIFY(coefficient_offsets_.Last() == global_system_size_,
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

void ParQuarticSpline1D::CompleteOffRankNeighbors()
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

ParQuarticSpline1D::BoundaryStencil
ParQuarticSpline1D::MakeBoundaryStencil(
   const std::array<BoundaryNode, 3> &nodes)
{
   const mfem::real_t x0 = nodes[0].x;
   const mfem::real_t x1 = nodes[1].x;
   const mfem::real_t x2 = nodes[2].x;
   MFEM_VERIFY(x0 != x1 && x0 != x2 && x1 != x2,
               "Finite-difference boundary nodes must be distinct.");

   BoundaryStencil stencil;
   stencil.nodes = nodes;

   // Derivatives at x0 of the quadratic Lagrange interpolant through the
   // three supplied nodes. These formulas support both the increasing-x
   // left stencil and the decreasing-x right stencil.
   stencil.first_derivative_weights[0] =
      (2.0 * x0 - x1 - x2) / ((x0 - x1) * (x0 - x2));
   stencil.first_derivative_weights[1] =
      (x0 - x2) / ((x1 - x0) * (x1 - x2));
   stencil.first_derivative_weights[2] =
      (x0 - x1) / ((x2 - x0) * (x2 - x1));

   stencil.second_derivative_weights[0] =
      2.0 / ((x0 - x1) * (x0 - x2));
   stencil.second_derivative_weights[1] =
      2.0 / ((x1 - x0) * (x1 - x2));
   stencil.second_derivative_weights[2] =
      2.0 / ((x2 - x0) * (x2 - x1));
   return stencil;
}

void ParQuarticSpline1D::BuildBoundaryStencils()
{
   // At most six owned nodes per rank are sufficient to recover the first
   // and last three global nodes, independent of the mesh partition.
   const std::vector<NodeSample> &owned = node_data_.OwnedNodes();
   std::vector<GlobalNodeRecord> local_candidates;
   local_candidates.reserve(std::min<std::size_t>(6, owned.size()));

   const auto append_candidate =
      [&local_candidates](const NodeSample &node)
      {
         const auto duplicate =
            std::find_if(local_candidates.begin(), local_candidates.end(),
                         [&node](const GlobalNodeRecord &record)
                         {
                            return record.global_true_dof ==
                                   node.global_true_dof;
                         });
         if (duplicate == local_candidates.end())
         {
            local_candidates.push_back({node.x, node.global_true_dof});
         }
      };

   const std::size_t end_count = std::min<std::size_t>(3, owned.size());
   for (std::size_t i = 0; i < end_count; ++i)
   {
      append_candidate(owned[i]);
   }
   for (std::size_t i = owned.size() - end_count; i < owned.size(); ++i)
   {
      append_candidate(owned[i]);
   }

   const std::size_t local_bytes_size =
      local_candidates.size() * sizeof(GlobalNodeRecord);
   MFEM_VERIFY(local_bytes_size <=
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
               "Boundary-node metadata exceeds an MPI message count.");
   const int local_bytes = static_cast<int>(local_bytes_size);

   int number_of_ranks = 0;
   MPI_Comm_size(comm_, &number_of_ranks);
   std::vector<int> byte_counts(static_cast<std::size_t>(number_of_ranks), 0);
   MPI_Allgather(&local_bytes, 1, MPI_INT, byte_counts.data(), 1, MPI_INT,
                 comm_);

   std::vector<int> byte_displacements(
      static_cast<std::size_t>(number_of_ranks), 0);
   int total_bytes = 0;
   for (int r = 0; r < number_of_ranks; ++r)
   {
      MFEM_VERIFY(byte_counts[static_cast<std::size_t>(r)] >= 0 &&
                     byte_counts[static_cast<std::size_t>(r)] <=
                        std::numeric_limits<int>::max() - total_bytes,
                  "Boundary-node metadata exceeds MPI_Allgatherv limits.");
      byte_displacements[static_cast<std::size_t>(r)] = total_bytes;
      total_bytes += byte_counts[static_cast<std::size_t>(r)];
   }
   MFEM_VERIFY(total_bytes % static_cast<int>(sizeof(GlobalNodeRecord)) == 0,
               "Received malformed boundary-node metadata.");

   std::vector<GlobalNodeRecord> global_candidates(
      static_cast<std::size_t>(total_bytes) / sizeof(GlobalNodeRecord));
   MPI_Allgatherv(local_candidates.data(), local_bytes, MPI_BYTE,
                  global_candidates.data(), byte_counts.data(),
                  byte_displacements.data(), MPI_BYTE, comm_);

   std::sort(global_candidates.begin(), global_candidates.end(),
             [](const GlobalNodeRecord &a, const GlobalNodeRecord &b)
             {
                return a.global_true_dof < b.global_true_dof;
             });
   global_candidates.erase(
      std::unique(global_candidates.begin(), global_candidates.end(),
                  [](const GlobalNodeRecord &a, const GlobalNodeRecord &b)
                  {
                     return a.global_true_dof == b.global_true_dof;
                  }),
      global_candidates.end());
   std::sort(global_candidates.begin(), global_candidates.end(),
             [](const GlobalNodeRecord &a, const GlobalNodeRecord &b)
             {
                if (a.x != b.x) { return a.x < b.x; }
                return a.global_true_dof < b.global_true_dof;
             });

   MFEM_VERIFY(global_candidates.size() >= 3,
               "At least three global interpolation nodes are required for "
               "the finite-difference boundary conditions.");
   for (std::size_t i = 1; i < global_candidates.size(); ++i)
   {
      MFEM_VERIFY(!Near(global_candidates[i - 1].x,
                        global_candidates[i].x),
                  "Distinct global true DOFs occupy the same coordinate.");
   }

   std::array<BoundaryNode, 3> left_nodes;
   std::array<BoundaryNode, 3> right_nodes;
   for (std::size_t i = 0; i < 3; ++i)
   {
      const GlobalNodeRecord &left = global_candidates[i];
      const GlobalNodeRecord &right =
         global_candidates[global_candidates.size() - 1 - i];
      left_nodes[i] = {left.x, left.global_true_dof};
      right_nodes[i] = {right.x, right.global_true_dof};
   }

   left_boundary_stencil_ = MakeBoundaryStencil(left_nodes);
   right_boundary_stencil_ = MakeBoundaryStencil(right_nodes);
}

void ParQuarticSpline1D::AddEntry(std::vector<RowEntry> &row,
                                  HYPRE_BigInt global_interval,
                                  int coefficient,
                                  mfem::real_t value) const
{
   if (value == 0.0) { return; }
   row.push_back(
      {CoefficientColumn(global_interval, coefficient), value});
}

void ParQuarticSpline1D::AddDerivative(
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

void ParQuarticSpline1D::BuildCSR(
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

void ParQuarticSpline1D::Assemble()
{
   // Assemble() is public, so release vectors before replacing the matrices
   // whose parallel partitions they use.
   coefficients_.reset();
   rhs_.reset();
   rhs_data_jacobian_.reset();
   A_.reset();

   const int local_ne = static_cast<int>(intervals_.size());
   const int local_rows = coefficients_per_interval * local_ne;
   std::vector<std::vector<RowEntry>> rows(
      static_cast<std::size_t>(local_rows));
   std::vector<std::vector<RowEntry>> rhs_data_rows(
      static_cast<std::size_t>(local_rows));

   const auto add_rhs_data_entry =
      [&rhs_data_rows](int local_row, HYPRE_BigInt global_true_dof,
                       mfem::real_t value)
      {
         MFEM_ASSERT(local_row >= 0 &&
                        local_row < static_cast<int>(rhs_data_rows.size()),
                     "Invalid local right-hand-side row.");
         MFEM_ASSERT(global_true_dof >= 0,
                     "Invalid global true-DOF column.");
         if (value != 0.0)
         {
            rhs_data_rows[static_cast<std::size_t>(local_row)].push_back(
               {global_true_dof, value});
         }
      };

   for (const Interval &interval : intervals_)
   {
      const int base_row =
         coefficients_per_interval * interval.local_element;
      const mfem::real_t h = interval.x_right - interval.x_left;
      MFEM_ASSERT(h > 0.0, "A positive interval length is required.");

      // Multiplying an order-d derivative equation by h^d makes its units
      // consistent with the interpolation equations.  The same factor is
      // applied to every term in a continuity row and to its right-hand side,
      // so this scaling does not change the spline coefficients.
      const mfem::real_t derivative_row_scale[order + 1] =
      {
         1.0, h, h * h, h * h * h, h * h * h * h
      };

      // Row 0: S_e(x_left) = y_left.
      AddDerivative(rows[static_cast<std::size_t>(base_row)],
                    interval.global_interval, interval.x_right,
                    h, interval.x_left, 0, 1.0);
      add_rhs_data_entry(base_row, interval.left_global_true_dof, 1.0);

      // Row 1: S_e(x_right) = a_{e,0} = y_right.
      AddEntry(rows[static_cast<std::size_t>(base_row + 1)],
               interval.global_interval, 0, 1.0);
      add_rhs_data_entry(base_row + 1,
                         interval.right_global_true_dof, 1.0);

      // Row 2: C1 at the right knot, or the right endpoint condition.
      if (interval.physical_right_boundary)
      {
         AddDerivative(rows[static_cast<std::size_t>(base_row + 2)],
                       interval.global_interval, interval.x_right,
                       h, interval.x_right, 1,
                       derivative_row_scale[1]);
         for (std::size_t k = 0; k < 3; ++k)
         {
            add_rhs_data_entry(
               base_row + 2,
               right_boundary_stencil_.nodes[k].global_true_dof,
               derivative_row_scale[1] *
                  right_boundary_stencil_.first_derivative_weights[k]);
         }
      }
      else
      {
         MFEM_VERIFY(interval.right_neighbor.IsSet(),
                     "Missing the interval to the right.");
         AddDerivative(rows[static_cast<std::size_t>(base_row + 2)],
                       interval.right_neighbor.global_interval,
                       interval.right_neighbor.x_right,
                       interval.right_neighbor.x_right -
                          interval.right_neighbor.x_left,
                       interval.x_right, 1, derivative_row_scale[1]);
         AddDerivative(rows[static_cast<std::size_t>(base_row + 2)],
                       interval.global_interval, interval.x_right,
                       h, interval.x_right, 1,
                       -derivative_row_scale[1]);
      }

      // Rows 3 and 4: C2/C3 at the left knot, or the two left
      // endpoint conditions. Assigning C1 to the left interval and C2/C3
      // to the right interval gives exactly five owned rows per interval.
      if (interval.physical_left_boundary)
      {
         AddDerivative(rows[static_cast<std::size_t>(base_row + 3)],
                       interval.global_interval, interval.x_right,
                       h, interval.x_left, 1,
                       derivative_row_scale[1]);
         for (std::size_t k = 0; k < 3; ++k)
         {
            add_rhs_data_entry(
               base_row + 3,
               left_boundary_stencil_.nodes[k].global_true_dof,
               derivative_row_scale[1] *
                  left_boundary_stencil_.first_derivative_weights[k]);
         }

         AddDerivative(rows[static_cast<std::size_t>(base_row + 4)],
                       interval.global_interval, interval.x_right,
                       h, interval.x_left, 2,
                       derivative_row_scale[2]);
         for (std::size_t k = 0; k < 3; ++k)
         {
            add_rhs_data_entry(
               base_row + 4,
               left_boundary_stencil_.nodes[k].global_true_dof,
               derivative_row_scale[2] *
                  left_boundary_stencil_.second_derivative_weights[k]);
         }
      }
      else
      {
         MFEM_VERIFY(interval.left_neighbor.IsSet(),
                     "Missing the interval to the left.");
         AddDerivative(rows[static_cast<std::size_t>(base_row + 3)],
                       interval.global_interval, interval.x_right,
                       h, interval.x_left, 2,
                       derivative_row_scale[2]);
         AddDerivative(rows[static_cast<std::size_t>(base_row + 3)],
                       interval.left_neighbor.global_interval,
                       interval.left_neighbor.x_right,
                       interval.left_neighbor.x_right -
                          interval.left_neighbor.x_left,
                       interval.x_left, 2, -derivative_row_scale[2]);

         AddDerivative(rows[static_cast<std::size_t>(base_row + 4)],
                       interval.global_interval, interval.x_right,
                       h, interval.x_left, 3,
                       derivative_row_scale[3]);
         AddDerivative(rows[static_cast<std::size_t>(base_row + 4)],
                       interval.left_neighbor.global_interval,
                       interval.left_neighbor.x_right,
                       interval.left_neighbor.x_right -
                          interval.left_neighbor.x_left,
                       interval.x_left, 3, -derivative_row_scale[3]);
      }
   }

   std::vector<int> I;
   std::vector<HYPRE_BigInt> J;
   std::vector<mfem::real_t> data;
   BuildCSR(rows, I, J, data);

   std::vector<int> rhs_data_I;
   std::vector<HYPRE_BigInt> rhs_data_J;
   std::vector<mfem::real_t> rhs_data_values;
   BuildCSR(rhs_data_rows, rhs_data_I, rhs_data_J, rhs_data_values);

   A_ = std::make_unique<mfem::HypreParMatrix>(
      comm_, local_rows, global_system_size_, global_system_size_, I.data(),
      J.data(), data.data(), coefficient_offsets_.GetData(),
      coefficient_offsets_.GetData());

   rhs_data_jacobian_ = std::make_unique<mfem::HypreParMatrix>(
      comm_, local_rows, global_system_size_,
      input_space_->GlobalTrueVSize(), rhs_data_I.data(),
      rhs_data_J.data(), rhs_data_values.data(),
      coefficient_offsets_.GetData(), input_space_->GetTrueDofOffsets());

   // A is square with identical row and column partitions. Constructing the
   // vectors from A avoids any dependence on an auxiliary FE space.
   rhs_ = std::make_unique<mfem::HypreParVector>(*A_);
   coefficients_ = std::make_unique<mfem::HypreParVector>(*A_);
   MFEM_VERIFY(rhs_->Size() == local_rows &&
                  coefficients_->Size() == local_rows,
               "Unexpected local Hypre vector size.");

   mfem::Vector true_values(input_space_->GetTrueVSize());
   input_->GetTrueDofs(true_values);
   rhs_data_jacobian_->Mult(true_values, *rhs_);
   *coefficients_ = 0.0;
   solved_ = false;
}

QuarticSolveResult ParQuarticSpline1D::Solve(
   const QuarticSolveOptions &options)
{
   MFEM_VERIFY(A_ && rhs_ && coefficients_,
               "The spline system has not been assembled.");
   MFEM_VERIFY(options.relative_tolerance >= 0.0 &&
                  options.absolute_tolerance >= 0.0,
               "Solver tolerances must be nonnegative.");
   MFEM_VERIFY(options.maximum_iterations > 0 &&
                  options.krylov_dimension > 0,
               "GMRES iteration limits must be positive.");

   QuarticSolveResult result;
   result.direct_solver = options.direct_solver;
   *coefficients_ = 0.0;
   mfem::real_t rnorm_0 = 0.0;
   mfem::real_t rnorm_f = 0.0;
   rnorm_0 = mfem::GlobalLpNorm(2, rhs_->Norml2(), comm_);
   if (options.direct_solver)
   { 
     mfem::MUMPSSolver solver(comm_);
     solver.SetOperator(*A_);
     solver.Mult(*rhs_, *coefficients_);
     mfem::Vector residual(rhs_->Size());
     A_->Mult(*coefficients_, residual);
     residual.Add(-1.0, *rhs_);
     //std::cout << "||r|| = " << residual.Normlinf() << std::endl; 
     rnorm_f = mfem::GlobalLpNorm(2, residual.Norml2(), comm_);
     if (rnorm_f < options.relative_tolerance * rnorm_0 || rnorm_f < options.absolute_tolerance)
     {
        result.converged = true;
     } 
     else
     {
        result.converged = false;
     }
     solved_ = result.converged;
   }
   else
   {
     mfem::GMRESSolver gmres(comm_);
     gmres.SetRelTol(options.relative_tolerance);
     gmres.SetAbsTol(options.absolute_tolerance);
     gmres.SetMaxIter(options.maximum_iterations);
     gmres.SetKDim(options.krylov_dimension);
     gmres.SetPrintLevel(3);
     gmres.SetOperator(*A_);
     gmres.Mult(*rhs_, *coefficients_);
     result.converged = gmres.GetConverged();
     result.iterations = gmres.GetNumIterations();
     result.initial_norm = gmres.GetInitialNorm();
     result.final_norm = gmres.GetFinalNorm();
     solved_ = result.converged;
   }
   

   if (options.require_convergence)
   {
      MFEM_VERIFY(result.converged,
                  "Linear solver did not converge for the quartic spline system.");
   }
   return result;
}

const mfem::HypreParMatrix &ParQuarticSpline1D::SystemMatrix() const
{
   MFEM_VERIFY(A_ != nullptr, "The spline matrix has not been assembled.");
   return *A_;
}

const mfem::HypreParMatrix &
ParQuarticSpline1D::RightHandSideDataJacobian() const
{
   MFEM_VERIFY(rhs_data_jacobian_ != nullptr,
               "The right-hand-side data Jacobian has not been assembled.");
   return *rhs_data_jacobian_;
}

const mfem::HypreParVector &ParQuarticSpline1D::RightHandSide() const
{
   MFEM_VERIFY(rhs_ != nullptr, "The spline right-hand side is unavailable.");
   return *rhs_;
}

const mfem::HypreParVector &ParQuarticSpline1D::CoefficientVector() const
{
   MFEM_VERIFY(coefficients_ != nullptr,
               "The spline coefficient vector is unavailable.");
   return *coefficients_;
}

std::array<mfem::real_t, ParQuarticSpline1D::coefficients_per_interval>
ParQuarticSpline1D::LocalCoefficients(int local_element) const
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

mfem::real_t ParQuarticSpline1D::EvaluateLocalElement(
   int local_element, mfem::real_t x) const
{
   MFEM_VERIFY(local_element >= 0 &&
                  local_element < static_cast<int>(intervals_.size()),
               "Invalid local element number.");
   const Interval &interval =
      intervals_[static_cast<std::size_t>(local_element)];
   MFEM_VERIFY((x > interval.x_left || Near(x, interval.x_left)) &&
                  (x < interval.x_right || Near(x, interval.x_right)),
               "Evaluation point is outside the requested local interval.");

   const auto a = LocalCoefficients(local_element);
   const mfem::real_t h = interval.x_right - interval.x_left;
   const mfem::real_t t = (x - interval.x_right) / h;
   mfem::real_t value = a[order];
   for (int j = order - 1; j >= 0; --j) { value = value * t + a[j]; }
   return value;
}

bool ParQuarticSpline1D::TryEvaluateLocal(mfem::real_t x,
                                          mfem::real_t &value) const
{
   MFEM_VERIFY(solved_, "Solve the spline system before evaluation.");
   for (const Interval &interval : intervals_)
   {
      if ((x > interval.x_left || Near(x, interval.x_left)) &&
          (x < interval.x_right || Near(x, interval.x_right)))
      {
         value = EvaluateLocalElement(interval.local_element, x);
         return true;
      }
   }
   return false;
}

mfem::real_t ParQuarticSpline1D::Evaluate(mfem::real_t x) const
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
   const bool can_evaluate = TryEvaluateLocal(x, value);
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

mfem::Vector ParQuarticSpline1D::EvaluateDisplacedNodes(
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

   const int local_count = static_cast<int>(coordinates.size());
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
