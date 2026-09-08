#ifndef PAR_LINEAR_1D_NODE_DATA_HPP
#define PAR_LINEAR_1D_NODE_DATA_HPP

#include "mfem.hpp"

#include <algorithm>
#include <vector>

namespace spline
{

/// Coordinate, value, and ownership information for one rank-local node.
struct NodeSample
{
   mfem::real_t x = 0.0;
   mfem::real_t value = 0.0;

   int local_vertex = -1;
   int local_dof = -1;

   // Global true-DOF number, valid on owned and non-owned shared copies.
   HYPRE_BigInt global_true_dof = -1;

   // Nonnegative only when this rank owns the corresponding true DOF.
   int local_true_dof = -1;

   // These flags are independent. An interface node owned by this rank has
   // both is_owned == true and is_shared == true.
   bool is_owned = false;
   bool is_shared = false;
};

/**
 * Collect the rank-local nodal coordinates and values of a scalar, order-one
 * H1 ParGridFunction on a conforming one-dimensional ParMesh.
 *
 * Nodes() is the union of:
 *   1. nodes whose true DOFs are owned by this rank, and
 *   2. shared interface nodes whose true DOFs may be owned by another rank.
 *
 * Under the stated P1 H1 assumptions, this union is the complete set of local
 * mesh vertices. No face-neighbor or ghost-element data are requested.
 *
 * Construction performs MPI communication and therefore must be called by
 * every rank in the ParFiniteElementSpace communicator.
 */
class ParLinear1DNodeData
{
public:
   explicit ParLinear1DNodeData(const mfem::ParGridFunction &input)
      : pfes_(input.ParFESpace()),
        pmesh_(pfes_ ? pfes_->GetParMesh() : nullptr),
        rank_(pfes_ ? pfes_->GetMyRank() : -1)
   {
      ValidateInput();
      Collect(input);
   }

   int Rank() const { return rank_; }

   /// All nodes available on this rank, sorted by physical coordinate.
   const std::vector<NodeSample> &Nodes() const { return nodes_; }

   /// Nodes for which this rank owns the corresponding true DOF.
   const std::vector<NodeSample> &OwnedNodes() const { return owned_nodes_; }

   /// Partition-interface nodes, including those owned by another rank.
   const std::vector<NodeSample> &SharedNodes() const { return shared_nodes_; }

   /// Convenience arrays corresponding exactly to Nodes().
   const std::vector<mfem::real_t> &Coordinates() const
   {
      return coordinates_;
   }

   const std::vector<mfem::real_t> &FunctionValues() const
   {
      return function_values_;
   }

private:
   mfem::ParFiniteElementSpace *pfes_ = nullptr; // not owned
   mfem::ParMesh *pmesh_ = nullptr;              // not owned
   int rank_ = -1;

   std::vector<NodeSample> nodes_;
   std::vector<NodeSample> owned_nodes_;
   std::vector<NodeSample> shared_nodes_;
   std::vector<mfem::real_t> coordinates_;
   std::vector<mfem::real_t> function_values_;

   static int DecodeDof(int encoded_dof)
   {
      return encoded_dof >= 0 ? encoded_dof : -1 - encoded_dof;
   }

   static mfem::real_t ReadSignedDof(const mfem::real_t *data,
                                     int encoded_dof)
   {
      return encoded_dof >= 0 ? data[encoded_dof]
                              : -data[-1 - encoded_dof];
   }

   void ValidateInput() const
   {
      MFEM_VERIFY(pfes_ != nullptr,
                  "ParLinear1DNodeData requires a ParGridFunction with a "
                  "valid ParFiniteElementSpace.");
      MFEM_VERIFY(pmesh_ != nullptr,
                  "ParLinear1DNodeData requires a valid ParMesh.");
      MFEM_VERIFY(pmesh_->Dimension() == 1,
                  "ParLinear1DNodeData requires a one-dimensional mesh.");
      MFEM_VERIFY(pmesh_->SpaceDimension() == 1,
                  "The mesh must be embedded in one-dimensional physical "
                  "space.");
      MFEM_VERIFY(pmesh_->Conforming(),
                  "ParLinear1DNodeData requires a conforming mesh.");
      MFEM_VERIFY(pfes_->GetVDim() == 1,
                  "ParLinear1DNodeData requires a scalar grid function.");
      MFEM_VERIFY(
         dynamic_cast<const mfem::H1_FECollection *>(pfes_->FEColl()) != nullptr,
         "ParLinear1DNodeData requires an H1 finite element collection.");
      MFEM_VERIFY(pfes_->GetMaxElementOrder() == 1,
                  "ParLinear1DNodeData requires order-one elements.");
      MFEM_VERIFY(pfes_->GetNDofs() == pmesh_->GetNV(),
                  "Linear H1 elements must have one scalar local DOF per "
                  "local mesh vertex.");
   }

   void Collect(const mfem::ParGridFunction &input)
   {
      // Extract the owned true DOFs and apply the parallel prolongation
      // operator. This gives every rank a consistent value for each shared
      // local copy without requesting face-neighbor element data.
      mfem::Vector true_dofs(pfes_->GetTrueVSize());
      input.GetTrueDofs(true_dofs);

      mfem::ParGridFunction synchronized(pfes_);
      synchronized.SetFromTrueDofs(true_dofs);
      const mfem::real_t *values = synchronized.HostRead();

      std::vector<bool> is_shared(
         static_cast<std::size_t>(pmesh_->GetNV()), false);

      // In one dimension, MFEM's shared faces are shared vertices, and
      // GetSharedFace() returns the corresponding local vertex index.
      for (int sf = 0; sf < pmesh_->GetNSharedFaces(); ++sf)
      {
         const int vertex = pmesh_->GetSharedFace(sf);
         MFEM_VERIFY(vertex >= 0 && vertex < pmesh_->GetNV(),
                     "Invalid shared vertex index returned by ParMesh.");
         is_shared[static_cast<std::size_t>(vertex)] = true;
      }

      mfem::Array<int> vertex_dofs;
      nodes_.reserve(static_cast<std::size_t>(pmesh_->GetNV()));

      for (int vertex = 0; vertex < pmesh_->GetNV(); ++vertex)
      {
         pfes_->GetVertexDofs(vertex, vertex_dofs);
         MFEM_VERIFY(vertex_dofs.Size() == 1,
                     "Expected exactly one scalar H1 DOF at each vertex.");

         const int encoded_dof = vertex_dofs[0];
         const int local_dof = DecodeDof(encoded_dof);
         const int local_true_dof = pfes_->GetLocalTDofNumber(local_dof);

         NodeSample sample;
         sample.x = pmesh_->GetVertex(vertex)[0];
         sample.value = ReadSignedDof(values, encoded_dof);
         sample.local_vertex = vertex;
         sample.local_dof = local_dof;
         sample.global_true_dof = pfes_->GetGlobalTDofNumber(local_dof);
         sample.local_true_dof = local_true_dof;
         MFEM_VERIFY(sample.global_true_dof >= 0,
                     "Could not determine a node's global true-DOF number.");
         sample.is_owned = local_true_dof >= 0;
         sample.is_shared = is_shared[static_cast<std::size_t>(vertex)];
         nodes_.push_back(sample);
      }

      std::sort(nodes_.begin(), nodes_.end(),
                [](const NodeSample &a, const NodeSample &b)
                {
                   return a.x < b.x;
                });

      coordinates_.resize(nodes_.size());
      function_values_.resize(nodes_.size());

      for (std::size_t i = 0; i < nodes_.size(); ++i)
      {
         const NodeSample &sample = nodes_[i];
         coordinates_[i] = sample.x;
         function_values_[i] = sample.value;

         if (sample.is_owned)
         {
            owned_nodes_.push_back(sample);
         }
         if (sample.is_shared)
         {
            shared_nodes_.push_back(sample);
         }
      }
   }
};

} // namespace spline

#endif // PAR_LINEAR_1D_NODE_DATA_HPP
