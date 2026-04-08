#include "mfem.hpp"
#include "utilities.hpp"


mfem::HypreParMatrix * GenerateHypreParMatrixFromSparseMatrix(HYPRE_BigInt * rowOffsetsloc, HYPRE_BigInt * colOffsetsloc, mfem::SparseMatrix * Asparse)
{
  int ncols_loc = colOffsetsloc[1] - colOffsetsloc[0];
  int nrows_loc = rowOffsetsloc[1] - rowOffsetsloc[0];
  HYPRE_BigInt ncols_glb, nrows_glb;
  MPI_Allreduce(&nrows_loc, &nrows_glb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&ncols_loc, &ncols_glb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  int * AI          = Asparse->GetI();
  HYPRE_BigInt * AJ = Asparse->GetJ();
  double * Adata    = Asparse->GetData();

  mfem::HypreParMatrix * Ahypre = new mfem::HypreParMatrix(MPI_COMM_WORLD, nrows_loc, nrows_glb, ncols_glb, AI, AJ, Adata, rowOffsetsloc, colOffsetsloc);
  return Ahypre;
}


mfem::HypreParMatrix * GenerateHypreParMatrixFromDiagonal(HYPRE_BigInt * offsetsloc, 
		mfem::Vector & diag)
{
   int n_loc = offsetsloc[1] - offsetsloc[0];
   int n_glb = 0;
   MPI_Allreduce(&n_loc, &n_glb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   
   mfem::SparseMatrix * Dsparse = new mfem::SparseMatrix(n_loc, n_glb);
   mfem::Array<int> cols;
   mfem::Vector entries;
   cols.SetSize(1);
   entries.SetSize(1);
   for(int j = 0; j < n_loc; j++)
   {
     cols[0] = offsetsloc[0] + j;
     entries(0) = diag(j);
     Dsparse->SetRow(j, cols, entries);
   }   
   Dsparse->Finalize();
   mfem::HypreParMatrix * Dhypre = nullptr;
   Dhypre = GenerateHypreParMatrixFromSparseMatrix(offsetsloc, offsetsloc, Dsparse);
   delete Dsparse;
   return Dhypre;   
}

mfem::HypreParMatrix * GenerateProjector(HYPRE_BigInt * reduced_offsets, HYPRE_BigInt * offsets, HYPRE_Int * mask)
{
  int n_cols_loc = offsets[1] - offsets[0];
  int n_cols_glb = 0;
  MPI_Allreduce(&n_cols_loc, &n_cols_glb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  int n_rows_loc = reduced_offsets[1] - reduced_offsets[0];

  mfem::SparseMatrix * Psparse = new mfem::SparseMatrix(n_rows_loc, n_cols_glb);
  mfem::Array<int> cols;
  mfem::Vector entries;
  cols.SetSize(1);
  entries.SetSize(1);

  int row = 0;
  for(int j = 0; j < n_cols_loc; j++)
  {
    if (mask[j] == 1)
    {
      cols[0] = offsets[0] + j;
      entries(0) = 1.0;
      Psparse->SetRow(row, cols, entries);
      row += 1;
    }
  }
  Psparse->Finalize();
  mfem::HypreParMatrix * Phypre = nullptr;
  Phypre = GenerateHypreParMatrixFromSparseMatrix(reduced_offsets, offsets, Psparse);
  delete Psparse;
  return Phypre;
}




mfem::HypreParMatrix * GenerateProjector(HYPRE_BigInt * reduced_offsets, HYPRE_BigInt * offsets, const mfem::HypreParVector & mask)
{
  int n_cols_loc = offsets[1] - offsets[0];
  int n_cols_glb = 0;
  MPI_Allreduce(&n_cols_loc, &n_cols_glb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  int n_rows_loc = reduced_offsets[1] - reduced_offsets[0];

  mfem::SparseMatrix * Psparse = new mfem::SparseMatrix(n_rows_loc, n_cols_glb);
  mfem::Array<int> cols;
  mfem::Vector entries;
  cols.SetSize(1);
  entries.SetSize(1);

  int row = 0;
  for(int j = 0; j < n_cols_loc; j++)
  {
    if (mask(j) > 0.5)
    {
      cols[0] = offsets[0] + j;
      entries(0) = 1.0;
      Psparse->SetRow(row, cols, entries);
      row += 1;
    }
  }
  Psparse->Finalize();
  mfem::HypreParMatrix * Phypre = nullptr;
  Phypre = GenerateHypreParMatrixFromSparseMatrix(reduced_offsets, offsets, Psparse);
  delete Psparse;
  return Phypre;
}




HYPRE_BigInt * offsetsFromLocalSizes(int n, MPI_Comm comm)
{
  
  int nprocs = 0;
  int myrank = 0;
  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm, &nprocs);
  
  HYPRE_BigInt * offsets = new HYPRE_BigInt[2];
  if (myrank == 0)
  {
    offsets[0] = 0;
    offsets[1] = n;
  }
  else
  {
    offsets[0] = 0;
    offsets[1] = 0;
  }
  
  // receive then send
  
  // Receive local size info from processes with rank less than myrank 
  // Populate that as entries of helper
  HYPRE_BigInt * helper;
  if (myrank > 0)
  {
    helper = new HYPRE_BigInt[static_cast<size_t>(myrank)];
  }
  int tag;
  for (int i = 0; i < myrank; i++)
  {
    tag = myrank + i * nprocs;
    MPI_Recv (&(helper[i]), 1, MPI_INT, i, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    offsets[0] += helper[i];
  }

  if (myrank > 0)
  {
    delete[] helper;
  }
  offsets[1] = offsets[0] + n;
  
  // Send local size info to all processes with rank greater than myrank
  for (int i = myrank + 1; i < nprocs; i++)
  {
    tag = i + myrank * nprocs;
    MPI_Send (&n, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
  }
  return offsets;
}


void HypreToMfemOffsets(HYPRE_BigInt * offsets)
{
  if (offsets[1] < offsets[0])
  {
    offsets[1] = offsets[0];
  }
  else
  {
    offsets[1] = offsets[1] + 1;
  }
}



mfem::HypreParMatrix * NonZeroRowMap(const mfem::HypreParMatrix& A)
{     
      mfem::SparseMatrix mergedA;
      const_cast<mfem::HypreParMatrix*>(&A)->MergeDiagAndOffd(mergedA);
      mfem::Array<int> nonZeroRows;
      for (int i = 0; i < mergedA.NumRows(); i++)
      {
         if (!mergedA.RowIsEmpty(i))
         {
            nonZeroRows.Append(i);
         }
      }
      int numNZRows = nonZeroRows.Size();
      mfem::SparseMatrix nzRowMap(numNZRows, A.GetGlobalNumRows());

      for (int i = 0; i < numNZRows; i++)
      {
         int nzRow_global = nonZeroRows[i] + A.RowPart()[0];
         nzRowMap.Set(i, nzRow_global, 1.0);
      }
      nzRowMap.Finalize();

      auto comm = A.GetComm();
      int rows_part[2];
      int cols_part[2];

      int row_offset;
      MPI_Scan(&numNZRows, &row_offset, 1, MPI_INT, MPI_SUM, comm);

      row_offset -= numNZRows;
      rows_part[0] = row_offset;
      rows_part[1] = row_offset + numNZRows;
      for (int i = 0; i < 2; i++)
      {
         cols_part[i] = A.RowPart()[i];
      }
      int glob_nrows;
      int glob_ncols = A.GetGlobalNumRows();
      MPI_Allreduce(&numNZRows, &glob_nrows, 1, MPI_INT, MPI_SUM, comm);

      return new mfem::HypreParMatrix(comm, numNZRows, glob_nrows, glob_ncols,
                                      nzRowMap.GetI(), nzRowMap.GetJ(), nzRowMap.GetData(), 
                                      rows_part, cols_part); 
      // HypreStealOwnership(*out, nzRowMap);
}

mfem::HypreParMatrix * NonZeroColMap(const mfem::HypreParMatrix& A)
{
      auto At = A.Transpose();
      auto mapT = NonZeroRowMap(*At);
      auto out = mapT->Transpose();
      delete mapT;
      delete At;
      return out;
}



DirectSolver::DirectSolver()
   : mfem::Solver(), solver(nullptr)
{  }

DirectSolver::DirectSolver(const mfem::Operator& op)
   : DirectSolver()
{
   SetOperator(op);
}

DirectSolver::~DirectSolver()
{
   if (solver)
   {
      delete solver;
   }
#if defined(MFEM_USE_STRUMPACK)
   if (Astrumpack)
   {
      delete Astrumpack;
   }
#endif
}

void DirectSolver::SetOperator(const mfem::Operator& op)
{
   height = op.NumRows();
   width = op.NumCols();

   auto op_ptr = dynamic_cast<const mfem::HypreParMatrix *>(&op);
   MFEM_VERIFY(op_ptr, "op must be a mfem::HypreParMatrix!");

   if (solver)
   {
      delete solver;
      solver = nullptr;
   }
#if defined(MFEM_USE_STRUMPACK)
   auto directSolver = new mfem::STRUMPACKSolver(op_ptr->GetComm());
   directSolver->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
   directSolver->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
   if (Astrumpack)
   {
      delete Astrumpack;
      Astrumpack = nullptr;
   }
   Astrumpack = new mfem::STRUMPACKRowLocMatrix(*op_ptr);
   directSolver->SetOperator(*Astrumpack);
#elif defined(MFEM_USE_MUMPS)
   auto directSolver = new mfem::MUMPSSolver(op_ptr->GetComm());
   directSolver->SetPrintLevel(0);
   directSolver->SetMatrixSymType(mfem::MUMPSSolver::MatType::UNSYMMETRIC);
   directSolver->SetOperator(*op_ptr);
#elif defined(MFEM_USE_MKL_CPARDISO)
   auto directSolver = new mfem::CPardisoSolver(op_ptr->GetComm());
   directSolver->SetOperator(*op_ptr);
#else
   MFEM_ABORT("DirectSolver will not work unless compiled mfem is with MUMPS, MKL_CPARDISO, or STRUMPACK");
#endif
#if defined(MFEM_USE_STRUMPACK) || defined(MFEM_USE_MUMPS) || defined(MFEM_USE_MKL_CPARDISO)
   solver = directSolver;
#endif
}

void DirectSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const 
{
   MFEM_VERIFY(solver, "SetOperator must be called before Mult!");
   solver->Mult(b, x);
}


mfem::HypreParMatrix * GenerateNullHypreParMatrix(HYPRE_BigInt * rowOffsets, HYPRE_BigInt * colOffsets)
{
  // sparse matrix with no entries, hence a null matrix
  int nnz = 0; // local number of nonzeros
  mfem::HypreParMatrix * A = nullptr;
  
  // local sizes of matrix based on offsets
  HYPRE_Int m = 0, n = 0;
  m = rowOffsets[1] - rowOffsets[0];
  n = colOffsets[1] - colOffsets[0];
  if (m < 0)
  {
     m = 0;
  }
  if (n < 0)
  {
     n = 0;
  }
  // determine global number of columns
  HYPRE_Int N = 0;
  MPI_Allreduce(&n, &N, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  auto temp = std::unique_ptr<mfem::SparseMatrix>(new mfem::SparseMatrix(m, N, nnz));
  
  // generate HypreParMatrix from SparseMatrix
  A = GenerateHypreParMatrixFromSparseMatrix(rowOffsets, colOffsets, temp.get());
  return A;
}
