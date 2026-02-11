#include "mfem.hpp"
#ifdef MFEM_USE_STRUMPACK
#include <StrumpackOptions.hpp>
#include <mfem/linalg/strumpack.hpp>
#endif

#ifndef UTILITY_FUNCTIONS
#define UTILITY_FUNCTIONS

void HypreToMfemOffsets(HYPRE_BigInt * offsets);

mfem::HypreParMatrix * GenerateHypreParMatrixFromSparseMatrix(HYPRE_BigInt * rowOffsetsloc, HYPRE_BigInt * colOffsetsloc, mfem::SparseMatrix * Asparse);

mfem::HypreParMatrix * GenerateHypreParMatrixFromDiagonal(HYPRE_BigInt * offsetsloc, 
		mfem::Vector & diag);


mfem::HypreParMatrix * GenerateProjector(HYPRE_BigInt * reduced_offsets, HYPRE_BigInt * offsets, HYPRE_Int * mask);

mfem::HypreParMatrix * GenerateProjector(HYPRE_BigInt * reduced_offsets, HYPRE_BigInt * offsets, const mfem::HypreParVector & mask);


HYPRE_BigInt * offsetsFromLocalSizes(int n, MPI_Comm comm = MPI_COMM_WORLD);

mfem::HypreParMatrix * NonZeroRowMap(const mfem::HypreParMatrix& A);

mfem::HypreParMatrix * NonZeroColMap(const mfem::HypreParMatrix& A);


class DirectSolver : public mfem::Solver
{
private:
   mfem::Solver* solver;
#ifdef MFEM_USE_STRUMPACK
    mfem::STRUMPACKRowLocMatrix* Astrumpack = nullptr;
#endif

public:
   DirectSolver();
   DirectSolver(const mfem::Operator& op);
   virtual ~DirectSolver();

   void SetOperator(const mfem::Operator& op) override;
   void Mult(const mfem::Vector &b, mfem::Vector &x) const override;  
};

#endif
