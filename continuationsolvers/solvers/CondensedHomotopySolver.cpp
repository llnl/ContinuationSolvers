
#include "CondensedHomotopySolver.hpp"
#include "../utilities.hpp"


void CondensedHomotopySolver::SetOperator(const mfem::Operator& op)
{
   auto blkOp = dynamic_cast<const mfem::BlockOperator *>(&op);
   MFEM_VERIFY(blkOp, "op must be a mfem::BlockOperator!");
   blkOp->RowOffsets().Copy(blockOffsets);

   // Extract blocks
   auto A00 = dynamic_cast<const mfem::HypreParMatrix *>(&blkOp->GetBlock(0, 0));
   auto A01 = dynamic_cast<const mfem::HypreParMatrix *>(&blkOp->GetBlock(0, 1));
   auto A10 = dynamic_cast<const mfem::HypreParMatrix *>(&blkOp->GetBlock(1, 0));
   auto A11 = dynamic_cast<const mfem::HypreParMatrix *>(&blkOp->GetBlock(1, 1));
   A12 = dynamic_cast<const mfem::HypreParMatrix *>(&blkOp->GetBlock(1, 2));
   A20 = dynamic_cast<const mfem::HypreParMatrix *>(&blkOp->GetBlock(2, 0));
   auto A22 = dynamic_cast<const mfem::HypreParMatrix *>(&blkOp->GetBlock(2, 2));

   if (use_amgf)
   {
      if (P)
      {
         delete P;
      }
      
      P = NonZeroColMap(*A12);
   }
   // A00, A01, A10, A11 are diagonal matrices
   // TODO: add a check?
   mfem::Vector A00_d, A01_d, A10_d, A11_d;
   A00->GetDiag(A00_d);
   A01->GetDiag(A01_d);
   A10->GetDiag(A10_d);
   A11->GetDiag(A11_d);

   // scaleij is the diagonal of the (i,j) block of 
   //                [A00 A01]^-1
   //                [A10 A11]
   // scale11 = (A11 - A10 * A00^{-1} * A01)^{-1}
   scale11 = A10_d;
   scale11 /= A00_d;
   scale11 *= A01_d;
   scale11.Neg();
   scale11 += A11_d;
   scale11.Reciprocal();

   // scale01 = -A00^{-1} * A01 * scale11
   // scale10 = -scale11 * A10 * A00^{-1}
   scale01 = scale11;
   scale01 /= A00_d;
   scale01.Neg();
   scale10 = scale01;
   scale01 *= A01_d;
   scale10 *= A10_d;

   // scale00 = A00^{-1} + A00^{-1} * A01 * scale11 * A10 * A00^{-1}
   scale00 = scale01;
   scale00.Neg();
   scale00 *= A10_d;
   scale00 += 1.0;
   scale00 /= A00_d;

   mfem::HypreParMatrix scaledA12(*A12);
   scale01.Neg();
   scaledA12.ScaleRows(scale01);
   scale01.Neg();
   mfem::HypreParMatrix * scaledProduct = mfem::ParMult(A20, &scaledA12);
   if (Areduced)
   {
      delete Areduced;
   }
   // FIXME: this requires A22 and scaledProduct to have the same offd_colmap?
   Areduced = ParAdd(A22, scaledProduct);
   if (use_amgf)
   {
      if (amgf)
      {
         delete amgf;
      }
      amgf = new AMGF(*Areduced, *P);
      auto iterative_solver = dynamic_cast<mfem::IterativeSolver *>(AreducedSolver);
      if (iterative_solver)
      {
         iterative_solver->SetPreconditioner(*amgf);
      }
   }
   if (AreducedSolver)
   {
      AreducedSolver->SetOperator(*Areduced);
   }
   delete scaledProduct;
}

void CondensedHomotopySolver::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
   mfem::BlockVector blk_x(x.GetData(), blockOffsets);
   mfem::BlockVector blk_b(b.GetData(), blockOffsets);
   Mult(blk_b, blk_x);
}

void CondensedHomotopySolver::Mult(const mfem::BlockVector& b, mfem::BlockVector& x) const
{
   mfem::Vector scaled_b0(b.GetBlock(0));
   mfem::Vector scaled_b1(b.GetBlock(1));
   mfem::Vector b_reduced(b.GetBlock(2));

   // form RHS for reduced system
   scaled_b0 *= scale00;
   scaled_b1 *= scale01;
   scaled_b0 += scaled_b1;
   A20->Mult(-1.0, scaled_b0, 1.0, b_reduced);

   if (AreducedSolver)
   {
       AreducedSolver->Mult(b_reduced, x.GetBlock(2));
      	   
       mfem::Vector residual(x.GetBlock(2).Size());
       Areduced->Mult(x.GetBlock(2), residual);
       residual.Add(-1.0, b_reduced);
       double err = 0.0;
       err = mfem::GlobalLpNorm(2, residual.Norml2(), MPI_COMM_WORLD);
       std::cout << "||Areduced xreduced - breduced||_2 = " << err << std::endl;
   }
   else
   {
      DirectSolver defaultSolver(*Areduced);
      defaultSolver.Mult(b_reduced, x.GetBlock(2));
   }
   


   // recover the solution to the original system
   mfem::Vector helper0(b.GetBlock(0));
   mfem::Vector helper1(b.GetBlock(1));
   A12->Mult(-1.0, x.GetBlock(2), 1.0, helper1);

   x.GetBlock(0) = helper0;
   mfem::Vector scaledHelper1(helper1);
   x.GetBlock(0) *= scale00;
   scaledHelper1 *= scale01;
   x.GetBlock(0) += scaledHelper1;

   x.GetBlock(1) = helper0;
   scaledHelper1 = helper1;
   x.GetBlock(1) *= scale10;
   scaledHelper1 *= scale11;
   x.GetBlock(1) += scaledHelper1;
}

CondensedHomotopySolver::~CondensedHomotopySolver()
{
   if (Areduced)
   {
      delete Areduced;
   }
   if (P)
   {
      delete P;
   }
   if (amgf)
   {
      delete amgf;
   }
}
