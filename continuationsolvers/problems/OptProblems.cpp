#include "mfem.hpp"
#include "OptProblems.hpp"



GeneralOptProblem::GeneralOptProblem() : block_offsetsx(3) { label = -1; }

void GeneralOptProblem::Init(HYPRE_BigInt * dofOffsetsU_, HYPRE_BigInt * dofOffsetsM_)
{
  dofOffsetsU = new HYPRE_BigInt[2];
  dofOffsetsM = new HYPRE_BigInt[2];
  for(int i = 0; i < 2; i++)
  {
    dofOffsetsU[i] = dofOffsetsU_[i];
    dofOffsetsM[i] = dofOffsetsM_[i];
  }
  dimU = dofOffsetsU[1] - dofOffsetsU[0];
  dimM = dofOffsetsM[1] - dofOffsetsM[0];
  dimC = dimM;
  
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();

  MPI_Allreduce(&dimU, &dimUglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dimM, &dimMglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

double GeneralOptProblem::CalcObjective(const mfem::BlockVector &x)
{
  int eval_err; // throw away
  return CalcObjective(x, eval_err);
}

void GeneralOptProblem::CalcObjectiveGrad(const mfem::BlockVector &x, mfem::BlockVector &y)
{
   Duf(x, y.GetBlock(0));
   Dmf(x, y.GetBlock(1));
}

void GeneralOptProblem::c(const mfem::BlockVector &x, mfem::Vector &y)
{
  int eval_err; // throw-away
  return c(x, y, eval_err);
}

GeneralOptProblem::~GeneralOptProblem()
{
   block_offsetsx.DeleteAll();
}


// min E(d) s.t. g(d) >= 0
// min_(d,s) E(d) s.t. c(d,s) := g(d) - s = 0, s >= 0
OptProblem::OptProblem() : GeneralOptProblem()
{
}

void OptProblem::Init(HYPRE_BigInt * dofOffsetsU_, HYPRE_BigInt * dofOffsetsM_)
{
  dofOffsetsU = new HYPRE_BigInt[2];
  dofOffsetsM = new HYPRE_BigInt[2];
  for(int i = 0; i < 2; i++)
  {
    dofOffsetsU[i] = dofOffsetsU_[i];
    dofOffsetsM[i] = dofOffsetsM_[i];
  }

  dimU = dofOffsetsU[1] - dofOffsetsU[0];
  dimM = dofOffsetsM[1] - dofOffsetsM[0];
  dimC = dimM;
  
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();

  MPI_Allreduce(&dimU, &dimUglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dimM, &dimMglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  ml.SetSize(dimM); ml = 0.0;
  mfem::Vector negIdentDiag(dimM);
  negIdentDiag = -1.0;
  Ih = GenerateHypreParMatrixFromDiagonal(dofOffsetsM, negIdentDiag);
}


double OptProblem::CalcObjective(const mfem::BlockVector &x, int & eval_err)
{ 
   return E(x.GetBlock(0), eval_err); 
}


void OptProblem::Duf(const mfem::BlockVector &x, mfem::Vector &y) { DdE(x.GetBlock(0), y); }

void OptProblem::Dmf(const mfem::BlockVector & /*x*/, mfem::Vector &y) { y = 0.0; }

mfem::Operator * OptProblem::Duuf(const mfem::BlockVector &x) 
{ 
   return DddE(x.GetBlock(0)); 
}

mfem::Operator * OptProblem::Dumf(const mfem::BlockVector &/*x*/) { return nullptr; }

mfem::Operator * OptProblem::Dmuf(const mfem::BlockVector &/*x*/) { return nullptr; }

mfem::Operator * OptProblem::Dmmf(const mfem::BlockVector &/*x*/) { return nullptr; }

void OptProblem::c(const mfem::BlockVector &x, mfem::Vector &y, int & eval_err) // c(u,m) = g(u) - m 
{
   g(x.GetBlock(0), y, eval_err);
   y.Add(-1.0, x.GetBlock(1));  
}


mfem::Operator * OptProblem::Duc(const mfem::BlockVector &x) 
{ 
   return Ddg(x.GetBlock(0)); 
}

mfem::Operator * OptProblem::Dmc(const mfem::BlockVector &/*x*/) 
{ 
   return Ih;
} 

OptProblem::~OptProblem() 
{
  delete[] dofOffsetsU;
  delete[] dofOffsetsM;
  delete Ih;
}






ReducedOptProblem::ReducedOptProblem(OptProblem * problem_, HYPRE_Int * constraintMask)
{
  problem = problem_;
  J = nullptr;
  P = nullptr;
  
  HYPRE_BigInt * dofOffsets = problem->GetDofOffsetsU();

  // given a constraint mask, lets update the constraintOffsets
  // from the original problem
  int nLocConstraints = 0;
  int nProblemConstraints = problem->GetDimM();
  for (int i = 0; i < nProblemConstraints; i++)
  {
    if (constraintMask[i] == 1)
    {
      nLocConstraints += 1;
    }
  }

  HYPRE_BigInt * constraintOffsets_reduced;
  constraintOffsets_reduced = offsetsFromLocalSizes(nLocConstraints);


  HYPRE_BigInt * constraintOffsets;
  constraintOffsets = offsetsFromLocalSizes(nProblemConstraints);
  
  P = GenerateProjector(constraintOffsets_reduced, constraintOffsets, constraintMask);

  Init(dofOffsets, constraintOffsets_reduced);
  delete[] constraintOffsets_reduced;
  delete[] constraintOffsets;
}

ReducedOptProblem::ReducedOptProblem(OptProblem * problem_, mfem::HypreParVector & constraintMask)
{
  problem = problem_;
  J = nullptr;
  P = nullptr;
  
  HYPRE_BigInt * dofOffsets = problem->GetDofOffsetsU();

  // given a constraint mask, lets update the constraintOffsets
  // from the original problem
  int nLocConstraints = 0;
  int nProblemConstraints = problem->GetDimM();
  for (int i = 0; i < nProblemConstraints; i++)
  {
    if (constraintMask[i] == 1)
    {
      nLocConstraints += 1;
    }
  }

  HYPRE_BigInt * constraintOffsets_reduced;
  constraintOffsets_reduced = offsetsFromLocalSizes(nLocConstraints);



  HYPRE_BigInt * constraintOffsets;
  constraintOffsets = offsetsFromLocalSizes(nProblemConstraints);
  
  P = GenerateProjector(constraintOffsets_reduced, constraintOffsets, constraintMask);

  Init(dofOffsets, constraintOffsets_reduced);
  delete[] constraintOffsets_reduced;
  delete[] constraintOffsets;
}

// energy objective E(d)
double ReducedOptProblem::E(const mfem::Vector &d, int & eval_err)
{
  return problem->E(d, eval_err);
}


// gradient of energy objective
void ReducedOptProblem::DdE(const mfem::Vector &d, mfem::Vector & gradE)
{
  problem->DdE(d, gradE);
}


mfem::Operator * ReducedOptProblem::DddE(const mfem::Vector &d)
{
  return problem->DddE(d);
}

void ReducedOptProblem::g(const mfem::Vector &d, mfem::Vector &gd, int & eval_err)
{
  mfem::Vector gdfull(problem->GetDimM()); gdfull = 0.0;
  problem->g(d, gdfull, eval_err);
  P->Mult(gdfull, gd);
}


mfem::Operator * ReducedOptProblem::Ddg(const mfem::Vector &d)
{
  mfem::Operator * Jfull = problem->Ddg(d);
  auto Jfull_hypre = dynamic_cast<mfem::HypreParMatrix *>(Jfull);
  MFEM_VERIFY(Jfull_hypre, "expecting Ddg to be a HypreParMatrix"); 
  if (J)
  {
    delete J; J = nullptr;
  }
  J = ParMult(P, Jfull_hypre, true);
  return J;
}

ReducedOptProblem::~ReducedOptProblem()
{
  delete P;
  if (J)
  {
    delete J;
  }
}


