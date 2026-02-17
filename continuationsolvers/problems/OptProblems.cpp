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



// Obstacle Problem, no essential boundary conditions enforced
// Hessian of energy term is K + M (stiffness + mass)
ObstacleProblem::ObstacleProblem(mfem::ParFiniteElementSpace *fesU_, 
                                       mfem::ParFiniteElementSpace *fesM_, 
                                       double (*fSource)(const mfem::Vector &)) : 
                                       OptProblem()
{
   
   Init(fesU_->GetTrueDofOffsets(), fesM_->GetTrueDofOffsets());
   

   Kform = new mfem::ParBilinearForm(fesU_);
   Kform->AddDomainIntegrator(new mfem::MassIntegrator);
   Kform->AddDomainIntegrator(new mfem::DiffusionIntegrator);
   Kform->Assemble();
   Kform->Finalize();
   Kform->FormSystemMatrix(ess_tdof_list, K);
   mfem::FunctionCoefficient fcoeff(fSource);
   fform = new mfem::ParLinearForm(fesU_);
   fform->AddDomainIntegrator(new mfem::DomainLFIntegrator(fcoeff));
   fform->Assemble();
   mfem::Vector F(dimU);
   fform->ParallelAssemble(F);
   f.SetSize(dimU);
   f.Set(1.0, F);
   psi.SetSize(dimU);
   psi = 0.0;
   
   mfem::Vector iDiag(dimU); iDiag = 1.0;
   J = GenerateHypreParMatrixFromDiagonal(dofOffsetsU, iDiag);
}

// Obstacle Problem, essential boundary conditions enforced
// Hessian of energy term is K (stiffness)
//ObstacleProblem::ObstacleProblem(ParFiniteElementSpace *fesU_, 
//                                       ParFiniteElementSpace *fesM_, 
//				       double (*fSource)(const Vector &),
//				       double (*obstacleSource)(const Vector &),
//				       Array<int> tdof_list, Vector &xDC) : OptProblem()
//{
//   Init(fesU_->GetTrueDofOffsets(), fesM_->GetTrueDofOffsets());
//   // elastic energy functional terms	
//   ess_tdof_list = tdof_list;
//   Kform = new ParBilinearForm(fesU);
//   Kform->AddDomainIntegrator(new DiffusionIntegrator);
//   Kform->Assemble();
//   Kform->Finalize();
//   Kform->FormSystemMatrix(ess_tdof_list, K);
//
//   FunctionCoefficient fcoeff(fSource);
//   fform = new ParLinearForm(fesU);
//   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
//   fform->Assemble();
//   Vector F(dimU);
//   fform->ParallelAssemble(F);
//   f.SetSize(dimU);
//   f.Set(1.0, F);
//   Kform->EliminateVDofsInRHS(ess_tdof_list, xDC, f);
//   
//   // obstacle constraints --  
//   Vector iDiag(dimU); iDiag = 1.0;
//   for(int i = 0; i < ess_tdof_list.Size(); i++)
//   {
//     iDiag(ess_tdof_list[i]) = 0.0;
//   }
//   SparseMatrix * Jacg = new SparseMatrix(iDiag);
//
//   J = new HypreParMatrix(fesU->GetComm(), dofOffsetsU, dofOffsetsU, Jacg);
//   HypreStealOwnership(*J, *Jacg);
//   delete Jacg;
//
//   FunctionCoefficient psi_fc(obstacleSource);
//   ParGridFunction psi_gf(fesU);
//   psi_gf.ProjectCoefficient(psi_fc);
//   psi.SetSize(dimU);
//   psi.Set(1.0, (*psi_gf.GetTrueDofs()));
//   for(int i = 0; i < ess_tdof_list.Size(); i++)
//   {
//     psi(ess_tdof_list[i]) -= 1.e-8;
//   }
//}



double ObstacleProblem::E(const mfem::Vector &d, int & eval_err)
{
   mfem::Vector Kd(K.Height()); Kd = 0.0;
   eval_err = 0;
   MFEM_VERIFY(d.Size() == K.Width(), "ParObstacleProblem::E - Inconsistent dimensions");
   K.Mult(d, Kd);
   return 0.5 * mfem::InnerProduct(MPI_COMM_WORLD, d, Kd) - mfem::InnerProduct(MPI_COMM_WORLD, f, d);
}

void ObstacleProblem::DdE(const mfem::Vector &d, mfem::Vector &gradE)
{
   gradE.SetSize(K.Height());
   MFEM_VERIFY(d.Size() == K.Width(), "ParObstacleProblem::DdE - Inconsistent dimensions");
   K.Mult(d, gradE);
   MFEM_VERIFY(f.Size() == K.Height(), "ParObstacleProblem::DdE - Inconsistent dimensions");
   gradE.Add(-1.0, f);
}

mfem::Operator * ObstacleProblem::DddE(const mfem::Vector &d)
{
   return &K; 
}

// g(d) = d >= \psi
void ObstacleProblem::g(const mfem::Vector &d, mfem::Vector &gd, int & eval_err)
{
   eval_err = 0;
   MFEM_VERIFY(d.Size() == J->Width(), "ParObstacleProblem::g - Inconsistent dimensions");
   J->Mult(d, gd);
   MFEM_VERIFY(gd.Size() == J->Height(), "ParObstacleProblem::g - Inconsistent dimensions");
   gd.Add(-1.0, psi);
}

mfem::Operator * ObstacleProblem::Ddg(const mfem::Vector &d)
{
   return J;
}

ObstacleProblem::~ObstacleProblem()
{
   delete Kform;
   delete fform;
   delete J;
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


