#include "OptProblems.hpp"
#include "mfem.hpp"

GeneralOptProblem::GeneralOptProblem() : block_offsetsx(3) { label = -1; }

void GeneralOptProblem::Init(HYPRE_BigInt *dofOffsetsU_,
                             HYPRE_BigInt *dofOffsetsM_) {
  dofOffsetsU = new HYPRE_BigInt[2];
  dofOffsetsM = new HYPRE_BigInt[2];
  for (int i = 0; i < 2; i++) {
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

double GeneralOptProblem::CalcObjective(const mfem::BlockVector &x) {
  int eval_err; // throw away
  return CalcObjective(x, eval_err);
}

void GeneralOptProblem::CalcObjectiveGrad(const mfem::BlockVector &x,
                                          mfem::BlockVector &y) {
  Duf(x, y.GetBlock(0));
  Dmf(x, y.GetBlock(1));
}

void GeneralOptProblem::c(const mfem::BlockVector &x, mfem::Vector &y) {
  int eval_err; // throw-away
  return c(x, y, eval_err);
}

GeneralOptProblem::~GeneralOptProblem() { block_offsetsx.DeleteAll(); }

// min E(d) s.t. g(d) >= 0
// min_(d,s) E(d) s.t. c(d,s) := g(d) - s = 0, s >= 0
OptProblem::OptProblem() : GeneralOptProblem() {}

void OptProblem::Init(HYPRE_BigInt *dofOffsetsU_, HYPRE_BigInt *dofOffsetsM_) {
  dofOffsetsU = new HYPRE_BigInt[2];
  dofOffsetsM = new HYPRE_BigInt[2];
  for (int i = 0; i < 2; i++) {
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

  ml.SetSize(dimM);
  ml = 0.0;
  mfem::Vector negIdentDiag(dimM);
  negIdentDiag = -1.0;
  Ih = GenerateHypreParMatrixFromDiagonal(dofOffsetsM, negIdentDiag);
}

double OptProblem::CalcObjective(const mfem::BlockVector &x, int &eval_err) {
  return E(x.GetBlock(0), eval_err);
}

void OptProblem::Duf(const mfem::BlockVector &x, mfem::Vector &y) {
  DdE(x.GetBlock(0), y);
}

void OptProblem::Dmf(const mfem::BlockVector & /*x*/, mfem::Vector &y) {
  y = 0.0;
}

mfem::Operator *OptProblem::Duuf(const mfem::BlockVector &x) {
  return DddE(x.GetBlock(0));
}

mfem::Operator *OptProblem::Dumf(const mfem::BlockVector & /*x*/) {
  return nullptr;
}

mfem::Operator *OptProblem::Dmuf(const mfem::BlockVector & /*x*/) {
  return nullptr;
}

mfem::Operator *OptProblem::Dmmf(const mfem::BlockVector & /*x*/) {
  return nullptr;
}

void OptProblem::c(const mfem::BlockVector &x, mfem::Vector &y,
                   int &eval_err) // c(u,m) = g(u) - m
{
  g(x.GetBlock(0), y, eval_err);
  y.Add(-1.0, x.GetBlock(1));
}

mfem::Operator *OptProblem::Duc(const mfem::BlockVector &x) {
  return Ddg(x.GetBlock(0));
}

mfem::Operator *OptProblem::Dmc(const mfem::BlockVector & /*x*/) { return Ih; }

mfem::Operator *OptProblem::Duucl(const mfem::BlockVector &x,
                                  const mfem::Vector &l) {
  return Dddgl(x.GetBlock(0), l);
}

mfem::Operator *OptProblem::Dumcl(const mfem::BlockVector & /*x*/,
                                  const mfem::Vector & /*l*/) {
  /* TODO: return empty HypreParMatrices of the appropriate sizes? */
  return nullptr;
}

mfem::Operator *OptProblem::Dmucl(const mfem::BlockVector & /*x*/,
                                  const mfem::Vector & /*l*/) {
  return nullptr;
}

mfem::Operator *OptProblem::Dmmcl(const mfem::BlockVector & /*x*/,
                                  const mfem::Vector & /*l*/) {
  return nullptr;
}

mfem::Operator *OptProblem::Dddgl(const mfem::Vector & /*d*/,
                                  const mfem::Vector & /*l*/) {
  MFEM_VERIFY(false, "child class must provide implementation of Dddgl method");
  return nullptr;
}

OptProblem::~OptProblem() {
  delete[] dofOffsetsU;
  delete[] dofOffsetsM;
  delete Ih;
}

ReducedOptProblem::ReducedOptProblem(OptProblem *problem_,
                                     HYPRE_Int *constraintMask) {
  problem = problem_;
  HYPRE_BigInt *dofOffsets = problem->GetDofOffsetsU();

  // given a constraint mask, lets update the constraintOffsets
  // from the original problem
  int nLocConstraints = 0;
  int nProblemConstraints = problem->GetDimM();
  for (int i = 0; i < nProblemConstraints; i++) {
    if (constraintMask[i] == 1) {
      nLocConstraints += 1;
    }
  }

  HYPRE_BigInt *constraintOffsets_reduced;
  constraintOffsets_reduced = offsetsFromLocalSizes(nLocConstraints);

  HYPRE_BigInt *constraintOffsets;
  constraintOffsets = offsetsFromLocalSizes(nProblemConstraints);

  auto Rc_mat = GenerateProjector(constraintOffsets_reduced, constraintOffsets,
                        constraintMask);
  Rc.reset(Rc_mat);
  Pc.reset(Rc->Transpose());
  
  // now set dof restriction
  // identity mapping
  int nDofs = problem->GetDimU();
  HYPRE_Int dofMask[nDofs];
  for (int i = 0; i < nDofs; i++)
  {
     dofMask[i] = 1;
  }
  auto Rdof_mat = GenerateProjector(dofOffsets, dofOffsets, dofMask);
  Rdof.reset(Rdof_mat);
  Pdof.reset(Rdof->Transpose());
  
  
  Init(dofOffsets, constraintOffsets_reduced);
  delete[] constraintOffsets_reduced;
  delete[] constraintOffsets;
  dDC.SetSize(nDofs); dDC = 0.1;
  gradEfull.SetSize(nDofs); gradEfull = 0.0;
  dFull.SetSize(nDofs); dFull = 0.0;
  gfull.SetSize(problem->GetDimM()); gfull = 0.0;
}


ReducedOptProblem::ReducedOptProblem(OptProblem *problem_,
		                     HYPRE_Int *dofMask,
                                     HYPRE_Int *constraintMask) {
  problem = problem_;
  //HYPRE_BigInt *dofOffsets = problem->GetDofOffsetsU();

  // given a constraint mask, lets update the constraintOffsets
  // from the original problem
  int nLocConstraints = 0;
  int nProblemConstraints = problem->GetDimM();
  for (int i = 0; i < nProblemConstraints; i++) {
    if (constraintMask[i] == 1) {
      nLocConstraints += 1;
    }
  }



  HYPRE_BigInt *constraintOffsets_reduced;
  constraintOffsets_reduced = offsetsFromLocalSizes(nLocConstraints);

  HYPRE_BigInt *constraintOffsets;
  constraintOffsets = offsetsFromLocalSizes(nProblemConstraints);

  auto Rc_mat = GenerateProjector(constraintOffsets_reduced, constraintOffsets,
                        constraintMask);
  Rc.reset(Rc_mat);
  Pc.reset(Rc->Transpose());
  
  // now set dof restriction
  int nLocDofs = 0;
  int nProblemDofs = problem->GetDimU();
  for (int i = 0; i < nProblemDofs; i++) {
    if (dofMask[i] == 1) {
      nLocDofs += 1;
    }
  }
  HYPRE_BigInt *dofOffsets_reduced;
  dofOffsets_reduced = offsetsFromLocalSizes(nLocDofs);
  HYPRE_BigInt *dofOffsets;
  dofOffsets = offsetsFromLocalSizes(nProblemDofs);
  auto Rdof_mat = GenerateProjector(dofOffsets_reduced, dofOffsets, dofMask);
  Rdof.reset(Rdof_mat);
  Pdof.reset(Rdof->Transpose());
  
  
  Init(dofOffsets_reduced, constraintOffsets_reduced);
  delete[] constraintOffsets_reduced;
  delete[] constraintOffsets;
  delete[] dofOffsets_reduced;
  delete[] dofOffsets;
  dDC.SetSize(nProblemDofs); dDC = 0.0;
  gradEfull.SetSize(nProblemDofs); gradEfull = 0.0;
  dFull.SetSize(nProblemDofs); dFull = 0.0;
  gfull.SetSize(nProblemConstraints); gfull = 0.0;
}

ReducedOptProblem::ReducedOptProblem(OptProblem *problem_,
                                     mfem::HypreParVector &constraintMask) {
  
  
  HYPRE_Int intConstraintMask[problem->GetDimM()];
  int nProblemConstraints = problem->GetDimM();
  for (int i = 0; i < nProblemConstraints; i++) {
    intConstraintMask[i] = 0;
    if (constraintMask[i] == 1)
    {
      intConstraintMask[i] = 1;
    }
  }
  ReducedOptProblem(problem_, intConstraintMask);  
}

// energy objective E(d)
double ReducedOptProblem::E(const mfem::Vector &d, int &eval_err) {
  Pdof->Mult(d, dFull);
  dFull.Add(1.0, dDC);
  return problem->E(dFull, eval_err);
}

// gradient of energy objective
void ReducedOptProblem::DdE(const mfem::Vector &d, mfem::Vector &gradE) {
  Pdof->Mult(d, dFull);
  dFull.Add(1.0, dDC);
  problem->DdE(dFull, gradEfull);
  MFEM_VERIFY(Rdof->Height() == gradE.Size(), "size issue");
  MFEM_VERIFY(Rdof->Width() == gradEfull.Size(), "size issue");
  Rdof->Mult(gradEfull, gradE);
}

mfem::Operator *ReducedOptProblem::DddE(const mfem::Vector &d) {
  Pdof->Mult(d, dFull);
  dFull.Add(1.0, dDC);
  mfem::HypreParMatrix * Hfull = dynamic_cast<mfem::HypreParMatrix *>(problem->DddE(dFull));
  MFEM_VERIFY(Hfull, "cast failure");
  H.reset(RAP(Hfull, Pdof.get())); 
  MFEM_VERIFY(H->Height() == dimU, "size issue");
  return H.get();
}

void ReducedOptProblem::g(const mfem::Vector &d, mfem::Vector &gd,
                          int &eval_err) {
  Pdof->Mult(d, dFull);
  dFull.Add(1.0, dDC);
  problem->g(dFull, gfull, eval_err);
  Rc->Mult(gfull, gd);
}

mfem::Operator *ReducedOptProblem::Ddg(const mfem::Vector &d) {
  Pdof->Mult(d, dFull);
  dFull.Add(1.0, dDC);
  mfem::HypreParMatrix *Jfull = dynamic_cast<mfem::HypreParMatrix *>(problem->Ddg(dFull));
  MFEM_VERIFY(Jfull, "cast failure");

  J.reset(RAP(Pc.get(), Jfull, Pdof.get()));
  return J.get();
}

void ReducedOptProblem::ProlongateToFullDofs(const mfem::Vector &d, mfem::Vector &df)
{
  Pdof->Mult(d, df);
}	


ReducedOptProblem::~ReducedOptProblem() {
}

// min E(d) s.t. g(d) = 0
OptEqProblem::OptEqProblem() : GeneralOptProblem() {}

void OptEqProblem::Init(HYPRE_BigInt *dofOffsetsU_,
                        HYPRE_BigInt *dofOffsetsC_) {
  dofOffsetsU = new HYPRE_BigInt[2];
  dofOffsetsM = new HYPRE_BigInt[2];
  dofOffsetsC = new HYPRE_BigInt[2];

  for (int i = 0; i < 2; i++) {
    dofOffsetsU[i] = dofOffsetsU_[i];
    dofOffsetsC[i] = dofOffsetsC_[i];
    dofOffsetsM[i] = 0;
  }

  dimU = dofOffsetsU[1] - dofOffsetsU[0];
  dimM = dofOffsetsM[1] - dofOffsetsM[0];
  dimC = dofOffsetsC[1] - dofOffsetsC[0];

  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();

  MPI_Allreduce(&dimU, &dimUglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  dimMglb = 0;

  ml.SetSize(dimM);
  ml = 0.0;

  dcdm.reset(GenerateNullHypreParMatrix(dofOffsetsC, dofOffsetsM));
  Hmmf.reset(GenerateNullHypreParMatrix(dofOffsetsM, dofOffsetsM));
}

double OptEqProblem::CalcObjective(const mfem::BlockVector &x, int &eval_err) {
  return E(x.GetBlock(0), eval_err);
}

void OptEqProblem::Duf(const mfem::BlockVector &x, mfem::Vector &y) {
  DdE(x.GetBlock(0), y);
}

void OptEqProblem::Dmf(const mfem::BlockVector & /*x*/, mfem::Vector &y) {
  y = 0.0;
}

mfem::Operator *OptEqProblem::Duuf(const mfem::BlockVector &x) {
  return DddE(x.GetBlock(0));
}

mfem::Operator *OptEqProblem::Dumf(const mfem::BlockVector & /*x*/) {
  return nullptr;
}

mfem::Operator *OptEqProblem::Dmuf(const mfem::BlockVector & /*x*/) {
  return nullptr;
}

mfem::Operator *OptEqProblem::Dmmf(const mfem::BlockVector & /*x*/) {
  return Hmmf.get();
}

// c(u, m) = g(u)
void OptEqProblem::c(const mfem::BlockVector &x, mfem::Vector &y,
                     int &eval_err) {
  g(x.GetBlock(0), y, eval_err);
}

mfem::Operator *OptEqProblem::Duc(const mfem::BlockVector &x) {
  return Ddg(x.GetBlock(0));
}

mfem::Operator *OptEqProblem::Dmc(const mfem::BlockVector & /*x*/) {
  return dcdm.get();
}

mfem::Operator *OptEqProblem::Duucl(const mfem::BlockVector &x,
                                    const mfem::Vector &l) {
  return Dddgl(x.GetBlock(0), l);
}

mfem::Operator *OptEqProblem::Dumcl(const mfem::BlockVector & /*x*/,
                                    const mfem::Vector & /*l*/) {
  return nullptr;
}

mfem::Operator *OptEqProblem::Dmucl(const mfem::BlockVector & /*x*/,
                                    const mfem::Vector & /*l*/) {
  return nullptr;
}

mfem::Operator *OptEqProblem::Dmmcl(const mfem::BlockVector & /*x*/,
                                    const mfem::Vector & /*l*/) {
  return nullptr;
}

mfem::Operator *OptEqProblem::Dddgl(const mfem::Vector & /*d*/,
                                    const mfem::Vector & /*l*/) {
  MFEM_VERIFY(false, "child class must provide implementation of Dddgl method");
  return nullptr;
}

OptEqProblem::~OptEqProblem() {
  delete[] dofOffsetsU;
  delete[] dofOffsetsM;
  delete[] dofOffsetsC;
}

ParamOptProblem::ParamOptProblem() : OptProblem() {}

void ParamOptProblem::InitTheta(const mfem::Vector &theta) {
  dimTheta = theta.Size();
  MPI_Allreduce(&dimTheta, &dimThetaglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  theta_default.SetSize(dimTheta);
  theta_default.Set(1.0, theta);
  if (!dofOffsetsTheta) {
    delete[] dofOffsetsTheta;
  }
  dofOffsetsTheta = offsetsFromLocalSizes(dimTheta);
  theta_initialized = true;
}

HYPRE_BigInt *ParamOptProblem::GetDofOffsetsTheta() const {
  MFEM_VERIFY(
      theta_initialized,
      "Attempting to call method when parameter hasn't been initialized");
  return dofOffsetsTheta;
}

double ParamOptProblem::E(const mfem::Vector &d, int &eval_err) {
  MFEM_VERIFY(
      theta_initialized,
      "Attempting to call method when parameter hasn't been initialized");
  return E(d, theta_default, eval_err);
}

void ParamOptProblem::DdE(const mfem::Vector &d, mfem::Vector &gradE) {
  MFEM_VERIFY(
      theta_initialized,
      "Attempting to call method when parameter hasn't been initialized");
  DdE(d, theta_default, gradE);
}

mfem::Operator *ParamOptProblem::DddE(const mfem::Vector &d) {
  return DddE(d, theta_default);
}

mfem::Operator *ParamOptProblem::DdddEl(const mfem::Vector & /*d*/,
                                        const mfem::Vector & /*l*/,
                                        const mfem::Vector & /*theta*/) {
  if (!HdddEl.get()) {
    HdddEl.reset(GenerateNullHypreParMatrix(dofOffsetsU, dofOffsetsU));
  }
  return HdddEl.get();
}

mfem::Operator *ParamOptProblem::DthddEl(const mfem::Vector & /*d*/,
                                         const mfem::Vector & /*l*/,
                                         const mfem::Vector & /*theta*/) {
  if (!HthddEl.get()) {
    HthddEl.reset(GenerateNullHypreParMatrix(dofOffsetsTheta, dofOffsetsU));
  }
  return HthddEl.get();
}

mfem::Operator *ParamOptProblem::DththdEl(const mfem::Vector & /*d*/,
                                          const mfem::Vector & /*l*/,
                                          const mfem::Vector & /*theta*/) {
  if (!HththdEl.get()) {
    HththdEl.reset(
        GenerateNullHypreParMatrix(dofOffsetsTheta, dofOffsetsTheta));
  }
  return HththdEl.get();
}

void ParamOptProblem::g(const mfem::Vector &d, mfem::Vector &gd,
                        int &eval_err) {
  g(d, theta_default, gd, eval_err);
}

mfem::Operator *ParamOptProblem::Ddg(const mfem::Vector &d) {
  return Ddg(d, theta_default);
}

mfem::Operator *ParamOptProblem::Dddgl(const mfem::Vector &d,
                                       const mfem::Vector &l) {
  return Dddgl(d, l, theta_default);
}

mfem::Operator *ParamOptProblem::Dddgl(const mfem::Vector & /*d*/,
                                       const mfem::Vector & /*l*/,
                                       const mfem::Vector & /*theta*/) {
  if (!Hddgl.get()) {
    Hddgl.reset(GenerateNullHypreParMatrix(dofOffsetsU, dofOffsetsU));
  }
  return Hddgl.get();
}

mfem::Operator *ParamOptProblem::Ddthgl(const mfem::Vector & /*d*/,
                                        const mfem::Vector & /*l*/,
                                        const mfem::Vector & /*theta*/) {
  if (!Hdthgl.get()) {
    Hdthgl.reset(GenerateNullHypreParMatrix(dofOffsetsU, dofOffsetsTheta));
  }
  return Hdthgl.get();
}

mfem::Operator *ParamOptProblem::Dththgl(const mfem::Vector & /*d*/,
                                         const mfem::Vector & /*l*/,
                                         const mfem::Vector & /*theta*/) {
  if (!Hththgl.get()) {
    Hththgl.reset(GenerateNullHypreParMatrix(dofOffsetsTheta, dofOffsetsTheta));
  }
  return Hththgl.get();
}

mfem::Operator *ParamOptProblem::Dddgl2(const mfem::Vector & /*d*/,
                                        const mfem::Vector & /*l*/,
                                        const mfem::Vector & /*theta*/) {
  if (!Hddgl2.get()) {
    Hddgl2.reset(GenerateNullHypreParMatrix(dofOffsetsM, dofOffsetsU));
  }
  return Hddgl2.get();
}

mfem::Operator *ParamOptProblem::Dthdgl2(const mfem::Vector & /*d*/,
                                         const mfem::Vector & /*l*/,
                                         const mfem::Vector & /*theta*/) {
  if (!Hthdgl2.get()) {
    Hthdgl2.reset(GenerateNullHypreParMatrix(dofOffsetsM, dofOffsetsTheta));
  }
  return Hthdgl2.get();
}

ParamOptProblem::~ParamOptProblem() {
  if (dofOffsetsTheta) {
    delete[] dofOffsetsTheta;
  }
}

ReducedParamOptProblem::ReducedParamOptProblem(ParamOptProblem * problem_, mfem::Array<int> tdof_list)
{
  problem = problem_;
  dimU = problem_->GetDimU() - tdof_list.Size();
  dimTheta = problem_->GetDimTheta() - tdof_list.Size(); 
  auto tmp_offsets = offsetsFromLocalSizes(dimU, MPI_COMM_WORLD);
  Init(tmp_offsets, tmp_offsets);
  delete[] tmp_offsets;
  HYPRE_Int * mask = new HYPRE_Int[problem_->GetDimU()];
  for (int i = 0; i < problem_->GetDimU(); i++)
  {
     mask[i] = 1;
  }
  for (int i = 0; i < tdof_list.Size(); i++)
  {
    mask[tdof_list[i]] = 0;
  }


  MFEM_VERIFY(problem_->GetDimU() == problem_->GetDimM(), "not setup for general case");
  MFEM_VERIFY(problem_->GetDimU() == problem_->GetDimTheta(), "not setup for general case");

  auto Rdof_mat = GenerateProjector(dofOffsetsU, problem->GetDofOffsetsU(), mask);
  Rdof.reset(Rdof_mat);
  auto Pdof_mat = Rdof_mat->Transpose();
  Pdof.reset(Pdof_mat);
  auto Rc_mat = GenerateProjector(dofOffsetsM, problem->GetDofOffsetsM(), mask);
  Rc.reset(Rc_mat);
  auto Pc_mat = Rc_mat->Transpose();
  Pc.reset(Pc_mat);

  mfem::Vector theta_init(dimU); theta_init = 0.0;
  Rc->Mult(problem->GetTheta(), theta_init);  
  InitTheta(theta_init);

  dDC.SetSize(problem->GetDimU()); dDC = 0.0;
  dFull.SetSize(problem->GetDimU()); dFull = 0.0;
  gradEFull.SetSize(problem->GetDimU()); gradEFull = 0.0;
  gFull.SetSize(problem->GetDimM()); gFull = 0.0;
  thetaFull.SetSize(problem->GetDimM()); thetaFull = 0.0;
}


double ReducedParamOptProblem::E(const mfem::Vector &d, const mfem::Vector &theta,
                   int &eval_err)
{
   Pdof->Mult(d, dFull);
   dFull.Add(1.0, dDC);
   Pc->Mult(theta, thetaFull);
   return problem->E(dFull, thetaFull, eval_err);
}


void ReducedParamOptProblem::DdE(const mfem::Vector &d, const mfem::Vector &theta,
                        mfem::Vector &gradE)
{
   Pdof->Mult(d, dFull);
   dFull.Add(1.0, dDC);
   Pc->Mult(theta, thetaFull);
   problem->DdE(dFull, thetaFull, gradEFull);
   Rdof->Mult(gradEFull, gradE);
}


mfem::Operator * ReducedParamOptProblem::DddE(const mfem::Vector &d, const mfem::Vector &theta)
{
   Pdof->Mult(d, dFull);
   dFull.Add(1.0, dDC);
   Pc->Mult(theta, thetaFull);
   mfem::HypreParMatrix * Hddfull = dynamic_cast<mfem::HypreParMatrix*>(problem->DddE(dFull, thetaFull));
   MFEM_VERIFY(Hddfull, "cast failure");
   HddE.reset(RAP(Hddfull, Pdof.get()));
   return HddE.get(); 
}

mfem::Operator * ReducedParamOptProblem::DdthE(const mfem::Vector &d, const mfem::Vector &theta)
{
   Pdof->Mult(d, dFull);
   dFull.Add(1.0, dDC);
   Pc->Mult(theta, thetaFull);
   mfem::HypreParMatrix * Hdthfull = dynamic_cast<mfem::HypreParMatrix*>(problem->DdthE(dFull, thetaFull));
   MFEM_VERIFY(Hdthfull, "cast failure");
   HdthE.reset(RAP(Hdthfull, Pdof.get()));
   return HdthE.get(); 
}


void ReducedParamOptProblem::g(const mfem::Vector &d, const mfem::Vector &theta,
                        mfem::Vector &gd, int &eval_err)
{
   Pdof->Mult(d, dFull);
   dFull.Add(1.0, dDC);
   Pc->Mult(theta, thetaFull);
   problem->g(dFull, thetaFull, gFull, eval_err);
   Rc->Mult(gFull, gd);
}

mfem::Operator * ReducedParamOptProblem::Ddg(const mfem::Vector &d, const mfem::Vector &theta)
{
   Pdof->Mult(d, dFull);
   dFull.Add(1.0, dDC);
   Pc->Mult(theta, thetaFull);
   mfem::HypreParMatrix *Jfull = dynamic_cast<mfem::HypreParMatrix *>(problem->Ddg(dFull, thetaFull));
   MFEM_VERIFY(Jfull, "cast failure");

   Jdg.reset(RAP(Pc.get(), Jfull, Pdof.get()));
   return Jdg.get();
}


mfem::Operator * ReducedParamOptProblem::Dthg(const mfem::Vector &d, const mfem::Vector &theta)
{
   Pdof->Mult(d, dFull);
   dFull.Add(1.0, dDC);
   Pc->Mult(theta, thetaFull);
   mfem::HypreParMatrix *Jfull = dynamic_cast<mfem::HypreParMatrix *>(problem->Dthg(dFull, thetaFull));
   MFEM_VERIFY(Jfull, "cast failure");

   Jthg.reset(RAP(Pc.get(), Jfull, Pdof.get()));
   return Jthg.get();
}

void ReducedParamOptProblem::ProlongateToFullDofs(const mfem::Vector &d, mfem::Vector &df)
{
  Pdof->Mult(d, df);
}	

ReducedParamOptProblem::~ReducedParamOptProblem() {
}




#if 0
// work in progress
// TODO: can we do something that is more compatible with 
//
ReducedOptEqProblem::ReducedOptEqProblem(OptEqProblem *problem_,
		                         HYPRE_Int *dofMask,
                                         HYPRE_Int *constraintMask) {
  problem = problem_;
  HYPRE_BigInt *dofOffsets = problem->GetDofOffsetsU();

  // given a constraint mask, lets update the constraintOffsets
  // from the original problem
  int nLocConstraints = 0;
  for (int i = 0; i < nProblemConstraints; i++) {
    if (constraintMask[i] == 1) {
      nLocConstraints += 1;
    }
  }
  HYPRE_BigInt *constraintOffsets_reduced;
  constraintOffsets_reduced = offsetsFromLocalSizes(nLocConstraints);

  int nProblemConstraints = problem->GetDimC();
  HYPRE_BigInt *constraintOffsets;
  constraintOffsets = offsetsFromLocalSizes(nProblemConstraints);

  auto Rc_mat = GenerateProjector(constraintOffsets_reduced, constraintOffsets,
		                  constraintMask); 
  Rc.reset(Rc_mat);





  Init(dofOffsets, constraintOffsets_reduced);
  delete[] constraintOffsets_reduced;
  delete[] constraintOffsets;
}

// energy objective E(d)
double ReducedOptProblem::E(const mfem::Vector &d, int &eval_err) {
  return problem->E(d, eval_err);
}

// gradient of energy objective
void ReducedOptProblem::DdE(const mfem::Vector &d, mfem::Vector &gradE) {
  problem->DdE(d, gradE);
}

mfem::Operator *ReducedOptProblem::DddE(const mfem::Vector &d) {
  return problem->DddE(d);
}

void ReducedOptProblem::g(const mfem::Vector &d, mfem::Vector &gd,
                          int &eval_err) {
  mfem::Vector gdfull(problem->GetDimM());
  gdfull = 0.0;
  problem->g(d, gdfull, eval_err);
  P->Mult(gdfull, gd);
}

mfem::Operator *ReducedOptProblem::Ddg(const mfem::Vector &d) {
  mfem::Operator *Jfull = problem->Ddg(d);
  auto Jfull_hypre = dynamic_cast<mfem::HypreParMatrix *>(Jfull);
  MFEM_VERIFY(Jfull_hypre, "expecting Ddg to be a HypreParMatrix");
  if (J) {
    delete J;
    J = nullptr;
  }
  J = ParMult(P, Jfull_hypre, true);
  return J;
}

ReducedOptProblem::~ReducedOptProblem() {
  delete P;
  if (J) {
    delete J;
  }
}

#endif // ReducedOptEqProblem
