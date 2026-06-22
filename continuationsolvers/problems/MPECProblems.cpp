#include "MPECProblems.hpp"
#include "mfem.hpp"

/* MPEC:
  constraints coming from a parametrized optimization problem
  \min_u E = E(u, th), s.t., g(u, th) >= 0
  optimality conditions
  L(u, p, th, s, z) = E - p^T(g - s) - z^T s
  grad_u L = grad_u E - (grad_u g)^T p = 0
             g - s                     = 0
  grad_s L = p - z                     = 0
            \Phi(s, z)                 = 0
  U = (u, p, th, s, z)

*/
MPECProblem::MPECProblem(ParamOptProblem *paramopt_) : OptEqProblem() {
  paramopt = paramopt_;
  auto dofoffsetsu = paramopt->GetDofOffsetsU(); // u
  auto dofoffsetsg = paramopt->GetDofOffsetsM(); // g
  HYPRE_BigInt *dofoffsetsth = paramopt->GetDofOffsetsTheta();

  HYPRE_BigInt primalOffsets[2];
  HYPRE_BigInt constraintOffsets[2];
  for (int i = 0; i < 2; i++) {
    // U = (u, p, th, s, z)
    constraintOffsets[i] = dofoffsetsu[i] + 3 * dofoffsetsg[i];
    primalOffsets[i] = constraintOffsets[i] + dofoffsetsth[i];
  }
  Init(primalOffsets, constraintOffsets);

  // what else needs to be set up here?
  // U = (u, p, th, s, z)
  primal_blockoffsets.SetSize(6);
  primal_blockoffsets[0] = 0;
  primal_blockoffsets[1] = paramopt->GetDimU();
  primal_blockoffsets[2] = paramopt->GetDimM();
  primal_blockoffsets[3] = paramopt->GetDimTheta();
  primal_blockoffsets[4] = paramopt->GetDimM();
  primal_blockoffsets[5] = paramopt->GetDimM();
  primal_blockoffsets.PartialSum();

  constraint_blockoffsets.SetSize(5);
  constraint_blockoffsets[0] = 0;
  constraint_blockoffsets[1] = paramopt->GetDimU();
  constraint_blockoffsets[2] = paramopt->GetDimM();
  constraint_blockoffsets[3] = paramopt->GetDimM();
  constraint_blockoffsets[4] = paramopt->GetDimM();
  constraint_blockoffsets.PartialSum();

  // default assignement for Hessian blocks
  HuuE.reset(GenerateNullHypreParMatrix(paramopt->GetDofOffsetsU(),
                                        paramopt->GetDofOffsetsU()));
  HupE.reset(GenerateNullHypreParMatrix(paramopt->GetDofOffsetsU(),
                                        paramopt->GetDofOffsetsM()));
  HuthE.reset(GenerateNullHypreParMatrix(paramopt->GetDofOffsetsU(),
                                         paramopt->GetDofOffsetsTheta()));
  HppE.reset(GenerateNullHypreParMatrix(paramopt->GetDofOffsetsM(),
                                        paramopt->GetDofOffsetsM()));
  HpthE.reset(GenerateNullHypreParMatrix(paramopt->GetDofOffsetsM(),
                                         paramopt->GetDofOffsetsTheta()));
  HththE.reset(GenerateNullHypreParMatrix(paramopt->GetDofOffsetsTheta(),
                                          paramopt->GetDofOffsetsTheta()));

  // Jacobian blocks
  mfem::Vector diag(paramopt->GetDimM());
  diag = 0.0;

  diag = -1.0;
  dg1ds.reset(
      GenerateHypreParMatrixFromDiagonal(paramopt->GetDofOffsetsM(), diag));

  diag = 1.0;
  dg2dp.reset(
      GenerateHypreParMatrixFromDiagonal(paramopt->GetDofOffsetsM(), diag));

  diag = -1.0;
  dg2dz.reset(
      GenerateHypreParMatrixFromDiagonal(paramopt->GetDofOffsetsM(), diag));

  slackScale.SetSize(paramopt->GetDimM());
  slackScale = 1.0;
  mfem::Vector tempMassLump(paramopt->GetDimM());
  tempMassLump = 1.0;
  ParamObstacleProblem *temp_ptr =
      dynamic_cast<ParamObstacleProblem *>(paramopt);
  if (temp_ptr) {
    temp_ptr->GetConstraintMassLump(tempMassLump);
  }
  slackScale /= tempMassLump;
}

double MPECProblem::E(const mfem::Vector &U, int &eval_err) {
  mfem::BlockVector Ublk(primal_blockoffsets);
  Ublk.Set(1.0, U);
  return E(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2), eval_err);
}

void MPECProblem::DuE(const mfem::Vector &u, const mfem::Vector &p,
                      const mfem::Vector &theta, mfem::Vector &gradE) {
  gradE = 0.0;
}

void MPECProblem::DpE(const mfem::Vector &u, const mfem::Vector &p,
                      const mfem::Vector &theta, mfem::Vector &gradE) {
  gradE = 0.0;
}

void MPECProblem::DthE(const mfem::Vector &u, const mfem::Vector &p,
                       const mfem::Vector &theta, mfem::Vector &gradE) {
  gradE = 0.0;
}

void MPECProblem::DdE(const mfem::Vector &U, mfem::Vector &gradE) {
  mfem::BlockVector Ublk(primal_blockoffsets);
  Ublk.Set(1.0, U);
  mfem::BlockVector gradEblk(primal_blockoffsets);
  gradEblk = 0.0;
  DuE(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2),
      gradEblk.GetBlock(0));
  DpE(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2),
      gradEblk.GetBlock(1));
  DthE(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2),
       gradEblk.GetBlock(2));
  gradE.Set(1.0, gradEblk);
}

mfem::Operator *MPECProblem::DuuE(const mfem::Vector &u, const mfem::Vector &p,
                                  const mfem::Vector &theta) {
  // assume zero
  // default behavior, user should override base class implementation if this
  // Hessian is nonzero
  return HuuE.get();
}

mfem::Operator *MPECProblem::DupE(const mfem::Vector &u, const mfem::Vector &p,
                                  const mfem::Vector &theta) {
  // assume zero
  // default behavior, user should override base class implementation if this
  // Hessian is nonzero
  return HupE.get();
}

mfem::Operator *MPECProblem::DuthE(const mfem::Vector &u, const mfem::Vector &p,
                                   const mfem::Vector &theta) {
  // assume zero
  // default behavior, user should override base class implementation if this
  // Hessian is nonzero
  return HuthE.get();
}

mfem::Operator *MPECProblem::DppE(const mfem::Vector &u, const mfem::Vector &p,
                                  const mfem::Vector &theta) {
  // assume zero
  // default behavior, user should override base class implementation if this
  // Hessian is nonzero
  return HppE.get();
}

mfem::Operator *MPECProblem::DpthE(const mfem::Vector &u, const mfem::Vector &p,
                                   const mfem::Vector &theta) {
  // assume zero
  // default behavior, user should override base class implementation if this
  // Hessian is nonzero
  return HpthE.get();
}

mfem::Operator *MPECProblem::DththE(const mfem::Vector &u,
                                    const mfem::Vector &p,
                                    const mfem::Vector &theta) {
  // assume zero
  // default behavior, user should override base class implementation if this
  // Hessian is nonzero
  return HththE.get();
}

mfem::Operator *MPECProblem::DddE(const mfem::Vector &U) {
  // cast to block vector, evaluate individual blocks, form monolithic matrix
  mfem::BlockVector Ublk(primal_blockoffsets);
  Ublk.Set(1.0, U);

  auto Huu = dynamic_cast<mfem::HypreParMatrix *>(
      DuuE(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2)));
  auto Hup = dynamic_cast<mfem::HypreParMatrix *>(
      DupE(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2)));
  auto Huth = dynamic_cast<mfem::HypreParMatrix *>(
      DuthE(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2)));
  auto Hpp = dynamic_cast<mfem::HypreParMatrix *>(
      DppE(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2)));
  auto Hpth = dynamic_cast<mfem::HypreParMatrix *>(
      DpthE(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2)));
  auto Hthth = dynamic_cast<mfem::HypreParMatrix *>(
      DththE(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2)));
  MFEM_VERIFY(Huu, "cast issue");
  MFEM_VERIFY(Hup, "cast issue");
  MFEM_VERIFY(Huth, "cast issue");
  MFEM_VERIFY(Hpp, "cast issue");
  MFEM_VERIFY(Hpth, "cast issue");
  MFEM_VERIFY(Hthth, "cast issue");

  std::unique_ptr<mfem::HypreParMatrix> Hpu;
  std::unique_ptr<mfem::HypreParMatrix> Hthu;
  std::unique_ptr<mfem::HypreParMatrix> Hthp;
  Hpu.reset(Hup->Transpose());
  Hthu.reset(Huth->Transpose());
  Hthp.reset(Hpth->Transpose());

  // build the monolithic HypreParMatrix Jacobian
  mfem::Array2D<const mfem::HypreParMatrix *> blockmat(
      primal_blockoffsets.Size() - 1, primal_blockoffsets.Size() - 1);
  for (int i = 0; i < blockmat.NumRows(); i++) {
    for (int j = 0; j < blockmat.NumCols(); j++) {
      blockmat(i, j) = nullptr;
    }
  }
  blockmat(0, 0) = Huu;
  blockmat(0, 1) = Hup;
  blockmat(0, 2) = Huth;
  blockmat(1, 0) = Hpu.get();
  blockmat(1, 1) = Hpp;
  blockmat(1, 2) = Hpth;
  blockmat(2, 0) = Hthu.get();
  blockmat(2, 1) = Hthp.get();
  blockmat(2, 2) = Hthth;

  // the following is adding some zero matrices with appropriate sizes to ensure
  // the monolithic Hessian is of the expected size Hus
  std::unique_ptr<mfem::HypreParMatrix> Hus;
  Hus.reset(GenerateNullHypreParMatrix(paramopt->GetDofOffsetsU(),
                                       paramopt->GetDofOffsetsM()));
  std::unique_ptr<mfem::HypreParMatrix> Hsu;
  Hsu.reset(GenerateNullHypreParMatrix(paramopt->GetDofOffsetsM(),
                                       paramopt->GetDofOffsetsU()));
  // dim(s) = dim(z) so we can reuse certain blocks
  blockmat(0, 3) = Hus.get();
  blockmat(0, 4) = Hus.get();
  blockmat(3, 0) = Hsu.get();
  blockmat(4, 0) = Hsu.get();
  // end adding in the zero matrix blocks

  HE.reset(HypreParMatrixFromBlocks(blockmat));

  MFEM_VERIFY(HE->Width() == dimU, "size issue");
  MFEM_VERIFY(HE->Height() == dimU, "size issue");

  return HE.get();
}

void MPECProblem::RegularizedComplementarity(const mfem::Vector &s,
                                             const mfem::Vector &z,
                                             const double &mu,
                                             mfem::Vector &phi) {
  MFEM_VERIFY(s.Size() == z.Size(), "sizes incorrect");
  MFEM_VERIFY(s.Size() == phi.Size(), "sizes incorrect");
  MFEM_VERIFY(s.Size() == paramopt->GetDimM(), "sizes incorrect");
  // ss = "scaled slack"
  mfem::Vector ss(s.Size());
  ss.Set(1.0, s);
  ss *= slackScale;
  for (int i = 0; i < s.Size(); i++) {
    phi(i) = ss(i) + z(i) -
             std::pow(std::pow(ss(i), 2) + std::pow(z(i), 2) + mu, 0.5);
  }
}

mfem::Operator *MPECProblem::DsRegularizedComplementarity(const mfem::Vector &s,
                                                          const mfem::Vector &z,
                                                          const double &mu) {
  MFEM_VERIFY(s.Size() == z.Size(), "sizes incorrect");
  MFEM_VERIFY(s.Size() == paramopt->GetDimM(), "sizes incorrect");
  mfem::Vector diag(s.Size());
  diag = 0.0;
  // ss = "scaled slack"
  mfem::Vector ss(s.Size());
  ss.Set(1.0, s);
  ss *= slackScale;
  for (int i = 0; i < s.Size(); i++) {
    diag(i) =
        slackScale(i) *
        (1.0 -
         ss(i) / std::pow(std::pow(ss(i), 2) + std::pow(z(i), 2) + mu, 0.5));
  }
  DsPhi.reset(
      GenerateHypreParMatrixFromDiagonal(paramopt->GetDofOffsetsM(), diag));
  return DsPhi.get();
}

mfem::Operator *MPECProblem::DzRegularizedComplementarity(const mfem::Vector &s,
                                                          const mfem::Vector &z,
                                                          const double &mu) {
  MFEM_VERIFY(s.Size() == z.Size(), "sizes incorrect");
  MFEM_VERIFY(s.Size() == paramopt->GetDimM(), "sizes incorrect");
  // ss = "scaled slack"
  mfem::Vector ss(s.Size());
  ss.Set(1.0, s);
  ss *= slackScale;
  mfem::Vector diag(s.Size());
  diag = 0.0;
  for (int i = 0; i < s.Size(); i++) {
    diag(i) =
        1.0 - z(i) / std::pow(std::pow(ss(i), 2) + std::pow(z(i), 2) + mu, 0.5);
  }
  DzPhi.reset(
      GenerateHypreParMatrixFromDiagonal(paramopt->GetDofOffsetsM(), diag));
  return DzPhi.get();
}

mfem::Operator *MPECProblem::DsslRegularizedComplementarity(
    const mfem::Vector &s, const mfem::Vector &z, const mfem::Vector &l,
    const double &mu) {
  MFEM_VERIFY(s.Size() == z.Size(), "sizes incorrect");
  MFEM_VERIFY(s.Size() == l.Size(), "sizes incorrect");
  MFEM_VERIFY(s.Size() == paramopt->GetDimM(), "sizes incorrect");
  mfem::Vector diag(s.Size());
  diag = 0.0;
  // ss = "scaled slack"
  mfem::Vector ss(s.Size());
  ss.Set(1.0, s);
  ss *= slackScale;
  for (int i = 0; i < s.Size(); i++) {
    diag(i) = -1.0 * std::pow(slackScale(i), 2) * (std::pow(z(i), 2) + mu) /
              std::pow(std::pow(ss(i), 2) + std::pow(z(i), 2) + mu, 1.5);
    diag(i) *= l(i);
  }
  DsslPhi.reset(
      GenerateHypreParMatrixFromDiagonal(paramopt->GetDofOffsetsM(), diag));
  return DsslPhi.get();
}

mfem::Operator *MPECProblem::DszlRegularizedComplementarity(
    const mfem::Vector &s, const mfem::Vector &z, const mfem::Vector &l,
    const double &mu) {
  MFEM_VERIFY(s.Size() == z.Size(), "sizes incorrect");
  MFEM_VERIFY(s.Size() == l.Size(), "sizes incorrect");
  MFEM_VERIFY(s.Size() == paramopt->GetDimM(), "sizes incorrect");
  mfem::Vector diag(s.Size());
  diag = 0.0;
  // ss = "scaled slack"
  mfem::Vector ss(s.Size());
  ss.Set(1.0, s);
  ss *= slackScale;
  for (int i = 0; i < s.Size(); i++) {
    diag(i) = slackScale(i) * ss(i) * z(i) /
              std::pow(std::pow(ss(i), 2) + std::pow(z(i), 2) + mu, 1.5);
    diag(i) *= l(i);
  }
  DszlPhi.reset(
      GenerateHypreParMatrixFromDiagonal(paramopt->GetDofOffsetsM(), diag));
  return DszlPhi.get();
}

mfem::Operator *MPECProblem::DzzlRegularizedComplementarity(
    const mfem::Vector &s, const mfem::Vector &z, const mfem::Vector &l,
    const double &mu) {
  MFEM_VERIFY(s.Size() == z.Size(), "sizes incorrect");
  MFEM_VERIFY(s.Size() == l.Size(), "sizes incorrect");
  MFEM_VERIFY(s.Size() == paramopt->GetDimM(), "sizes incorrect");
  mfem::Vector diag(s.Size());
  diag = 0.0;
  // ss = "scaled slack"
  mfem::Vector ss(s.Size());
  ss.Set(1.0, s);
  ss *= slackScale;
  for (int i = 0; i < s.Size(); i++) {
    diag(i) = -1.0 * (std::pow(ss(i), 2) + mu) /
              std::pow(std::pow(ss(i), 2) + std::pow(z(i), 2) + mu, 1.5);
    diag(i) *= l(i);
  }
  DzzlPhi.reset(
      GenerateHypreParMatrixFromDiagonal(paramopt->GetDofOffsetsM(), diag));
  return DzzlPhi.get();
}

void MPECProblem::g(const mfem::Vector &U, mfem::Vector &gU, int &eval_err) {
  // cast U and gU to BlockVectors
  mfem::BlockVector Ublk(primal_blockoffsets);
  Ublk.Set(1.0, U);
  mfem::BlockVector gUblk(constraint_blockoffsets);
  gUblk = 0.0;

  // \nabla_u E - (\nabla_u g)^T p
  paramopt->DdE(Ublk.GetBlock(0), Ublk.GetBlock(2),
                gUblk.GetBlock(0)); // \nabla_u E
  auto dg = paramopt->Ddg(Ublk.GetBlock(0),
                          Ublk.GetBlock(2)); // compute Jacobian \nabla_u g
  dg->AddMultTranspose(Ublk.GetBlock(1), gUblk.GetBlock(0), -1.0);

  // g(u, theta) - s
  paramopt->g(Ublk.GetBlock(0), Ublk.GetBlock(2), gUblk.GetBlock(1),
              eval_err);                         // g
  gUblk.GetBlock(1).Add(-1.0, Ublk.GetBlock(3)); // -s

  // p - z
  gUblk.GetBlock(2).Set(1.0, Ublk.GetBlock(1));
  gUblk.GetBlock(2).Add(-1.0, Ublk.GetBlock(4));

  // Phi(s, z)
  RegularizedComplementarity(Ublk.GetBlock(3), Ublk.GetBlock(4),
                             compl_reg_const, gUblk.GetBlock(3));

  gU.Set(1.0, gUblk);
}

mfem::Operator *MPECProblem::Ddg(const mfem::Vector &U) {
  mfem::BlockVector Ublk(primal_blockoffsets);
  Ublk.Set(1.0, U);
  auto hessddE = dynamic_cast<mfem::HypreParMatrix *>(
      paramopt->DddE(Ublk.GetBlock(0), Ublk.GetBlock(2)));
  auto hessddgl = dynamic_cast<mfem::HypreParMatrix *>(
      paramopt->Dddgl(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2)));
  auto hessdthgl = dynamic_cast<mfem::HypreParMatrix *>(
      paramopt->Ddthgl(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2)));
  auto hessdthE = dynamic_cast<mfem::HypreParMatrix *>(
      paramopt->DdthE(Ublk.GetBlock(0), Ublk.GetBlock(2)));
  auto jacdg = dynamic_cast<mfem::HypreParMatrix *>(
      paramopt->Ddg(Ublk.GetBlock(0), Ublk.GetBlock(2)));
  auto jacthg = dynamic_cast<mfem::HypreParMatrix *>(
      paramopt->Dthg(Ublk.GetBlock(0), Ublk.GetBlock(2)));
  MFEM_VERIFY(jacdg, "cast issue");
  MFEM_VERIFY(jacthg, "cast issue");
  MFEM_VERIFY(hessddE, "cast issue");
  MFEM_VERIFY(hessdthE, "cast issue");
  MFEM_VERIFY(hessddgl, "cast issue");
  MFEM_VERIFY(hessdthgl, "cast issue");

  std::unique_ptr<mfem::HypreParMatrix> jacdgT;
  jacdgT.reset(jacdg->Transpose());
  mfem::Vector scale(jacdgT->Height());
  scale = -1.0;
  jacdgT->ScaleRows(scale);

  // after these calls the member variables DsPhi and Dzphi will be available
  // for use
  auto temp1 = DsRegularizedComplementarity(Ublk.GetBlock(3), Ublk.GetBlock(4),
                                            compl_reg_const);
  auto temp2 = DzRegularizedComplementarity(Ublk.GetBlock(3), Ublk.GetBlock(4),
                                            compl_reg_const);

  // build the monolithic HypreParMatrix Jacobian
  mfem::Array2D<const mfem::HypreParMatrix *> blockmat(
      constraint_blockoffsets.Size() - 1, primal_blockoffsets.Size() - 1);
  for (int i = 0; i < blockmat.NumRows(); i++) {
    for (int j = 0; j < blockmat.NumCols(); j++) {
      blockmat(i, j) = nullptr;
    }
  }

  std::unique_ptr<mfem::HypreParMatrix> hessddL;
  hessddL.reset(ParAdd(hessddE, hessddgl));
  std::unique_ptr<mfem::HypreParMatrix> hessdthL;
  hessdthL.reset(ParAdd(hessdthE, hessdthgl));
  blockmat(0, 0) = hessddL.get();
  blockmat(0, 1) = jacdgT.get();
  blockmat(0, 2) = hessdthL.get();
  blockmat(1, 0) = jacdg;
  blockmat(1, 2) = jacthg;
  blockmat(1, 3) = dg1ds.get(); // d/ds (g(u, \theta) - s)
  blockmat(2, 1) = dg2dp.get(); // d/dp (p - z)
  blockmat(2, 4) = dg2dz.get(); // d/dz (p - z)
  blockmat(3, 3) = DsPhi.get();
  blockmat(3, 4) = DzPhi.get();

  constraintJacobian.reset(HypreParMatrixFromBlocks(blockmat));

  return constraintJacobian.get();
}

// note neglecting third derivatives of constraint function in the parametrized
// optimization problem
mfem::Operator *MPECProblem::Dddgl(const mfem::Vector &U,
                                   const mfem::Vector &l) {
  mfem::BlockVector Ublk(primal_blockoffsets);
  Ublk.Set(1.0, U);
  mfem::BlockVector lblk(constraint_blockoffsets);
  lblk.Set(1.0, l);
  // build the monolithic HypreParMatrix Hessian
  mfem::Array2D<const mfem::HypreParMatrix *> blockmat(
      primal_blockoffsets.Size() - 1, primal_blockoffsets.Size() - 1);
  for (int i = 0; i < blockmat.NumRows(); i++) {
    for (int j = 0; j < blockmat.NumCols(); j++) {
      blockmat(i, j) = nullptr;
    }
  }
  // after these calls the member variables DsslPhi, DszlPhi and DzzlPhi will be
  // available for use
  auto temp1 = DsslRegularizedComplementarity(
      Ublk.GetBlock(3), Ublk.GetBlock(4), lblk.GetBlock(3), compl_reg_const);
  auto temp2 = DszlRegularizedComplementarity(
      Ublk.GetBlock(3), Ublk.GetBlock(4), lblk.GetBlock(3), compl_reg_const);
  auto temp3 = DzzlRegularizedComplementarity(
      Ublk.GetBlock(3), Ublk.GetBlock(4), lblk.GetBlock(3), compl_reg_const);

  mfem::HypreParMatrix *HuuuEl1_mat = dynamic_cast<mfem::HypreParMatrix *>(
      paramopt->DdddEl(Ublk.GetBlock(0), lblk.GetBlock(0), Ublk.GetBlock(2)));
  mfem::HypreParMatrix *Huugl2_mat = dynamic_cast<mfem::HypreParMatrix *>(
      paramopt->Dddgl(Ublk.GetBlock(0), lblk.GetBlock(1), Ublk.GetBlock(2)));
  MFEM_VERIFY(HuuuEl1_mat && Huugl2_mat, "CAST issue");
  std::unique_ptr<mfem::HypreParMatrix> Huucl;
  Huucl.reset(ParAdd(HuuuEl1_mat, Huugl2_mat));

  mfem::HypreParMatrix *HththuEl1_mat = dynamic_cast<mfem::HypreParMatrix *>(
      paramopt->DththdEl(Ublk.GetBlock(0), lblk.GetBlock(0), Ublk.GetBlock(2)));
  mfem::HypreParMatrix *Hththgl2_mat = dynamic_cast<mfem::HypreParMatrix *>(
      paramopt->Dththgl(Ublk.GetBlock(0), lblk.GetBlock(1), Ublk.GetBlock(2)));
  MFEM_VERIFY(HththuEl1_mat && Hththgl2_mat, "CAST issue");
  std::unique_ptr<mfem::HypreParMatrix> Hththcl;
  Hththcl.reset(ParAdd(HththuEl1_mat, Hththgl2_mat));

  //
  mfem::HypreParMatrix *Hpucl_mat = dynamic_cast<mfem::HypreParMatrix *>(
      paramopt->Dddgl2(Ublk.GetBlock(0), lblk.GetBlock(0), Ublk.GetBlock(2)));
  std::unique_ptr<mfem::HypreParMatrix> Hpucl;
  std::unique_ptr<mfem::HypreParMatrix> Hupcl;
  if (Hpucl_mat) {
    Hpucl.reset(new mfem::HypreParMatrix(*Hpucl_mat)); // deep copy
    // scale rows as the block is -1.0 * (nabla_(u,u)g) l
    mfem::Vector scale(Hpucl_mat->Height());
    scale = -1.0;
    Hpucl->ScaleRows(scale);
    Hupcl.reset(Hpucl->Transpose());
    blockmat(1, 0) = Hpucl.get();
    blockmat(0, 1) = Hupcl.get();
  }

  mfem::HypreParMatrix *Hpthcl_mat = dynamic_cast<mfem::HypreParMatrix *>(
      paramopt->Dthdgl2(Ublk.GetBlock(0), lblk.GetBlock(0), Ublk.GetBlock(2)));
  std::unique_ptr<mfem::HypreParMatrix> Hpthcl;
  std::unique_ptr<mfem::HypreParMatrix> Hthpcl;
  if (Hpthcl_mat) {
    Hpthcl.reset(new mfem::HypreParMatrix(*Hpthcl_mat)); // deep copy
    // scale rows as the block is -1.0 * (nabla_(th,u)g) l
    mfem::Vector scale(Hpthcl_mat->Height());
    scale = -1.0;
    Hpthcl->ScaleRows(scale);
    Hthpcl.reset(Hpthcl->Transpose());
    blockmat(1, 2) = Hpthcl.get();
    blockmat(2, 1) = Hthpcl.get();
  }

  blockmat(0, 0) = Huucl.get();
  blockmat(2, 2) = Hththcl.get();
  blockmat(3, 3) = DsslPhi.get();
  blockmat(3, 4) = DszlPhi.get();
  blockmat(4, 3) = DszlPhi.get();
  blockmat(4, 4) = DzzlPhi.get();
  constraintHessian.reset(HypreParMatrixFromBlocks(blockmat));
  return constraintHessian.get();
}

MPECProblem::~MPECProblem() {}

ObstacleDesignProblem::ObstacleDesignProblem(ParamOptProblem *paramopt_)
    : MPECProblem(paramopt_) {
  // reconfigure HththE
  int dimTheta = paramopt->GetDimTheta();
  mfem::Vector diag(dimTheta);
  diag = 1.0;
  HththE.reset(
      GenerateHypreParMatrixFromDiagonal(paramopt->GetDofOffsetsTheta(), diag));
}

double ObstacleDesignProblem::E(const mfem::Vector &U, int &eval_err) {
  mfem::BlockVector Ublk(primal_blockoffsets);
  Ublk.Set(1.0, U);
  return E(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2), eval_err);
}

double ObstacleDesignProblem::E(const mfem::Vector &u, const mfem::Vector &p,
                                const mfem::Vector &theta, int &eval_err) {
  eval_err = 0;
  mfem::Vector shift(theta.Size());
  shift = 0.0;
  mfem::Vector temp(theta.Size());
  temp = 0.0;
  temp.Set(1.0, theta);
  temp.Add(-1.0, shift);

  return 0.5 * InnerProduct(MPI_COMM_WORLD, temp, temp);
}

void ObstacleDesignProblem::DthE(const mfem::Vector &u, const mfem::Vector &p,
                                 const mfem::Vector &theta,
                                 mfem::Vector &gradE) {
  mfem::Vector shift(theta.Size());
  shift = 0.0;
  gradE.Set(1.0, theta);
  gradE.Add(-1.0, shift);
}

mfem::Operator *ObstacleDesignProblem::DththE(const mfem::Vector &u,
                                              const mfem::Vector &p,
                                              const mfem::Vector &theta) {
  return HththE.get();
}
