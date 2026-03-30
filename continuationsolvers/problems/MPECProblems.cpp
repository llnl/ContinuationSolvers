#include "mfem.hpp"
#include "MPECProblems.hpp"

/* MPEC:
  constraints coming from a parametrized optimization problem
  \min_u E(u, th), s.t., g(u, th) >= 0
  optimality conditions
  L(u, p, th, s, z) = E - p^T(g - s) - z^T s
  grad_u L = grad_u E - (grad_u g)^T p = 0
  grad_s L = p - z                     = 0
             g - s                     = 0
            \Phi(s, z)                 = 0
  U = (u, p, th, s, z)

*/ 
MPECProblem::MPECProblem(ParamOptProblem *paramopt_) : 
                                       OptEqProblem()
{
  paramopt = paramopt_;
  auto dofoffsetsu = paramopt->GetDofOffsetsU(); // u
  auto dofoffsetsg = paramopt->GetDofOffsetsM(); // g
  HYPRE_BigInt * dofoffsetsth = paramopt->GetDofOffsetsTheta(); 


  HYPRE_BigInt primalOffsets[2];
  HYPRE_BigInt constraintOffsets[2];
  for (int i = 0; i < 2; i++)
  {
    // U = (u, p, th, s, z)
    constraintOffsets[i] = dofoffsetsu[i] + 3 * dofoffsetsg[i];
    primalOffsets[i] =     constraintOffsets[i] + dofoffsetsth[i];
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
}

double MPECProblem::E(const mfem::Vector &U, int &eval_err)
{
   mfem::BlockVector Ublk(primal_blockoffsets);
   Ublk.Set(1.0, U);
   return E(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2), eval_err);
}

void MPECProblem::DuE(const mfem::Vector &u, const mfem::Vector &p, const mfem::Vector &theta, mfem::Vector &gradE)
{
   gradE = 0.0;
}

void MPECProblem::DpE(const mfem::Vector &u, const mfem::Vector &p, const mfem::Vector &theta, mfem::Vector &gradE)
{
   gradE = 0.0;
}

void MPECProblem::DthE(const mfem::Vector &u, const mfem::Vector &p, const mfem::Vector &theta, mfem::Vector &gradE)
{
   gradE = 0.0;
}

void MPECProblem::DdE(const mfem::Vector &U, mfem::Vector &gradE)
{
   mfem::BlockVector Ublk(primal_blockoffsets);
   Ublk.Set(1.0, U);
   mfem::BlockVector gradEblk(primal_blockoffsets); 
   gradEblk = 0.0;
   DuE(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2), gradEblk.GetBlock(0));
   DpE(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2), gradEblk.GetBlock(1));
   DthE(Ublk.GetBlock(0), Ublk.GetBlock(1), Ublk.GetBlock(2), gradEblk.GetBlock(2));
   gradE.Set(1.0, gradEblk);
}






void MPECProblem::RegularizedComplementarity(
    const mfem::Vector &s, const mfem::Vector & z, const double & mu, 
    mfem::Vector & phi)
{
  MFEM_VERIFY(s.Size() == z.Size(), "sizes incorrect");
  MFEM_VERIFY(s.Size() == phi.Size(), "sizes incorrect");
  MFEM_VERIFY(s.Size() == paramopt->GetDimM(), "sizes incorrect");
  for (int i = 0; i < s.Size(); i++)
  {
    phi(i) = s(i) + z(i) - 
             std::pow(std::pow(s(i), 2) + std::pow(z(i), 2) + mu, 0.5);
  }
}

mfem::Operator * MPECProblem::DsRegularizedComplementarity(
    const mfem::Vector &s, const mfem::Vector & z, const double & mu)
{
   // Verify sizes
   mfem::Vector diag(s.Size()); diag = 0.0;
   for (int i = 0; i < s.Size(); i++)
   {
     diag(i) = 1.0 + s(i) / std::pow(std::pow(s(i), 2) + std::pow(z(i), 2) + mu, 0.5);
   }
   if (DsPhi)
   {
      delete DsPhi;
   }
   DsPhi = GenerateHypreParMatrixFromDiagonal(paramopt->GetDofOffsetsM(), diag);
   return DsPhi;
}

mfem::Operator * MPECProblem::DzRegularizedComplementarity(
    const mfem::Vector &s, const mfem::Vector & z, const double & mu)
{
   // Verify sizes
   mfem::Vector diag(s.Size()); diag = 0.0;
   for (int i = 0; i < s.Size(); i++)
   {
     diag(i) = 1.0 + z(i) / std::pow(std::pow(s(i), 2) + std::pow(z(i), 2) + mu, 0.5);
   }
   if (DzPhi)
   {
      delete DzPhi;
   }
   DzPhi = GenerateHypreParMatrixFromDiagonal(paramopt->GetDofOffsetsM(), diag);
   return DzPhi;
}

void MPECProblem::g(const mfem::Vector &U, mfem::Vector & gU, int & eval_err)
{
   // cast U and gU to BlockVectors
   mfem::BlockVector Ublk(primal_blockoffsets);
   Ublk.Set(1.0, U);
   mfem::BlockVector gUblk(constraint_blockoffsets); gUblk = 0.0;
   
   // \nabla_u E - (\nabla_u g)^T p
   paramopt->DdE(Ublk.GetBlock(0), Ublk.GetBlock(2), gUblk.GetBlock(0));
   auto dg = paramopt->Ddg(Ublk.GetBlock(0), Ublk.GetBlock(2));
   dg->AddMult(Ublk.GetBlock(1), gUblk.GetBlock(0), -1.0);


   // g(u, theta) - s
   paramopt->g(Ublk.GetBlock(0), Ublk.GetBlock(2), gUblk.GetBlock(1), eval_err);
   gUblk.GetBlock(1).Add(-1.0, Ublk.GetBlock(3));


   // p - z
   gUblk.GetBlock(2).Set(1.0, Ublk.GetBlock(1));
   gUblk.GetBlock(2).Add(-1.0, Ublk.GetBlock(4));

   // Phi
   RegularizedComplementarity(Ublk.GetBlock(3), Ublk.GetBlock(4), compl_reg_const, gUblk.GetBlock(3));

   gU.Set(1.0, gUblk);
}


mfem::Operator * MPECProblem::Ddg(const mfem::Vector &U)
{
   mfem::BlockVector Ublk(primal_blockoffsets);
   Ublk.Set(1.0, U);
   auto hessddE = dynamic_cast<mfem::HypreParMatrix*>(paramopt->DddE(Ublk.GetBlock(0), Ublk.GetBlock(2)));
   auto hessdthE = dynamic_cast<mfem::HypreParMatrix*>(paramopt->DdthE(Ublk.GetBlock(0), Ublk.GetBlock(2)));
   auto jacdg = dynamic_cast<mfem::HypreParMatrix*>(paramopt->Ddg(Ublk.GetBlock(0), Ublk.GetBlock(2)));
   auto jacthg = dynamic_cast<mfem::HypreParMatrix*>(paramopt->Dthg(Ublk.GetBlock(0), Ublk.GetBlock(2)));
   MFEM_VERIFY(jacdg, "cast issue");
   MFEM_VERIFY(jacthg, "cast issue");
   MFEM_VERIFY(hessddE, "cast issue");
   MFEM_VERIFY(hessdthE, "cast issue");
   // deep copy
   mfem::HypreParMatrix *jacdg_copy = new mfem::HypreParMatrix(*jacdg);
   mfem::Vector scale(jacdg_copy->Height()); scale = -1.0;
   jacdg_copy->ScaleRows(scale);
   
   mfem::HypreParMatrix * jacdgT = jacdg_copy->Transpose();

   // deep copy
   mfem::HypreParMatrix *jacthg_copy = new mfem::HypreParMatrix(*jacthg);
   jacthg_copy->ScaleRows(scale);



   // construct diagonal blocks
   mfem::HypreParMatrix * Ident;
   mfem::HypreParMatrix * negIdent;

   scale = 1.0;
   Ident = GenerateHypreParMatrixFromDiagonal(paramopt->GetDofOffsetsM(), scale);
   scale = -1.0;
   negIdent = GenerateHypreParMatrixFromDiagonal(paramopt->GetDofOffsetsM(), scale);
 
   // after these calls the member variables DsPhi and Dzphi will be available for use
   auto temp1 = DsRegularizedComplementarity(Ublk.GetBlock(3), Ublk.GetBlock(4), compl_reg_const);
   auto temp2 = DzRegularizedComplementarity(Ublk.GetBlock(3), Ublk.GetBlock(4), compl_reg_const);


   // build the monolithic HypreParMatrix Jacobian
   mfem::Array2D<const mfem::HypreParMatrix *> blockmat(4, 5);
   blockmat(0, 0) = hessddE; // should be Hessian of Lagrangian and not just of energy
   blockmat(0, 1) = jacdgT;
   blockmat(0, 2) = hessdthE; // should be Hessian of Lagrangian and not just of energy
   blockmat(0, 3) = nullptr;
   blockmat(0, 4) = nullptr;
   blockmat(1, 0) = jacdg_copy;
   blockmat(1, 1) = nullptr;
   blockmat(1, 2) = jacthg_copy;
   blockmat(1, 3) = Ident;
   blockmat(1, 4) = nullptr;
   blockmat(2, 0) = nullptr;
   blockmat(2, 1) = Ident;
   blockmat(2, 2) = nullptr;
   blockmat(2, 3) = nullptr;
   blockmat(2, 4) = negIdent;
   blockmat(3, 0) = nullptr;
   blockmat(3, 1) = nullptr;
   blockmat(3, 2) = nullptr;
   blockmat(3, 3) = DsPhi;
   blockmat(3, 4) = DzPhi;
    
   constraintJacobian = HypreParMatrixFromBlocks(blockmat);

   delete jacdg_copy;
   delete jacthg_copy;
   delete jacdgT;
   delete Ident;
   delete negIdent;
   
   return constraintJacobian;
}
