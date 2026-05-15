#include "mfem.hpp"
#include "ObstacleProblems.hpp"



// Obstacle Problem, no essential boundary conditions enforced
// Hessian of energy term is K + M (stiffness + mass)
ObstacleProblem::ObstacleProblem(mfem::ParFiniteElementSpace *fesU_, 
                                       mfem::ParFiniteElementSpace *fesM_, 
                                       mfem_fun_ptr_type fSource) : 
                                       OptProblem()
{
   
   Init(fesU_->GetTrueDofOffsets(), fesM_->GetTrueDofOffsets());
   

   Kform.reset(new mfem::ParBilinearForm(fesU_));
   Kform->AddDomainIntegrator(new mfem::MassIntegrator);
   Kform->AddDomainIntegrator(new mfem::DiffusionIntegrator);
   Kform->Assemble();
   Kform->Finalize();
   Kform->FormSystemMatrix(ess_tdof_list, K);
   mfem::FunctionCoefficient fcoeff(fSource);
   fform.reset(new mfem::ParLinearForm(fesU_));
   fform->AddDomainIntegrator(new mfem::DomainLFIntegrator(fcoeff));
   fform->Assemble();
   f.SetSize(dimU);
   fform->ParallelAssemble(f);
   psi.SetSize(dimU);
   psi = 0.0;
   
   mfem::Vector iDiag(dimU); iDiag = 1.0;
   J.reset(GenerateHypreParMatrixFromDiagonal(dofOffsetsU, iDiag));
}

// Obstacle Problem, essential boundary conditions enforced
// Hessian of energy term is K (stiffness)
ObstacleProblem::ObstacleProblem(mfem::ParFiniteElementSpace *fesU_, 
                                       mfem::ParFiniteElementSpace *fesM_, 
				       mfem_fun_ptr_type fSource,
				       mfem_fun_ptr_type obstacleSource,
				       mfem::Array<int> tdof_list, mfem::Vector &xDC) : OptProblem()
{
   Init(fesU_->GetTrueDofOffsets(), fesM_->GetTrueDofOffsets());
   // elastic energy functional terms	
   ess_tdof_list = tdof_list;
   Kform.reset(new mfem::ParBilinearForm(fesU_));
   Kform->AddDomainIntegrator(new mfem::DiffusionIntegrator);
   Kform->Assemble();
   Kform->Finalize();
   Kform->FormSystemMatrix(ess_tdof_list, K);

   mfem::FunctionCoefficient fcoeff(fSource);
   fform.reset(new mfem::ParLinearForm(fesU_));
   fform->AddDomainIntegrator(new mfem::DomainLFIntegrator(fcoeff));
   fform->Assemble();
   f.SetSize(dimU);
   fform->ParallelAssemble(f);
   Kform->EliminateVDofsInRHS(ess_tdof_list, xDC, f);
   
   // obstacle constraints --  
   mfem::Vector iDiag(dimU); iDiag = 1.0;
   for(int i = 0; i < ess_tdof_list.Size(); i++)
   {
     iDiag(ess_tdof_list[i]) = 0.0;
   }
   J.reset(GenerateHypreParMatrixFromDiagonal(dofOffsetsU, iDiag));

   mfem::FunctionCoefficient psi_fc(obstacleSource);
   mfem::ParGridFunction psi_gf(fesU_);
   psi_gf.ProjectCoefficient(psi_fc);
   psi.SetSize(dimU);
   psi.Set(1.0, (*psi_gf.GetTrueDofs()));
   /*
    Not eliminating dofs is great with regard to the application of linear solvers e.g., AMG
    However, we need to be careful that we don't have u_i = 0 (essential BC) and
    u_i >= 0, as with the application of the interior-point method we will encounter singularities
    \log( u_i = 0)
   */

   for(int i = 0; i < ess_tdof_list.Size(); i++)
   {
     psi(ess_tdof_list[i]) -= 1.e-8;
   }
}



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
   return J.get();
}

ObstacleProblem::~ObstacleProblem()
{
}


// Obstacle Problem, no essential boundary conditions enforced
// Hessian of energy term is K + M (stiffness + mass)
ParamObstacleProblem::ParamObstacleProblem(mfem::ParFiniteElementSpace *fesU_, 
                                       mfem_fun_ptr_type fSource,
                                       mfem_fun_ptr_type obstacleSource) : 
                                       ParamOptProblem()
{
   Vh = fesU_;
   Init(Vh->GetTrueDofOffsets(), Vh->GetTrueDofOffsets());
   
   Kform.reset(new mfem::ParBilinearForm(Vh));
   Kform->AddDomainIntegrator(new mfem::MassIntegrator);
   Kform->AddDomainIntegrator(new mfem::DiffusionIntegrator);
   Kform->Assemble();
   Kform->Finalize();
   Kform->FormSystemMatrix(ess_tdof_list, K);
   mfem::FunctionCoefficient fcoeff(fSource);
   fform.reset(new mfem::ParLinearForm(Vh));
   fform->AddDomainIntegrator(new mfem::DomainLFIntegrator(fcoeff));
   fform->Assemble();
   f.SetSize(dimU);
   fform->ParallelAssemble(f);

   // provided obstacle will be default param value
   theta_default.SetSize(dimU);
   theta_default = 0.0;
   mfem::FunctionCoefficient theta_fc(obstacleSource);
   mfem::ParGridFunction theta_gf(Vh);
   theta_gf.ProjectCoefficient(theta_fc);
   theta_gf.GetTrueDofs(theta_default);
   mfem::Vector theta_default_copy(dimU);
   theta_default_copy.Set(1.0, theta_default);
   InitTheta(theta_default_copy);
   
   mfem::Vector iDiag(dimU); iDiag = 1.0;
   Jd.reset(GenerateHypreParMatrixFromDiagonal(dofOffsetsU, iDiag));
   iDiag = -1.0;
   Jth.reset(GenerateHypreParMatrixFromDiagonal(dofOffsetsU, iDiag));

   Hddgl.reset(GenerateNullHypreParMatrix(dofOffsetsU, dofOffsetsU));
   Hdthgl.reset(GenerateNullHypreParMatrix(dofOffsetsU, dofOffsetsU));
   HdthE.reset(GenerateNullHypreParMatrix(dofOffsetsU, dofOffsetsU));
}

// Obstacle Problem, no essential boundary conditions enforced
// Hessian of energy term is K + M (stiffness + mass)
ParamObstacleProblem::ParamObstacleProblem(mfem::ParFiniteElementSpace *fesU_, 
                                       mfem_fun_ptr_type fSource,
                                       mfem_fun_ptr_type obstacleSource,
				       mfem::Array<int> tdof_list,
				       mfem::Vector &ud) : 
                                       ParamOptProblem()
{
   Vh = fesU_;
   dimU = Vh->GetTrueVSize();
   dimM = dimU - tdof_list.Size();
   auto tempUOffsets = offsetsFromLocalSizes(dimU, MPI_COMM_WORLD);
   auto tempMOffsets = offsetsFromLocalSizes(dimM, MPI_COMM_WORLD);
   Init(tempUOffsets, tempMOffsets);
   delete[] tempUOffsets;
   delete[] tempMOffsets;
   HYPRE_Int dofMask[dimU];
   for (int i = 0; i < dimU; i++)
   {
      dofMask[i] = 1;
   }
   for (int i = 0; i < tdof_list.Size(); i++)
   {
     dofMask[tdof_list[i]] = 0;
   }
   R.reset(GenerateProjector(dofOffsetsM, dofOffsetsU, dofMask));
   P.reset(R->Transpose());
   Jd.reset(GenerateProjector(dofOffsetsM, dofOffsetsU, dofMask));
   


   Kform.reset(new mfem::ParBilinearForm(Vh));
   Kform->AddDomainIntegrator(new mfem::MassIntegrator);
   Kform->AddDomainIntegrator(new mfem::DiffusionIntegrator);
   Kform->Assemble();
   Kform->Finalize();
   Kform->FormSystemMatrix(tdof_list, K);
   mfem::FunctionCoefficient fcoeff(fSource);
   fform.reset(new mfem::ParLinearForm(Vh));
   fform->AddDomainIntegrator(new mfem::DomainLFIntegrator(fcoeff));
   fform->Assemble();
   f.SetSize(dimU);
   mfem::Vector F(Vh->GetTrueVSize()); F = 0.0;
   fform->ParallelAssemble(F);
   f.Set(1.0, F);
   for (int i = 0; i < tdof_list.Size(); i++)
   {
      f(tdof_list[i]) = 0.0;
   }

   // provided obstacle will be default param value
   theta_default.SetSize(dimM);
   theta_default = 0.0;
   mfem::FunctionCoefficient theta_fc(obstacleSource);
   mfem::ParGridFunction theta_gf(Vh);
   theta_gf.ProjectCoefficient(theta_fc);
   mfem::Vector theta_tmp(Vh->GetTrueVSize()); theta_tmp = 0.0;
   theta_gf.GetTrueDofs(theta_tmp);
   R->Mult(theta_tmp, theta_default);
   
   
   mfem::Vector theta_default_copy(dimM);
   theta_default_copy.Set(1.0, theta_default);
   InitTheta(theta_default_copy);
   
   mfem::Vector iDiag(dimU); iDiag = 1.0;
   iDiag = -1.0;
   Jth.reset(GenerateHypreParMatrixFromDiagonal(dofOffsetsM, iDiag));

   Hddgl.reset(GenerateNullHypreParMatrix(dofOffsetsU, dofOffsetsU));
   Hdthgl.reset(GenerateNullHypreParMatrix(dofOffsetsU, dofOffsetsM));
   HdthE.reset(GenerateNullHypreParMatrix(dofOffsetsU, dofOffsetsM));
}


double ParamObstacleProblem::E(const mfem::Vector &d, const mfem::Vector & theta, int & eval_err)
{
   MFEM_VERIFY(d.Size() == K.Width(), "ParamObstacleProblem::E - Inconsistent dimensions");
   MFEM_VERIFY(f.Size() == K.Height(), "ParamObstacleProblem::E - Inconsistent dimensions");
   eval_err = 0;
   mfem::Vector Kd(K.Height()); Kd = 0.0;
   K.Mult(d, Kd);
   return 0.5 * mfem::InnerProduct(MPI_COMM_WORLD, d, Kd) - mfem::InnerProduct(MPI_COMM_WORLD, f, d);
}

void ParamObstacleProblem::DdE(const mfem::Vector &d, const mfem::Vector &theta, mfem::Vector &gradE)
{
   MFEM_VERIFY(d.Size() == K.Width(), "ParamObstacleProblem::DdE - Inconsistent dimensions");
   MFEM_VERIFY(f.Size() == K.Height(), "ParamObstacleProblem::DdE - Inconsistent dimensions");
   gradE.SetSize(K.Height()); gradE = 0.0;
   K.Mult(d, gradE);
   gradE.Add(-1.0, f);
}

mfem::Operator * ParamObstacleProblem::DddE(const mfem::Vector &d, const mfem::Vector & theta)
{
   return &K; 
}

mfem::Operator * ParamObstacleProblem::DdthE(const mfem::Vector & d, const mfem::Vector &theta)
{
  return HdthE.get();
}


// g(d, \theta) = d - \theta >= 0
void ParamObstacleProblem::g(const mfem::Vector &d, const mfem::Vector & theta, mfem::Vector &gd, int & eval_err)
{
   eval_err = 0;
   MFEM_VERIFY(d.Size() == Jd->Width(), "ParamObstacleProblem::g - Inconsistent dimensions");
   MFEM_VERIFY(gd.Size() == Jd->Height(), "ParamObstacleProblem::g - Inconsistent dimensions");
   MFEM_VERIFY(theta.Size() == Jd->Height(), "ParamObstacleProblem::g - Inconsistent dimensions");
   Jd->Mult(d, gd);
   gd.Add(-1.0, theta);
}

mfem::Operator * ParamObstacleProblem::Ddg(const mfem::Vector &d, const mfem::Vector & theta)
{
   return Jd.get();
}

mfem::Operator * ParamObstacleProblem::Dthg(const mfem::Vector &d, const mfem::Vector & theta)
{
   return Jth.get();
}

mfem::Operator * ParamObstacleProblem::Dddgl(const mfem::Vector &d, const mfem::Vector &l, const mfem::Vector & theta)
{
   return Hddgl.get();
}

mfem::Operator * ParamObstacleProblem::Ddthgl(const mfem::Vector &d, const mfem::Vector &l, const mfem::Vector & theta)
{
   return Hdthgl.get();
}


ParamObstacleProblem::~ParamObstacleProblem()
{
}
