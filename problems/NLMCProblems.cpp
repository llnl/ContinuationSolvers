#include "mfem.hpp"
#include "NLMCProblems.hpp"
#include "../utilities.hpp"



GeneralNLMCProblem::GeneralNLMCProblem() 
{ 
  dofOffsetsx = nullptr;
  dofOffsetsy = nullptr;
  label = -1;
}

void GeneralNLMCProblem::Init(HYPRE_BigInt * dofOffsetsx_, HYPRE_BigInt * dofOffsetsy_)
{
  dofOffsetsx = new HYPRE_BigInt[2];
  dofOffsetsy = new HYPRE_BigInt[2];
  for(int i = 0; i < 2; i++)
  {
    dofOffsetsx[i] = dofOffsetsx_[i];
    dofOffsetsy[i] = dofOffsetsy_[i];
  }
  dimx = dofOffsetsx[1] - dofOffsetsx[0];
  dimy = dofOffsetsy[1] - dofOffsetsy[0];

  xyoffsets.SetSize(3); 
  xyoffsets[0] = 0;
  xyoffsets[1] = dimx;
  xyoffsets[2] = dimy;
  xyoffsets.PartialSum(); 
  MPI_Allreduce(&dimx, &dimxglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dimy, &dimyglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}



GeneralNLMCProblem::~GeneralNLMCProblem() 
{ 
   if (dofOffsetsx != nullptr)
   {
      delete[] dofOffsetsx;
   }
   if (dofOffsetsy != nullptr)
   {
      delete[] dofOffsetsy;
   }
}


// ------------------------------------


OptNLMCProblem::OptNLMCProblem(OptProblem * optproblem_)
{
   optproblem = optproblem_;
   
   // x = dual variable
   // y = primal variable
   Init(optproblem->GetDofOffsetsM(), optproblem->GetDofOffsetsU());

   {
      mfem::Vector temp(dimx); temp = 0.0;
      dFdx = GenerateHypreParMatrixFromDiagonal(dofOffsetsx, temp);
   }
}

// F(x, y) = g(y)
void OptNLMCProblem::F(const mfem::Vector & x, const mfem::Vector & y, mfem::Vector & feval, int & eval_err, bool /*new_pt*/) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && feval.Size() == dimx, "OptNLMCProblem::F -- Inconsistent dimensions");
  optproblem->g(y, feval, eval_err);
}




// Q(x, y) = \nabla_y L(y, x) = \nabla_y E(y) - (dg(y)/ dy)^T x
void OptNLMCProblem::Q(const mfem::Vector & x, const mfem::Vector & y, mfem::Vector & qeval, int &eval_err, bool /*new_pt*/) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && qeval.Size() == dimy, "OptNLMCProblem::Q -- Inconsistent dimensions");
  
  optproblem->DdE(y, qeval);
  
  mfem::Operator * J = optproblem->Ddg(y);
  mfem::Vector temp(dimy); temp = 0.0;
  J->MultTranspose(x, temp);
  
  eval_err = 0;
  qeval.Add(-1.0, temp);
}


// dF/dx = 0
mfem::Operator * OptNLMCProblem::DxF(const mfem::Vector & /*x*/, const mfem::Vector & /*y*/, bool /*new_pt*/)
{
   return dFdx;
}

// dF/dy = dg/dy
mfem::Operator * OptNLMCProblem::DyF(const mfem::Vector & /*x*/, const mfem::Vector & y, bool /*new_pt*/)
{
   return optproblem->Ddg(y);
}


// dQ/dx = -(dg/dy)^T
mfem::Operator * OptNLMCProblem::DxQ(const mfem::Vector & /*x*/, const mfem::Vector & y, bool /*new_pt*/)
{
   mfem::Operator * J = optproblem->Ddg(y);
   auto Jhypre = dynamic_cast<mfem::HypreParMatrix *>(J);
   MFEM_VERIFY(Jhypre, "expecting a HypreParMatrix Ddg");
   if (dQdx)
   {
      delete dQdx;
      dQdx = nullptr;
   }
   dQdx = Jhypre->Transpose();
   mfem::Vector temp(dimy); temp = -1.0;
   dQdx->ScaleRows(temp);
   return dQdx;
}


// dQdy = Hessian(E) - second order derivaives in g
mfem::Operator * OptNLMCProblem::DyQ(const mfem::Vector & /*x*/, const mfem::Vector & y, bool /*new_pt*/)
{
   return optproblem->DddE(y);
}


OptNLMCProblem::~OptNLMCProblem()
{
   if (dFdx)
   {
      delete dFdx;
   }
   if (dQdx)
   {
      delete dQdx;
   }
};

EqualityConstrainedHomotopyProblem::EqualityConstrainedHomotopyProblem()
{
   EqualityConstrainedHomotopyInit();
};

void EqualityConstrainedHomotopyProblem::EqualityConstrainedHomotopyInit()
{
   y_partition.SetSize(3);
   adjoint_solver = new DirectSolver();
   fixed_tdof_list_.SetSize(0);
   disp_tdof_list_.SetSize(0);
   uDC_.SetSize(0);
};


EqualityConstrainedHomotopyProblem::EqualityConstrainedHomotopyProblem(mfem::Array<int> fixed_tdof_list, mfem::Array<int> disp_tdof_list, const mfem::Vector uDC)
{
  EqualityConstrainedHomotopyInit();
  fixed_tdof_list_ = fixed_tdof_list;
  disp_tdof_list_ = disp_tdof_list;
  uDC_ = uDC;
  has_essential_dofs = true;  
};


mfem::Vector EqualityConstrainedHomotopyProblem::GetDisplacement(mfem::Vector &X)
{
   MFEM_VERIFY(X.Size() == dimx + dimy, "input vector of an invalid size");   
   mfem::Vector ur(X, 0, dimu_); // reduced
   mfem::Vector u(dimufull_); u = 0.0;
   prolongation_->Mult(ur, u);
   return u;
};

mfem::Vector EqualityConstrainedHomotopyProblem::GetLagrangeMultiplier(mfem::Vector &X)
{
   MFEM_VERIFY(X.Size() == dimx + dimy, "input vector of an invalid size");   
   mfem::Vector multiplier(X, dimu_, dimc_);
   return multiplier;
};

void EqualityConstrainedHomotopyProblem::SetSizes(int dimu, int dimc)
{ 
  std::unique_ptr<HYPRE_BigInt> uOffsets;
  uOffsets.reset(offsetsFromLocalSizes(dimu, MPI_COMM_WORLD));
  
  std::unique_ptr<HYPRE_BigInt[]> cOffsets = std::make_unique<HYPRE_BigInt[]>(2);
  HYPRE_BigInt cOffset = 0;
  MPI_Scan(&dimc, &cOffset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  cOffset -= dimc;
  cOffsets[0] = cOffset;
  cOffsets[1] = cOffsets[0] + dimc;
  SetSizes(uOffsets.get(), cOffsets.get());
}



void EqualityConstrainedHomotopyProblem::SetSizes(HYPRE_BigInt * uOffsets, HYPRE_BigInt * cOffsets)
{
   set_sizes = true;
   dimu_ = uOffsets[1] - uOffsets[0];
   dimc_ = cOffsets[1] - cOffsets[0];
   MPI_Allreduce(&dimc_, &dimcglb_, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&dimu_, &dimuglb_, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   uOffsets_ = new HYPRE_BigInt[2];
   cOffsets_ = new HYPRE_BigInt[2];
   for (int i = 0; i < 2; i++)
   {
      uOffsets_[i] = uOffsets[i];
      cOffsets_[i] = cOffsets[i];
   }
   
   y_partition[0] = 0;
   y_partition[1] = dimu_;
   y_partition[2] = dimc_;
   y_partition.PartialSum();

   HYPRE_BigInt dofOffsets[2];
   HYPRE_BigInt complementarityOffsets[2];
   for (int i = 0; i < 2; i++) {
      dofOffsets[i] = uOffsets_[i] + cOffsets_[i];
      complementarityOffsets[i] = 0;
   }
   Init(complementarityOffsets, dofOffsets);
   
   // dF / dx 0 x 0 matrix
   {
     int nentries = 0;
     auto temp = new mfem::SparseMatrix(dimx, dimxglb, nentries);
     dFdx = GenerateHypreParMatrixFromSparseMatrix(dofOffsetsx, dofOffsetsx, temp);
     delete temp;
   }

   // dF / dy 0 x dimy matrix
   {
     int nentries = 0;
     auto temp = new mfem::SparseMatrix(dimx, dimyglb, nentries);
     dFdy = GenerateHypreParMatrixFromSparseMatrix(dofOffsetsx, dofOffsetsy, temp);
     delete temp;
   }

   // dQ / dx dimy x 0 matrix
   {
     int nentries = 0;
     auto temp = new mfem::SparseMatrix(dimy, dimxglb, nentries);
     dQdx = GenerateHypreParMatrixFromSparseMatrix(dofOffsetsy, dofOffsetsx, temp);
     delete temp;
   }
   q_cache.SetSize(dimy); q_cache = 0.0;

   // construct prolongation and restriction operators
   dimufull_ = dimu_;
   if (has_essential_dofs)
   {
      dimufull_ = uDC_.Size();
   }
   else
   {
      uDC_.SetSize(dimufull_);
      uDC_ = 0.0;
   }
   ufull_.SetSize(dimufull_); ufull_ = 0.0;

   // uDC should be a vector on the entire space (essential + non-essential dofs)
   std::unique_ptr<HYPRE_BigInt> ufullOffsets;
   ufullOffsets.reset(offsetsFromLocalSizes(dimufull_, MPI_COMM_WORLD)); 


   // mask out essential dofs
   std::unique_ptr<HYPRE_Int[]> mask = std::make_unique<HYPRE_Int[]>(static_cast<size_t>(dimufull_));
   for (int i = 0; i < dimufull_; i++) {
     mask[static_cast<size_t>(i)] = 1;
   }
   for (int i = 0; i < fixed_tdof_list_.Size(); i++) {
     mask[static_cast<size_t>(fixed_tdof_list_[i])] = 0;
   }
   for (int i = 0; i < disp_tdof_list_.Size(); i++) {
     mask[static_cast<size_t>(disp_tdof_list_[i])] = 0;
   }

   // now mask out any dofs that aren't disp BC dofs
   std::unique_ptr<HYPRE_Int[]> dispmask = std::make_unique<HYPRE_Int[]>(static_cast<size_t>(dimufull_));
   for (int i = 0; i < dimufull_; i++) {
     dispmask[static_cast<size_t>(i)] = 0;
   }
   for (int i = 0; i < disp_tdof_list_.Size(); i++) {
     dispmask[static_cast<size_t>(disp_tdof_list_[i])] = 1;
   }

   restriction_.reset(
       GenerateProjector(uOffsets_, ufullOffsets.get(), mask.get()));

   prolongation_.reset(restriction_->Transpose());

   // need disp offsets
   std::unique_ptr<HYPRE_BigInt> dispuOffsets;
   dispuOffsets.reset(offsetsFromLocalSizes(disp_tdof_list_.Size(), MPI_COMM_WORLD));
   disp_restriction_.reset(GenerateProjector(
       dispuOffsets.get(), ufullOffsets.get(), dispmask.get()));

   disp_prolongation_.reset(disp_restriction_->Transpose());

   // remove any nonzero entries in dispBC vec that are not strictly needed
   mfem::Vector RdispBC(disp_restriction_->Height());
   disp_restriction_->Mult(uDC_, RdispBC);
   disp_prolongation_->Mult(RdispBC, uDC_);
};

void EqualityConstrainedHomotopyProblem::F(const mfem::Vector& x, const mfem::Vector& y, mfem::Vector& feval, int& Feval_err, bool /*new_pt*/) const
{
   MFEM_VERIFY(set_sizes, "need to set sizes in problem constructor");
   MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && feval.Size() == dimx,
              "F -- Inconsistent dimensions");
  feval = 0.0;
  Feval_err = 0;
};

// Q = [  r + (dc/du)^T l]
//     [ -c ]
void EqualityConstrainedHomotopyProblem::Q(const mfem::Vector& x, const mfem::Vector& y, mfem::Vector& qeval, int& Qeval_err, bool new_pt) const
{
  MFEM_VERIFY(set_sizes, "need to set sizes in problem constructor");
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && qeval.Size() == dimy,
              "Q -- Inconsistent dimensions");
  Qeval_err = 0;
  if (new_pt)
  {
     qeval = 0.0;
     try {
     mfem::BlockVector yblock(y_partition);
     yblock.Set(1.0, y);
     mfem::BlockVector qblock(y_partition);
     qblock = 0.0;

     auto u = yblock.GetBlock(0);
     auto l = yblock.GetBlock(1);
     if (has_essential_dofs)
     {
       prolongation_->Mult(u, ufull_);
       ufull_.Add(1.0, uDC_);
     }
     else
     {
       ufull_.Set(1.0, u);
     }
     auto res_vec = residual(ufull_, new_pt);
     if (has_essential_dofs)
     {
       restriction_->Mult(res_vec, qblock.GetBlock(0));
     }
     else
     {
       qblock.GetBlock(0).Set(1.0, res_vec);
     }
     auto constraint_eval = constraint(ufull_, new_pt);
     qblock.GetBlock(1).Set(-1.0, constraint_eval);
     // if constraint has also cached derivative values
     // then we let the constraint know that this is not a new point but
     // that we should determine a new derivative
     // however the problem class can control whether or not
     // it will compute a new derivative
     // if the point is not new then it can choose to not compute a new derivative
     bool new_constraint_pt = false;
     bool new_constraint_deriv = new_pt;
     auto residual_contribution = constraintJacobianTvp(ufull_, l, new_constraint_pt, new_constraint_deriv);
     if (has_essential_dofs)
     {
       restriction_->AddMult(residual_contribution, qblock.GetBlock(0));
     }
     else
     {
       qblock.GetBlock(0).Add(1.0, residual_contribution);
     }

     qeval.Set(1.0, qblock);
     q_cache.Set(1.0, qeval);
     } catch (const std::runtime_error& e)
     {
	Qeval_err = 1;
     }
  }
  else
  {
     qeval.Set(1.0, q_cache);
  }
  if (Qeval_err == 0)
  {
     Qeval_err = 0;
     int Qeval_err_loc = 0;
     for (int i = 0; i < qeval.Size(); i++) {
       if (std::isnan(qeval(i))) {
         Qeval_err_loc = 1;
         break;
       }
     }
     MPI_Allreduce(&Qeval_err_loc, &Qeval_err, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  }
};





// dQdy = [ dr/du   dc/du^T]
//        [-dc/du   0  ]
// recomputation such as HypreParMatrixFromBlocks is done
// at every DyQ call independent of value of new_pt
// recomputation may be avoided in the residualJacobian/constraintJacobian calls 
mfem::Operator* EqualityConstrainedHomotopyProblem::DyQ(const mfem::Vector& /*x*/, const mfem::Vector& y, bool new_pt)
{
  MFEM_VERIFY(set_sizes, "need to set sizes in problem constructor");
  MFEM_VERIFY(y.Size() == dimy, "InertialReliefProblem::DyQ -- Inconsistent dimensions");
  
  // note we are neglecting Hessian constraint terms
  
  mfem::BlockVector yblock(y_partition);
  yblock.Set(1.0, y);
  auto u = yblock.GetBlock(0);

  if (dQdy) {
    delete dQdy;
  }
  {
    bool new_deriv = new_pt;
    if (has_essential_dofs)
    {
      prolongation_->Mult(u, ufull_);
      ufull_.Add(1.0, uDC_);
    }
    else
    {
      ufull_.Set(1.0, u);
    }
    
    mfem::Array2D<const mfem::HypreParMatrix*> BlockMat(2, 2);
    
    mfem::HypreParMatrix * drdu = nullptr;
    if (has_essential_dofs)
    {
       auto drdufull = residualJacobian(ufull_, new_pt, new_deriv);    
       drdu = mfem::RAP(drdufull, prolongation_.get());
    }
    else
    {
       drdu = residualJacobian(ufull_, new_pt, new_deriv);
    }
    
    mfem::HypreParMatrix * dcdu = nullptr;
    if (has_essential_dofs)
    {
       auto dcdufull = constraintJacobian(ufull_, new_pt, new_deriv);
       dcdu = mfem::ParMult(dcdufull, prolongation_.get(), true);
    }
    else
    {
       dcdu = constraintJacobian(ufull_, new_pt, new_deriv);
    }
    auto dcduT = dcdu->Transpose();
    (*dcdu) *= -1.0; 
    BlockMat(0, 0) = drdu;
    BlockMat(0, 1) = dcduT;
    BlockMat(1, 0) = dcdu; 
    BlockMat(1, 1) = nullptr;
    dQdy = HypreParMatrixFromBlocks(BlockMat);
    delete dcduT;
    if (has_essential_dofs)
    {
      delete dcdu;
      delete drdu;
    }
  }
  
  return dQdy;
};

void EqualityConstrainedHomotopyProblem::SetAdjointSolver(mfem::Solver * adjoint_solver_)
{
   own_adjoint_solver = false;
   adjoint_solver = adjoint_solver_;
};


// evaluation_u_point: point at which adjoint system will be evaluated
// adjoint_load: rhs forcing term of the adjoint equation, determined by 
//               design objective, etc.
// adjoint: solution of the adjoint equation
void EqualityConstrainedHomotopyProblem::AdjointSolve(const mfem::Vector & evaluation_u_point, const mfem::Vector & adjoint_load, 
   mfem::Vector & adjoint)
{
   MFEM_VERIFY(adjoint_load.Size() == dimu_ + dimc_, "Adjoint load not of the correct size");
   MFEM_VERIFY(adjoint.Size() == dimu_ + dimc_, "Adjoint solution vector not of the correct size");
   mfem::BlockVector evaluation_y_point(y_partition); evaluation_y_point = 0.0;
   evaluation_y_point.GetBlock(0).Set(1.0, evaluation_u_point);
   mfem::Vector evaluation_x_point(dimx); evaluation_x_point = 0.0;
   auto A = DyQ(evaluation_x_point, evaluation_y_point);
   if (adjoint_is_symmetric)
   {
      adjoint_solver->SetOperator(*A);
      adjoint_solver->Mult(adjoint_load, adjoint);
   }
   else
   {
      auto Ahypre = dynamic_cast<mfem::HypreParMatrix*>(A);
      auto Aadjoint = Ahypre->Transpose();
      
      adjoint_solver->SetOperator(*Aadjoint);
      adjoint_solver->Mult(adjoint_load, adjoint);
      delete Aadjoint;
   }
};


//void EqualityConstrainedHomotopyProblem::fullDisplacement(const mfem::Vector& X, mfem::Vector& u)
//{
//  MFEM_VERIFY(X.Size() == dimu_ + dimc_, "input X not correct size");
//  MFEM_VERIFY(u.Size() == dimufull_, "input u not correct size");
//  mfem::BlockVector Xblock(y_partition);
//  Xblock.Set(1.0, X);
//  prolongation_->Mult(Xblock.GetBlock(0), u);
//  u.Add(1.0, uDC_);
//}

EqualityConstrainedHomotopyProblem::~EqualityConstrainedHomotopyProblem()
{
  if (set_sizes)
  {
     delete[] uOffsets_;
     delete[] cOffsets_;
     delete dFdx;
     delete dFdy;
     delete dQdx;
  }
  if (dQdy)
  {
     delete dQdy;
  }
  if (own_adjoint_solver)
  {
     delete adjoint_solver;
  }
};
