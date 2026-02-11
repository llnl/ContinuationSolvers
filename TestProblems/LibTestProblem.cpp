#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "../continuationsolvers/problems/Problems.hpp"
#include "../continuationsolvers/solvers/HomotopySolver.hpp"
#include "../continuationsolvers/utilities.hpp"
#include "LibTestProblem.hpp"


using namespace std;
using namespace mfem;


class DiabloProblem : public GeneralNLMCProblem
{
protected:
   HypreParMatrix * dFdx;
   HypreParMatrix * dFdy;
   HypreParMatrix * dQdx;
   HypreParMatrix * dQdy;

   void * ctx;
public:
   DiabloProblem(void * ctx_);
   virtual ~DiabloProblem();
   void F(const Vector &x, const Vector &y, Vector &feval, int &Feval_err) const;
   void Q(const Vector &x, const Vector &y, Vector &qeval, int &Qeval_err) const;
   HypreParMatrix * DxF(const Vector &x, const Vector &y);
   HypreParMatrix * DyF(const Vector &x, const Vector &y);
   HypreParMatrix * DxQ(const Vector &x, const Vector &y);
   HypreParMatrix * DyQ(const Vector &x, const Vector &y);
};


void mfemIPSolve(GeneralNLMCProblem & problem, Vector & x, Vector &y)
{
   HomotopySolver solver(&problem);
   int dimx = problem.GetDimx();
   int dimy = problem.GetDimy();
   Vector x0(dimx); x0.Randomize();
   x.SetSize(dimx); x = 0.0;

   Vector y0(dimy); y0.Randomize();
   y.SetSize(dimy); y = 0.0;

   solver.Mult(x0, y0, x, y);
   



   
}

int main(int argc, char *argv[])
{
  Mpi::Init();
  Hypre::Init();

  int myrank = Mpi::WorldRank();
  // initialize the diablo problem
  void * ctx = diablo_Init();

  //// create a OptProblem which uses the diablo problem functionality
  DiabloProblem problem(ctx);

  Vector xOptimal, yOptimal;
  

  //// solve the diablo defined optimization problem by the IPM
  mfemIPSolve(problem, xOptimal, yOptimal);

  for (int i = 0; i < xOptimal.Size(); i++)
  {
     cout << "xOptimal(" << i << ") = " << xOptimal(i) << ", (rank = " << myrank << ")\n";
  }
  
  for (int i = 0; i < yOptimal.Size(); i++)
  {
     cout << "yOptimal(" << i << ") = " << yOptimal(i) << ", (rank = " << myrank << ")\n";
  }

  diablo_PrintProblemInfo(ctx);
  
  // free memory
  diablo_Finalize(ctx);
  Mpi::Finalize();
  return 0;
}

DiabloProblem::DiabloProblem(void * ctx_)
{
   ctx = ctx_;
   dFdx = nullptr;
   dFdy = nullptr;
   dQdx = nullptr;
   dQdy = nullptr;
  
   HYPRE_BigInt * dofOffsetsx_ = nullptr;
   HYPRE_BigInt * dofOffsetsy_ = nullptr;

   dofOffsetsx_ = diablo_GetDofOffsetsx(ctx);
   dofOffsetsy_ = diablo_GetDofOffsetsy(ctx);

   HypreToMfemOffsets(dofOffsetsx_);
   HypreToMfemOffsets(dofOffsetsy_);
   
   // initialize the problem, passing parallel partition information
   Init(dofOffsetsx_, dofOffsetsy_);
   
   free(dofOffsetsx_);
   free(dofOffsetsy_);
}

DiabloProblem::~DiabloProblem()
{
  if (dFdx != nullptr)
  {
    delete dFdx;
  }
  if (dFdy != nullptr)
  {
     delete dFdy;
  }
  if (dQdx != nullptr)
  {
     delete dQdx;
  }
  if (dQdy != nullptr)
  {
     delete dQdy;
  }
}



void DiabloProblem::F(const Vector& x, const Vector& y, Vector& feval, int &Feval_err) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && feval.Size() == dimx, "DiabloProblem::F -- Inconsistent dimensions");
  HypreParVector feval_par(MPI_COMM_WORLD, dimxglb, dofOffsetsx);
  HypreParVector x_par(MPI_COMM_WORLD, dimxglb, dofOffsetsx);
  HypreParVector y_par(MPI_COMM_WORLD, dimyglb, dofOffsetsy);
  x_par.Set(1.0, x);
  y_par.Set(1.0, y);

  HYPRE_ParVector par_feval = (HYPRE_ParVector) feval_par;
  HYPRE_ParVector par_x = (HYPRE_ParVector) x_par;
  HYPRE_ParVector par_y = (HYPRE_ParVector) y_par;

  diablo_F(ctx, &par_x, &par_y, &par_feval);
  HypreParVector feval_par1 = HypreParVector(par_feval);
  feval.Set(1.0, feval_par1);
  Feval_err = 0; // TODO check that there were no evaluation errors
}


void DiabloProblem::Q(const Vector& x, const Vector& y, Vector& qeval, int & Qeval_err) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && qeval.Size() == dimy, "DiabloProblem::Q -- Inconsistent dimensions");
  HypreParVector qeval_par(MPI_COMM_WORLD, dimyglb, dofOffsetsy);
  HypreParVector x_par(MPI_COMM_WORLD, dimxglb, dofOffsetsx);
  HypreParVector y_par(MPI_COMM_WORLD, dimyglb, dofOffsetsy);
  x_par.Set(1.0, x);
  y_par.Set(1.0, y);

  HYPRE_ParVector par_qeval = (HYPRE_ParVector) qeval_par;
  HYPRE_ParVector par_x = (HYPRE_ParVector) x_par;
  HYPRE_ParVector par_y = (HYPRE_ParVector) y_par;

  diablo_Q(ctx, &par_x, &par_y, &par_qeval);
  HypreParVector qeval_par1 = HypreParVector(par_qeval);
  qeval.Set(1.0, qeval_par1);
  Qeval_err = 0; // TODO: check that there were no evaluation errors
}


HypreParMatrix * DiabloProblem::DxF(const Vector& x, const Vector& y)
{
  if (dFdx != nullptr)
  {
     delete dFdx;
  }

  HYPRE_ParCSRMatrix dFdx_csr;
  
  HypreParVector x_par(MPI_COMM_WORLD, dimxglb, dofOffsetsx);
  HypreParVector y_par(MPI_COMM_WORLD, dimyglb, dofOffsetsy);
  x_par.Set(1.0, x);
  y_par.Set(1.0, y);
  
  HYPRE_ParVector par_x = (HYPRE_ParVector) x_par;
  HYPRE_ParVector par_y = (HYPRE_ParVector) y_par;
  
  diablo_DxF(ctx, &par_x, &par_y, &dFdx_csr);

  dFdx = new HypreParMatrix(dFdx_csr, false);
  return dFdx;
}

HypreParMatrix * DiabloProblem::DyF(const Vector& x, const Vector& y)
{
  if (dFdy != nullptr)
  {
     delete dFdy;
  }

  HYPRE_ParCSRMatrix dFdy_csr;
  
  HypreParVector x_par(MPI_COMM_WORLD, dimxglb, dofOffsetsx);
  HypreParVector y_par(MPI_COMM_WORLD, dimyglb, dofOffsetsy);
  x_par.Set(1.0, x);
  y_par.Set(1.0, y);
  
  HYPRE_ParVector par_x = (HYPRE_ParVector) x_par;
  HYPRE_ParVector par_y = (HYPRE_ParVector) y_par;
  
  diablo_DyF(ctx, &par_x, &par_y, &dFdy_csr);

  dFdy = new HypreParMatrix(dFdy_csr, false);
  return dFdy;
}

HypreParMatrix * DiabloProblem::DxQ(const Vector& x, const Vector& y)
{
  if (dQdx != nullptr)
  {
     delete dQdx;
  }

  HYPRE_ParCSRMatrix dQdx_csr;
  
  HypreParVector x_par(MPI_COMM_WORLD, dimxglb, dofOffsetsx);
  HypreParVector y_par(MPI_COMM_WORLD, dimyglb, dofOffsetsy);
  x_par.Set(1.0, x);
  y_par.Set(1.0, y);
  
  HYPRE_ParVector par_x = (HYPRE_ParVector) x_par;
  HYPRE_ParVector par_y = (HYPRE_ParVector) y_par;
  
  diablo_DxQ(ctx, &par_x, &par_y, &dQdx_csr);

  dQdx = new HypreParMatrix(dQdx_csr, false);
  return dQdx;
}

HypreParMatrix * DiabloProblem::DyQ(const Vector& x, const Vector& y)
{
  if (dQdy != nullptr)
  {
     delete dQdy;
  }

  HYPRE_ParCSRMatrix dQdy_csr;
  
  HypreParVector x_par(MPI_COMM_WORLD, dimxglb, dofOffsetsx);
  HypreParVector y_par(MPI_COMM_WORLD, dimyglb, dofOffsetsy);
  x_par.Set(1.0, x);
  y_par.Set(1.0, y);
  
  HYPRE_ParVector par_x = (HYPRE_ParVector) x_par;
  HYPRE_ParVector par_y = (HYPRE_ParVector) y_par;
  
  diablo_DyQ(ctx, &par_x, &par_y, &dQdy_csr);

  dQdy = new HypreParMatrix(dQdy_csr, false);
  return dQdy;
}
