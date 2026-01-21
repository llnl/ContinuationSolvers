//                         Example Problem 4a
//
//
// Compile with: make TestProblem1a
//
// Sample runs: mpirun -np 4 ./TestProblem4a
//
//
// Description: This example code demonstrates the use of the MFEM based
//              homotopy solver to solve the
//              equality-constrained minimization problem
//
//              minimize_(u \in R^n) E(u) = 1/2 u^T u subject to a^T u - rho = 0
//              by providing the residual of the objective
//              r(u) = \nabla_u E(u) = u
//              the constraint
//              c(u) := a^T u - rho = 0
//              and various first order derivatives of r(u) and c(u) 
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "../problems/Problems.hpp"
#include "../solvers/HomotopySolver.hpp"
#include "../utilities.hpp"

using namespace std;
using namespace mfem;


// No constraints
//
/* NLMCP of the form
 * 0 <= x \perp F(x, y) >= 0
 *              Q(x, y)  = 0
 * with F(x, y) = { empty }
 *           x  = { empty }
 *      Q(x, y) = [ I   a ] [ u ] - [0]
 *                [ a^T 0 ] [ l ]   [rho]
 *           y = [ u ]
 *               [ l ]
 *      l a single Lagrange multiplier associated to the equality constraint a^T u = 0
 */
class Ex4Problem : public EqualityConstrainedHomotopyProblem
{
protected:
   Vector a;
   double rho;
   HypreParMatrix * I = nullptr;
   HypreParMatrix * J = nullptr;
public:
   Ex4Problem(int n);
   mfem::Vector residual(const mfem::Vector &u, bool new_pt) const;
   mfem::Vector constraintJacobianTvp(const mfem::Vector &u, const mfem::Vector &l, bool new_pt) const;
   mfem::HypreParMatrix * residualJacobian(const mfem::Vector & u, bool new_pt);
   mfem::Vector constraint(const mfem::Vector & u, bool new_pt) const;
   mfem::HypreParMatrix * constraintJacobian(const mfem::Vector &u, bool new_pt); 
   virtual ~Ex4Problem();
};


int main(int argc, char *argv[])
{
  // Initialize MPI
   Mpi::Init();
   int myid = Mpi::WorldRank();
   bool iAmRoot = (myid == 0);
   Hypre::Init();   
   OptionsParser args(argc, argv);


   int n = 10;
   
   real_t nmcpSolverTol = 1.e-8;
   int nmcpSolverMaxIter = 30;
   args.AddOption(&n, "-n", "--n", 
		   "Size of optimization variable.");
   args.AddOption(&nmcpSolverTol, "-nmcptol", "--nmcp-tol", 
		   "Tolerance for NMCP solver.");
   args.AddOption(&nmcpSolverMaxIter, "-nmcpmaxiter", "--nmcp-maxiter",
                  "Maximum number of iterations for the NMCP solver.");
   args.Parse();
   if (!args.Good())
   {
      if (iAmRoot)
      {
          args.PrintUsage(cout);
      }
      return 1;
   }
   if (iAmRoot)
   {
      args.PrintOptions(cout);
   }

   Ex4Problem problem(n);
   HomotopySolver solver(&problem);
   solver.SetMaxIter(nmcpSolverMaxIter);
   solver.SetPrintLevel(1);
   solver.EnableRegularizedNewtonMode(); 


   auto X0 = problem.GetOptimizationVariable(); 
   auto Xf = problem.GetOptimizationVariable();
   
   X0 = 1.0;
   Xf = 0.0;
   solver.Mult(X0, Xf);
   bool converged = solver.GetConverged();
   MFEM_VERIFY(converged, "solver did not converge\n");
   
   int dimu = problem.GetDisplacementDim();
   int dimc = problem.GetMultiplierDim();
   for (int i = 0; i < dimu; i++)
   {
      cout << "uf(" << i << ") = " << Xf.GetBlock(0)(i) << ", (rank = " << myid << ")\n";
   }
   for (int i = 0; i < dimc; i++)
   {
      cout << "lf(" << i << ") = " << Xf.GetBlock(0)(dimu+i) << ", (rank = " << myid << ")\n";
   }
   
   Mpi::Finalize();
   return 0;
}


// Ex4Problem
Ex4Problem::Ex4Problem(int n) : EqualityConstrainedHomotopyProblem() 
{
  MFEM_VERIFY(n >= 1, "Ex4Problem::Ex4Problem -- problem must have nontrivial global size");
	
  // generate parallel partition  
  int nprocs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
 
  HYPRE_BigInt * uOffsets = new HYPRE_BigInt[2];
  HYPRE_BigInt * cOffsets = new HYPRE_BigInt[2];

  if (n >= nprocs)
  {
     uOffsets[0] = HYPRE_BigInt((myid * n) / nprocs);
     uOffsets[1] = HYPRE_BigInt(((myid + 1) * n) / nprocs);
  }
  else
  {
     if (myid < n)
     {
        uOffsets[0] = myid;
        uOffsets[1] = myid + 1;
     }
     else
     {
        uOffsets[0] = n;
        uOffsets[1] = n;
     }
  }
  if (myid == 0)
  {
     cOffsets[0] = 0;
  }
  else
  {
     cOffsets[0] = 1;
  }
  cOffsets[1] = 1;

  SetSizes(uOffsets, cOffsets);
  Vector one(dimu_); one = 1.0;
  I = GenerateHypreParMatrixFromDiagonal(uOffsets, one);
  a.SetSize(dimu_); a = 10.0;
  rho = 1.e2;

  int nentries = dimuglb_;
  if (dimc_ == 0)
  {
     nentries = 0;
  }
  SparseMatrix Jmat(dimc_, dimuglb_, nentries);
  
  HypreParVector aVec(MPI_COMM_WORLD, dimuglb_, uOffsets);
  aVec.Set(1.0, a);
  Vector * aglbVec = aVec.GlobalVector();
  if (dimc_ > 0)
  {
     mfem::Array<int> cols;
     cols.SetSize(dimuglb_);
     for (int i = 0; i < dimuglb_; i++)
     {
        cols[i] = i;
     }
     Jmat.SetRow(0, cols, *aglbVec);
  }
  Jmat.Finalize();
  J = GenerateHypreParMatrixFromSparseMatrix(cOffsets, uOffsets, &Jmat);
  delete aglbVec;
}

mfem::Vector Ex4Problem::residual(const mfem::Vector & u, bool new_pt) const
{
   mfem::Vector output(dimu_); output = 0.0;
   output.Set(1.0, u);
   return output;
}

mfem::HypreParMatrix * Ex4Problem::residualJacobian(const mfem::Vector & u, bool new_pt)
{
   return I;
}

mfem::Vector Ex4Problem::constraint(const mfem::Vector & u, bool new_pt) const
{
   mfem::Vector c(dimc_); c = 0.0;
   double c_val = InnerProduct(MPI_COMM_WORLD, u, a) - rho;
   if (dimc_ > 0)
   {
      c(0) = c_val;
   }
   return c;
}

mfem::HypreParMatrix * Ex4Problem::constraintJacobian(const mfem::Vector & u, bool new_pt) 
{
   return J;
}

mfem::Vector Ex4Problem::constraintJacobianTvp(const mfem::Vector & u, const mfem::Vector &l, bool new_pt) const
{
   mfem::Vector output(dimu_); output = 0.0;
   J->MultTranspose(l, output);
   return output;
}


Ex4Problem::~Ex4Problem()
{
   delete I;
   delete J;
}


