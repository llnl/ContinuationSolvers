//                         Example Problem 1a
//
//
// Compile with: make TestProblem1a
//
// Sample runs: mpirun -np 4 ./TestProblem1a
//
//
// Description: This example code demonstrates the use of the MFEM based
//              interior-point solver to solve the
//              bound-constrained minimization problem
//
//              minimize_(u \in R^n) {\sum_i u_i^4 / 12. + u_i^2 / 2.} subject to u_1 - ul = 0.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "../continuationsolvers/problems/Problems.hpp"
#include "../continuationsolvers/solvers/IPSolver.hpp"
#include "../continuationsolvers/utilities.hpp"

using namespace std;
using namespace mfem;


class Ex1dProblem : public OptEqProblem
{
protected:
   Vector ul;
   HypreParMatrix * dgdu = nullptr;
   HypreParMatrix * d2Edu2 = nullptr;
public:
   Ex1dProblem(int n);
   double E(const Vector & u, int & eval_err);

   void DdE(const Vector & u, Vector & gradE);

   HypreParMatrix * DddE(const Vector & u);

   void g(const Vector & u, Vector & gu, int & eval_err);

   HypreParMatrix * Ddg(const Vector &);

   virtual ~Ex1dProblem();
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
   bool condensed_solve = false;
   bool use_AMGF = false;
   args.AddOption(&n, "-n", "--n", 
		   "Size of optimization variable.");
   args.AddOption(&nmcpSolverTol, "-nmcptol", "--nmcp-tol", 
		   "Tolerance for NMCP solver.");
   args.AddOption(&nmcpSolverMaxIter, "-nmcpmaxiter", "--nmcp-maxiter",
                  "Maximum number of iterations for the NMCP solver.");
   args.AddOption(&condensed_solve, "-condensed-solve", "--condensed-solve", "-monolithic-solve",
                  "--monolithic-solve", "Whether or not to use the CondensedHomotopySolver.");
   args.AddOption(&use_AMGF, "-AMGF", "--use-AMGF", "-no-AMGF", "--not-use-AMGF",
                  "Whether or not to use AMGF for the reduced system in CondensedHomotopySolver.");
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


   Ex1dProblem problem(n);
   

   int dimU = problem.GetDimU();
   Vector x0(dimU); x0 = 100.0;
   Vector xf(dimU); xf = 0.0;
   InteriorPointSolver optimizer(&problem); 
   optimizer.SetTol(1.e-8);
   optimizer.SetMaxIter(30);
   optimizer.Mult(x0, xf);
   for (int i = 0; i < dimU; i++)
   {
      std::cout << "uf(" << i << ") = " << xf(i) << std::endl;
   } 
   Mpi::Finalize();
   return 0;
}


// Ex1Problem
Ex1dProblem::Ex1dProblem(int n) : OptEqProblem() 
{
  MFEM_VERIFY(n >= 1, "Ex1dProblem::Ex1dProblem -- problem must have nontrivial size");
	
  // generate parallel partition  
  int nprocs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  
  HYPRE_BigInt * dofOffsets = new HYPRE_BigInt[2];
  HYPRE_BigInt * cOffsets   = new HYPRE_BigInt[2];
  if (n >= nprocs)
  {
     dofOffsets[0] = HYPRE_BigInt((myid * n) / nprocs);
     dofOffsets[1] = HYPRE_BigInt(((myid + 1) * n) / nprocs);
  }
  else
  {
     if (myid < n)
     {
        dofOffsets[0] = myid;
        dofOffsets[1] = myid + 1;
     }
     else
     {
        dofOffsets[0] = n;
	dofOffsets[1] = n;
     }
  }
  cOffsets[0] = 1;
  cOffsets[1] = 1;
  if (myid == 0)
  {
    cOffsets[0] = 0;
  }
  Init(dofOffsets, cOffsets);
  delete[] dofOffsets;
  delete[] cOffsets;
  Vector temp(dimU); 
  temp = 1.0;
  
  SparseMatrix * dgdumat;
  dgdumat = new SparseMatrix(dimC, n, dimC);
  if (myid == 0)
  {
     Array<int> cols;
     cols.SetSize(1);
     cols[0] = 0;
     Vector entries(1); entries = 1.0;
     dgdumat->SetRow(0, cols, entries);
  }
  dgdu = GenerateHypreParMatrixFromSparseMatrix(dofOffsetsC, dofOffsetsU, dgdumat);
  delete dgdumat;

  d2Edu2 = GenerateHypreParMatrixFromDiagonal(dofOffsetsU, temp);

  ul.SetSize(dimC);
  if (myid == 0)
  {
    ul = 1.0;
  }
}

double Ex1dProblem::E(const Vector & u, int & eval_err)
{
   eval_err = 0;
   Vector usqr(dimU);
   usqr.Set(1.0, u);
   usqr *= u;
   double Eeval = InnerProduct(MPI_COMM_WORLD, usqr, usqr) / 12. + InnerProduct(MPI_COMM_WORLD, u, u) / 2.;
   return Eeval;
}

void Ex1dProblem::DdE(const Vector & u, Vector & gradE)
{
  gradE.Set(1.0, u);
  gradE *= u;
  gradE *= u;
  gradE /= 3.;
  gradE.Add(1.0, u);
}

HypreParMatrix * Ex1dProblem::DddE(const Vector & u)
{
  Vector temp(dimU);
  temp.Set(1.0, u);
  temp *= u; 
  temp += 1.0;
  if (!d2Edu2)
  {
    delete d2Edu2;
  }
  d2Edu2 = GenerateHypreParMatrixFromDiagonal(dofOffsetsU, temp);
  return d2Edu2;
}

void Ex1dProblem::g(const Vector & u, Vector & gu, int & eval_err)
{
   eval_err = 0;
   dgdu->Mult(u, gu);
   gu.Add(-1.0, ul);
}

HypreParMatrix * Ex1dProblem::Ddg(const Vector & u)
{
   return dgdu;
}

Ex1dProblem::~Ex1dProblem()
{
   delete dgdu;
   delete d2Edu2;
}


