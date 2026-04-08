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
//              minimize_(x \in R^n) 1/2 x^T x

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "../continuationsolvers/problems/Problems.hpp"
#include "../continuationsolvers/solvers/IPSolver.hpp"
#include "../continuationsolvers/utilities.hpp"

using namespace std;
using namespace mfem;


class Ex1bProblem : public OptProblem
{
protected:
   HypreParMatrix * dgdu;
   HypreParMatrix * d2Edu2;
public:
   Ex1bProblem(int n);
   double E(const Vector & u, int & eval_err);

   void DdE(const Vector & u, Vector & gradE);

   Operator * DddE(const Vector & u);

   void g(const Vector & u, Vector & gu, int & eval_err);

   Operator * Ddg(const Vector &);

   virtual ~Ex1bProblem();
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



   Ex1bProblem problem(n);
   InteriorPointSolver solver(&problem);
   int dimx = problem.GetDimU();
   int dimm = problem.GetDimM();
   Vector x0(dimx); x0 = 0.0;
   x0.Randomize();
   Vector xf(dimx); xf = 0.0;
   solver.SetTol(nmcpSolverTol);
   solver.Mult(x0, xf);


   bool converged = solver.GetConverged();
   MFEM_VERIFY(converged, "solver did not converge\n");
   for (int i = 0; i < dimx; i++)
   {
      cout << "xf(" << i << ") = " << xf(i) << ", (rank = " << myid << ")\n";
   }
   Mpi::Finalize();
   return 0;
}


// Ex1Problem
Ex1bProblem::Ex1bProblem(int n) : OptProblem(), 
	dgdu(nullptr), d2Edu2(nullptr)
{
  MFEM_VERIFY(n >= 1, "Ex1bProblem::Ex1bProblem -- problem must have nontrivial size");
	
  // generate parallel partition  
  int nprocs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  
  HYPRE_BigInt * dofOffsets = new HYPRE_BigInt[2];
  HYPRE_BigInt * constraintOffsets = new HYPRE_BigInt[2];
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
  constraintOffsets[0] = 0;
  constraintOffsets[1] = 0;
  Init(dofOffsets, constraintOffsets);
  delete[] dofOffsets;
  delete[] constraintOffsets;

  {
     Vector temp(dimU); temp = 1.0;
     d2Edu2 = GenerateHypreParMatrixFromDiagonal(dofOffsetsU, temp);
  }
  
  dgdu = GenerateNullHypreParMatrix(dofOffsetsM, dofOffsetsU);
}

double Ex1bProblem::E(const Vector & u, int & eval_err)
{
   eval_err = 0;
   double Eeval = 0.5 * InnerProduct(MPI_COMM_WORLD, u, u);
   return Eeval;
}

void Ex1bProblem::DdE(const Vector & u, Vector & gradE)
{
  gradE.Set(1.0, u);
}

Operator * Ex1bProblem::DddE(const Vector & u)
{
   return d2Edu2;
}

void Ex1bProblem::g(const Vector & u, Vector & gu, int & eval_err)
{
   eval_err = 0;
   gu = 0.0;
}

Operator * Ex1bProblem::Ddg(const Vector & u)
{
   return dgdu;
}


Ex1bProblem::~Ex1bProblem()
{
   delete dgdu;
   delete d2Edu2;
}


