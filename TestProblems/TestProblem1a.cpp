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
//              minimize_(x \in R^n) 1/2 x^T x subject to x - xl ≥ 0 (component-wise).

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "../continuationsolvers/problems/Problems.hpp"
#include "../continuationsolvers/solvers/HomotopySolver.hpp"
#include "../continuationsolvers/solvers/CondensedHomotopySolver.hpp"
#include "../continuationsolvers/utilities.hpp"

using namespace std;
using namespace mfem;



/* NLMCP of the form
 * 0 <= x \perp F(x, y) >= 0
 *              Q(x, y)  = 0
 * with F(x, y) = y - u_l
 *      Q(x, y) = y - x
 * which corresponds to the first-order optimality conditions
 * for the convex quadratic programming problem
 * min_u (u^T u) / 2
 *  s.t.  u - u_l >= 0
 *  where x = z (Lagrange multiplier)
 *        y = u primal variable)
 */
class Ex1aProblem : public OptProblem
{
protected:
   Vector ul;
   HypreParMatrix * dgdu = nullptr;
   HypreParMatrix * d2Edu2 = nullptr;
public:
   Ex1aProblem(int n);
   double E(const Vector & u, int & eval_err);

   void DdE(const Vector & u, Vector & gradE);

   Operator * DddE(const Vector & u);

   void g(const Vector & u, Vector & gu, int & eval_err);

   Operator * Ddg(const Vector &);
   void Displayul(int myid);

   virtual ~Ex1aProblem();
};


int main(int argc, char *argv[])
{
  // Initialize MPI
   Mpi::Init();
   int myid = Mpi::WorldRank();
   bool iAmRoot = (myid == 0);
   //Hypre::Init();   
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



   Ex1aProblem optproblem(n);
   OptNLMCProblem problem(&optproblem); 
   int dimx = problem.GetDimx();
   int dimy = problem.GetDimy();

   Vector x0(dimx); x0 = 0.0; x0.Randomize();
   Vector y0(dimy); y0 = 0.0; y0.Randomize();
   Vector xf(dimx); xf = 0.0;
   Vector yf(dimy); yf = 0.0;
   HomotopySolver solver(&problem);
   //solver.SetTol(nmcpSolverTol);
   //solver.SetMaxIter(nmcpSolverMaxIter);
   //mfem::CGSolver iterative_solver(MPI_COMM_WORLD);
   //iterative_solver.SetPrintLevel(1);
   //iterative_solver.SetMaxIter(1000);
   //iterative_solver.SetRelTol(1.e-12);

   //CondensedHomotopySolver condensed_solver;
   //condensed_solver.SetPreconditioner(iterative_solver);
   //condensed_solver.SetUseAMGF(true);
   //solver.SetLinearSolver(condensed_solver);
   solver.Mult(x0, y0, xf, yf);
   bool converged = solver.GetConverged();
   MFEM_VERIFY(converged, "solver did not converge\n");
   for (int i = 0; i < dimx; i++)
   {
      cout << "xf(" << i << ") = " << xf(i) << ", yf(" << i << ") = " << yf(i) << ", (rank = " << myid << ")\n";
   }
   optproblem.Displayul(myid);
   Mpi::Finalize();
   return 0;
}


// Ex1Problem
Ex1aProblem::Ex1aProblem(int n) : OptProblem() 
{
  MFEM_VERIFY(n >= 1, "Ex1aProblem::Ex1aProblem -- problem must have nontrivial size");
	
  // generate parallel partition  
  int nprocs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  
  HYPRE_BigInt * dofOffsets = new HYPRE_BigInt[2];
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
  Init(dofOffsets, dofOffsets);
  delete[] dofOffsets;

  Vector temp(dimU); 
  temp = 1.0;
  dgdu = GenerateHypreParMatrixFromDiagonal(dofOffsetsU, temp);

  d2Edu2 = GenerateHypreParMatrixFromDiagonal(dofOffsetsU, temp);

  // random entries in [-1, 1]
  ul.SetSize(dimU);
  ul.Randomize(myid);
  ul *= 2.0;
  ul -= 1.0;
}

double Ex1aProblem::E(const Vector & u, int & eval_err)
{
   eval_err = 0;
   double Eeval = 0.5 * InnerProduct(MPI_COMM_WORLD, u, u);
   return Eeval;
}

void Ex1aProblem::DdE(const Vector & u, Vector & gradE)
{
  gradE.Set(1.0, u);
}

Operator * Ex1aProblem::DddE(const Vector & u)
{
   return d2Edu2;
}

void Ex1aProblem::g(const Vector & u, Vector & gu, int & eval_err)
{
   eval_err = 0;
   gu.Set(1.0, u);
   gu.Add(-1.0, ul);
}

Operator * Ex1aProblem::Ddg(const Vector & u)
{
   return dgdu;
}


void Ex1aProblem::Displayul(int myid)
{
   for (int i = 0; i < dimU; i++)
   {
      cout << "ul(" << i << ") = " << ul(i) << ", (rank = " << myid << ")\n";
   }
}

Ex1aProblem::~Ex1aProblem()
{
   delete dgdu;
   delete d2Edu2;
}


