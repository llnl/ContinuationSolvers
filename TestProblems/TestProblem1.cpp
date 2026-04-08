//                         Example Problem 1
//
//
// Compile with: make TestProblem1
//
// Sample runs: mpirun -np 4 ./TestProblem1
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
class Ex1Problem : public GeneralNLMCProblem
{
protected:
   HypreParMatrix * dFdx;
   HypreParMatrix * dFdy;
   HypreParMatrix * dQdx;
   HypreParMatrix * dQdy;
   Vector ul;
   HYPRE_BigInt * dofOffsets;
   HYPRE_BigInt * constraintOffsets;
   bool constraints;
public:
   Ex1Problem(int n, bool constraints_);
   void F(const Vector &x, const Vector &y, Vector &feval, int &Feval_err, bool new_pt) const;
   void Q(const Vector &x, const Vector &y, Vector &qeval, int &Qeval_err, bool new_pt) const;
   HypreParMatrix * DxF(const Vector &x, const Vector &y, bool new_pt);
   HypreParMatrix * DyF(const Vector &x, const Vector &y, bool new_pt);
   HypreParMatrix * DxQ(const Vector &x, const Vector &y, bool new_pt);
   HypreParMatrix * DyQ(const Vector &x, const Vector &y, bool new_pt);
   //void Displayul(int myid);
   void Getul(Vector & ul_);
   virtual ~Ex1Problem();
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
   bool constraints = true;
   args.AddOption(&n, "-n", "--n", 
		   "Size of optimization variable.");
   args.AddOption(&constraints, "-constraints", "--constraints", "-no-constraints",
                  "--no-constraints",
                  "Enable or disable bound constraints.");
   
   args.AddOption(&nmcpSolverTol, "-nmcptol", "--nmcp-tol", 
		   "Tolerance for NMCP solver.");
   args.AddOption(&nmcpSolverMaxIter, "-nmcpmaxiter", "--nmcp-maxiter",
                  "Maximum number of iterations for the NMCP solver.");
   args.ParseCheck();



   Ex1Problem problem(n, constraints);
   int dimx = problem.GetDimx();
   int dimy = problem.GetDimy();

   Vector x0(dimx); x0 = 0.0;
   Vector y0(dimy); y0 = 0.0;
   if (!constraints)
   {
      x0.Randomize();
      y0.Randomize();
   }
   Vector xf(dimx); xf = 0.0;
   Vector yf(dimy); yf = 0.0;
   HomotopySolver solver(&problem);
   solver.SetTol(nmcpSolverTol);
   solver.SetMaxIter(nmcpSolverMaxIter);
   solver.Mult(x0, y0, xf, yf);
   bool converged = solver.GetConverged();
   MFEM_VERIFY(converged, "solver did not converge\n");
   
   Vector ul;
   problem.Getul(ul);
   
   for (int i = 0; i < dimy; i++)
   {
      cout << "yf(" << i << ") = " << yf(i) << ", (rank = " << myid << ")\n";
      if (constraints)
      {
         cout << "xf(" << i << ") = " << xf(i) << ", (rank = " << myid << ")\n";
         cout << "ul(" << i << ") = " << ul(i) << ", (rank = " << myid << ")\n";
      }
      cout << "\n";
   }
   Mpi::Finalize();
   return 0;
}


// Ex1Problem
Ex1Problem::Ex1Problem(int n, bool constraints_) : GeneralNLMCProblem(), 
	dFdx(nullptr), dFdy(nullptr), dQdx(nullptr), dQdy(nullptr), dofOffsets(nullptr), constraints(constraints_)
{
  MFEM_VERIFY(n >= 1, "Ex1Problem::Ex1Problem -- problem must have nontrivial size");
	
  // generate parallel partition  
  int nprocs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  
  dofOffsets = new HYPRE_BigInt[2];
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
 
  constraintOffsets = new HYPRE_BigInt[2];
  if (constraints)
  {
     constraintOffsets[0] = dofOffsets[0];
     constraintOffsets[1] = dofOffsets[1];
  }
  else
  {
     constraintOffsets[0] = 0;
     constraintOffsets[1] = 0;
  }
   
  
  Init(constraintOffsets, dofOffsets);
  
  Vector temp(dimy); 
  temp = 0.0;
  {
     Vector tempx(dimx); tempx = 0.;
     dFdx = GenerateHypreParMatrixFromDiagonal(constraintOffsets, tempx);
  }
  
  temp = 1.0;
  dQdy = GenerateHypreParMatrixFromDiagonal(dofOffsets, temp);
  if (constraints)
  {
     dFdy = GenerateHypreParMatrixFromDiagonal(dofOffsets, temp);
  }
  else
  {
     dFdy = GenerateNullHypreParMatrix(dofOffsetsx, dofOffsetsy);
  }

  temp = -1.0;
  if  (constraints)
  {
     dQdx = GenerateHypreParMatrixFromDiagonal(dofOffsets, temp);
  }
  else
  {
     dQdx = GenerateNullHypreParMatrix(dofOffsetsy, dofOffsetsx);
  }
  // random entries in [-1, 1]
  ul.SetSize(dimx); ul = 0.0;
  if (constraints)
  {
     ul.Randomize(myid);
     ul *= 2.0;
     ul -= 1.0;
  }
}

void Ex1Problem::F(const Vector& x, const Vector& y, Vector& feval, int &Feval_err, bool new_pt) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && feval.Size() == dimx, "Ex1Problem::F -- Inconsistent dimensions");
  if (constraints)
  {
     feval.Set( 1.0, y);
     feval.Add(-1.0, ul);
  }
  Feval_err = 0;
}


void Ex1Problem::Q(const Vector& x, const Vector& y, Vector& qeval, int &Qeval_err, bool new_pt) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && qeval.Size() == dimy, "Ex1Problem::Q -- Inconsistent dimensions");
  qeval.Set( 1.0, y);
  if (constraints)
  {
     qeval.Add(-1.0, x);
  }
  Qeval_err = 0;
}


HypreParMatrix * Ex1Problem::DxF(const Vector& x, const Vector& y, bool new_pt)
{
  return dFdx;
}


HypreParMatrix * Ex1Problem::DyF(const Vector& x, const Vector& y, bool new_pt)
{
  return dFdy; 
}


HypreParMatrix * Ex1Problem::DxQ(const Vector& x, const Vector& y, bool new_pt)
{
  return dQdx; 
}


HypreParMatrix * Ex1Problem::DyQ(const Vector& x, const Vector& y, bool new_pt)
{
  return dQdy; 
}

void Ex1Problem::Getul(Vector & ul_)
{
   ul_.SetSize(ul.Size());
   ul_.Set(1.0, ul);
}

//void Ex1Problem::Displayul(int myid)
//{
//   for (int i = 0; i < dimx; i++)
//   {
//      cout << "ul(" << i << ") = " << ul(i) << ", (rank = " << myid << ")\n";
//   }
//}

Ex1Problem::~Ex1Problem()
{
   delete[] dofOffsets;
   delete[] constraintOffsets;
   delete dFdx;
   delete dFdy;
   delete dQdx;
   delete dQdy;
}


