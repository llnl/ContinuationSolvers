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
//              minimize_(u \in R^n) [(u^2 / 3 - 1 / 2) * (u - 3 / 4)] subject to u - ul ≥ 0 (component-wise).

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
 *      Q(x, y) = (y^2 / 3 - 1 / 2) + 2 y / 3 * (y - 3 / 4) - x
 * which corresponds to the first-order optimality conditions
 * of the problem described above
 *  s.t.  u - u_l >= 0
 *  where x = z (Lagrange multiplier)
 *        y = u primal variable)
 */
class Ex2Problem : public GeneralNLMCProblem
{
protected:
   HypreParMatrix * dFdx;
   HypreParMatrix * dFdy;
   HypreParMatrix * dQdx;
   HypreParMatrix * dQdy;
   Vector ul;
   HYPRE_BigInt * dofOffsets;
public:
   Ex2Problem(int n);
   void F(const Vector &x, const Vector &y, Vector &feval, int &Feval_err, const bool new_pt) const;
   void Q(const Vector &x, const Vector &y, Vector &qeval, int &Qeval_err, const bool new_pt) const;
   HypreParMatrix * DxF(const Vector &x, const Vector &y);
   HypreParMatrix * DyF(const Vector &x, const Vector &y);
   HypreParMatrix * DxQ(const Vector &x, const Vector &y);
   HypreParMatrix * DyQ(const Vector &x, const Vector &y);
   void Displayul(int myid);
   virtual ~Ex2Problem();
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



   Ex2Problem problem(n);
   int dimx = problem.GetDimx();
   int dimy = problem.GetDimy();

   cout << "dimx = " << dimx << endl;
   Vector x0(dimx); x0 = 0.0;
   Vector y0(dimy); y0 = 0.0;
   y0.Randomize();
   y0 *= 20.; 
   y0 -= 10.;
   Vector xf(dimx); xf = 0.0;
   Vector yf(dimy); yf = 0.0;
   HomotopySolver solver(&problem);
   solver.SetTol(nmcpSolverTol);
   solver.SetMaxIter(nmcpSolverMaxIter);
   solver.Mult(x0, y0, xf, yf);
   bool converged = solver.GetConverged();
   MFEM_VERIFY(converged, "solver did not converge\n");
   for (int i = 0; i < dimx; i++)
   {
      cout << "xf(" << i << ") = " << xf(i) << ", yf(" << i << ") = " << yf(i) << ", (rank = " << myid << ")\n";
   }
   problem.Displayul(myid);
   Mpi::Finalize();
   return 0;
}


// Ex2Problem
Ex2Problem::Ex2Problem(int n) : GeneralNLMCProblem(), 
	dFdx(nullptr), dFdy(nullptr), dQdx(nullptr), dQdy(nullptr), dofOffsets(nullptr)
{
  MFEM_VERIFY(n >= 1, "Ex2Problem::Ex2Problem -- problem must have nontrivial size");
	
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
  Init(dofOffsets, dofOffsets);
   
  Vector temp(dimx); 
  temp = 0.0;
  dFdx = GenerateHypreParMatrixFromDiagonal(dofOffsets, temp);

  temp = 1.0;
  dFdy = GenerateHypreParMatrixFromDiagonal(dofOffsets, temp);

  temp = -1.0;
  dQdx = GenerateHypreParMatrixFromDiagonal(dofOffsets, temp);

  // random entries in [0, 2]
  ul.SetSize(dimx); ul = 0.0;
  ul.Randomize(myid); ul *= 2.0;
}

void Ex2Problem::F(const Vector& x, const Vector& y, Vector& feval, int &Feval_err, const bool new_pt) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && feval.Size() == dimx, "Ex2Problem::F -- Inconsistent dimensions");
  feval.Set( 1.0, y);
  feval.Add(-1.0, ul);
  Feval_err = 0;
}

//      Q(x, y) = (y^2 / 3 - 1 / 2) + 2 y / 3 * (y - 3 / 4) - x
void Ex2Problem::Q(const Vector& x, const Vector& y, Vector& qeval, int &Qeval_err, const bool new_pt) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && qeval.Size() == dimx, "Ex2Problem::Q -- Inconsistent dimensions");
  for (int i = 0; i < dimx; i++)
  {
     qeval(i) = (pow(y(i), 2) / 3. - 0.5) + 2. * y(i) / 3. * (y(i) - 3. / 4.) - x(i);
  }
  Qeval_err = 0;
}


HypreParMatrix * Ex2Problem::DxF(const Vector& x, const Vector& y)
{
  return dFdx;
}


HypreParMatrix * Ex2Problem::DyF(const Vector& x, const Vector& y)
{
  return dFdy; 
}


HypreParMatrix * Ex2Problem::DxQ(const Vector& x, const Vector& y)
{
  return dQdx; 
}


HypreParMatrix * Ex2Problem::DyQ(const Vector& x, const Vector& y)
{
  Vector temp(dimx); temp = 0.0;
  temp.Set(2.0, y);
  temp -= 0.5;
  if (dQdy != nullptr)
  {
     delete dQdy;
  }
  dQdy = GenerateHypreParMatrixFromDiagonal(dofOffsets, temp);
  return dQdy;
}

void Ex2Problem::Displayul(int myid)
{
   for (int i = 0; i < dimx; i++)
   {
      cout << "ul(" << i << ") = " << ul(i) << ", (rank = " << myid << ")\n";
   }
}

Ex2Problem::~Ex2Problem()
{
   delete[] dofOffsets;
   delete dFdx;
   delete dFdy;
   delete dQdx;
   if (dQdy != nullptr)
   {
      delete dQdy;
   }
}


