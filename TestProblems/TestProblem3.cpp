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
//              minimize_(u \in R^n) [(u^2 / 3 - 1 / 2) * (u - 3 / 4)] subject to (1/2)(uu - u) * (u - ul) ≥ 0 (component-wise).

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
class Ex3Problem : public GeneralNLMCProblem
{
protected:
   HypreParMatrix * dFdx;
   HypreParMatrix * dFdy;
   HypreParMatrix * dQdx;
   HypreParMatrix * dQdy;
   Vector ul;
   Vector uu;
   HYPRE_BigInt * dofOffsets;
public:
   Ex3Problem(int n);
   void F(const Vector &x, const Vector &y, Vector &feval, int &Feval_err, const bool new_pt) const;
   void Q(const Vector &x, const Vector &y, Vector &qeval, int &Qeval_err, const bool new_pt) const;
   HypreParMatrix * DxF(const Vector &x, const Vector &y);
   HypreParMatrix * DyF(const Vector &x, const Vector &y);
   HypreParMatrix * DxQ(const Vector &x, const Vector &y);
   HypreParMatrix * DyQ(const Vector &x, const Vector &y);
   void Displayuluu(int myid);
   virtual ~Ex3Problem();
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



   Ex3Problem problem(n);
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
   problem.Displayuluu(myid);
   Mpi::Finalize();
   return 0;
}


// Ex3Problem
Ex3Problem::Ex3Problem(int n) : GeneralNLMCProblem(), 
	dFdx(nullptr), dFdy(nullptr), dQdx(nullptr), dQdy(nullptr), dofOffsets(nullptr)
{
  MFEM_VERIFY(n >= 1, "Ex3Problem::Ex3Problem -- problem must have nontrivial size");
	
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

  // random entries in [-2, 0]
  ul.SetSize(dimx); ul = 0.0;
  ul.Randomize(myid); ul -= 1.0; ul *= 2.0;

  // random entries in [2, 3]
  uu.SetSize(dimx); uu = 0.0;
  uu.Randomize(myid); uu += 2.0;
}

void Ex3Problem::F(const Vector& x, const Vector& y, Vector& feval, int &Feval_err, const bool new_pt) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && feval.Size() == dimx, "Ex3Problem::F -- Inconsistent dimensions");
  for (int i = 0; i < dimx; i++)
  {
     feval(i) = 0.5 * (uu(i) - y(i)) * (y(i) - ul(i));
  }
  Feval_err = 0;
}

//      Q(x, y) = (y^2 / 3 - 1 / 2) + 2 y / 3 * (y - 3 / 4) - x
void Ex3Problem::Q(const Vector& x, const Vector& y, Vector& qeval, int &Qeval_err, const bool new_pt) const
{
  MFEM_VERIFY(x.Size() == dimx && y.Size() == dimy && qeval.Size() == dimx, "Ex3Problem::Q -- Inconsistent dimensions");
  for (int i = 0; i < dimx; i++)
  {
     qeval(i) = (pow(y(i), 2) / 3. - 0.5) + 2. * y(i) / 3. * (y(i) - 3. / 4.) - x(i) * ((ul(i) + uu(i)) / 2.0 - y(i));
  }
  Qeval_err = 0;
}


HypreParMatrix * Ex3Problem::DxF(const Vector& x, const Vector& y)
{
  return dFdx;
}


HypreParMatrix * Ex3Problem::DyF(const Vector& x, const Vector& y)
{
  if (dFdy != nullptr)
  {
     delete dFdy;
  }
  Vector temp(dimy); temp = 0.0;
  for (int i = 0; i < dimy; i++)
  {
     temp(i) = (ul(i) + uu(i)) / 2.0 - y(i);
  }
  dFdy = GenerateHypreParMatrixFromDiagonal(dofOffsets, temp);
  return dFdy; 
}


HypreParMatrix * Ex3Problem::DxQ(const Vector& x, const Vector& y)
{
  if (dQdx != nullptr)
  {
     delete dQdx;
  }
  Vector temp(dimx); temp = 0.0;
  for (int i = 0; i < dimx; i++)
  {
     temp(i) = (ul(i) + uu(i)) / 2.0 - y(i);
  }
  temp *= -1.0;
  dQdx = GenerateHypreParMatrixFromDiagonal(dofOffsets, temp);
  return dQdx; 
}


HypreParMatrix * Ex3Problem::DyQ(const Vector& x, const Vector& y)
{
  if (dQdy != nullptr)
  {
     delete dQdy;
  }
  Vector temp(dimx); temp = 0.0;
  temp.Set(2.0, y);
  temp -= 0.5;
  temp.Add(-1.0, x);
  dQdy = GenerateHypreParMatrixFromDiagonal(dofOffsets, temp);
  return dQdy;
}

void Ex3Problem::Displayuluu(int myid)
{
   for (int i = 0; i < dimx; i++)
   {
      cout << "ul(" << i << ") = " << ul(i) << ", uu(" << i << ") = " << uu(i) << ", (rank = " << myid << ")\n";
   }
}



Ex3Problem::~Ex3Problem()
{
   delete[] dofOffsets;
   delete dFdx;
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


