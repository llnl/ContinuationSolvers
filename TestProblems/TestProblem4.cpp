//                         Example Problem 4
//
//
// Compile with: make TestProblem4
//
// Sample runs: mpirun -np 4 ./TestProblem4
//
//
// Description: This example code demonstrates the use of the MFEM based
//              interior-point solver to solve the
//              bound-constrained minimization problem
//
//              minimize_(u \in R^n) 1/2 u^T u subject to a^T u - \rho = 0
// A simpler version of this example problem can be found in 
// TestProblem4a which utilizes the
// EqualityConstrainedHomotopyProblem a class derived specifically for
// utilizing the HomotopySolver for problems that do not have complementarity constraints
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
class Ex4Problem : public GeneralNLMCProblem
{
protected:
   Vector a;
   double rho;
   int dimu, dimc;
   HypreParMatrix * dFdx = nullptr;
   HypreParMatrix * dFdy = nullptr;
   HypreParMatrix * dQdx = nullptr;
   HypreParMatrix * dQdy = nullptr;
   mutable Vector q_cached;
   Array<int> y_partition; // y partitioned into [u, l]
public:
   Ex4Problem(int n);
   void F(const Vector &x, const Vector &y, Vector &feval, int &Feval_err, const bool new_pt) const;
   void Q(const Vector &x, const Vector &y, Vector &qeval, int &Qeval_err, const bool new_pt) const;
   HypreParMatrix * DxF(const Vector &x, const Vector &y, bool new_pt);
   HypreParMatrix * DyF(const Vector &x, const Vector &y, bool new_pt);
   HypreParMatrix * DxQ(const Vector &x, const Vector &y, bool new_pt);
   HypreParMatrix * DyQ(const Vector &x, const Vector &y, bool new_pt);
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


   int dimx = problem.GetDimx();
   int dimy = problem.GetDimy();

   Vector x0(dimx); x0 = 0.0;
   Vector y0(dimy); y0 = 0.0; y0.Randomize(); 
   y0 *= 2.0; y0 -= 1.0;
   Vector xf(dimx); xf = 0.0;
   Vector yf(dimy); yf = 0.0;
   HomotopySolver solver(&problem);
   solver.SetMaxIter(nmcpSolverMaxIter);
   solver.SetPrintLevel(1);
   solver.Mult(x0, y0, xf, yf);
   bool converged = solver.GetConverged();
   MFEM_VERIFY(converged, "solver did not converge\n");
   
   
   
   for (int i = 0; i < dimy; i++)
   {
      cout << "yf(" << i << ") = " << yf(i) << ", (rank = " << myid << ")\n";
   }
   Mpi::Finalize();
   return 0;
}


// Ex4Problem
Ex4Problem::Ex4Problem(int n) : GeneralNLMCProblem() 
{
  MFEM_VERIFY(n >= 1, "Ex4Problem::Ex4Problem -- problem must have nontrivial size");
	
  // generate parallel partition  
  int nprocs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
 
  HYPRE_BigInt * uOffsets = new HYPRE_BigInt[2];
  HYPRE_BigInt * cOffsets = new HYPRE_BigInt[2];
  HYPRE_BigInt * dofOffsets = new HYPRE_BigInt[2];
  HYPRE_BigInt * constraintOffsets = new HYPRE_BigInt[2];

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

  
  dimu = uOffsets[1] - uOffsets[0];
  dimc = cOffsets[1] - cOffsets[0];
  a.SetSize(n); a = 1.0;
  rho = 1.0;
  y_partition.SetSize(3);
  y_partition[0] = 0;
  y_partition[1] = dimu;
  y_partition[2] = dimc;
  y_partition.PartialSum();
  
  for (int i = 0; i < 2; i++)
  {
     dofOffsets[i] = uOffsets[i] + cOffsets[i];
  }
  constraintOffsets[0] = 0;
  constraintOffsets[1] = 0;
  Init(constraintOffsets, dofOffsets);

  // dF / dx 0 x 0 matrix
  {
     int nentries = 0;
     SparseMatrix * temp = new SparseMatrix(dimx, dimxglb, nentries);
     dFdx = GenerateHypreParMatrixFromSparseMatrix(dofOffsetsx, dofOffsetsx, temp);
     delete temp;
  }

  // dF / dy 0 x dimy matrix
  {
     int nentries = 0;
     SparseMatrix * temp = new SparseMatrix(dimx, dimyglb, nentries);
     dFdy = GenerateHypreParMatrixFromSparseMatrix(dofOffsetsx, dofOffsetsy, temp);
     delete temp;
  }

  // dQ / dx dimy x 0 matrix
  {
     int nentries = 0;
     SparseMatrix * temp = new SparseMatrix(dimy, dimxglb, nentries);
     dQdx = GenerateHypreParMatrixFromSparseMatrix(dofOffsetsy, dofOffsetsx, temp);
     delete temp;
  }

  // dQ / dy = [ I   a ]
  //           [ a^T 0 ]
  {
     // task construct I dimu x dimu
     // construct amatrix = [a] dimu x 1
     // construct transpose
     // use block matrix technology to form dQdy
     Vector one(dimu); one = 1.0;
     HypreParMatrix * I = nullptr;
     I = GenerateHypreParMatrixFromDiagonal(uOffsets, one);
     

     // construct [a]^T
     SparseMatrix * aTmat;
     if (dimc > 0)
     {
        aTmat = new SparseMatrix(dimc, n, n);
        Array<int> cols;
        cols.SetSize(n);
        for (int i = 0; i < n; i++)
        {
           cols[i] = i;
        }
        aTmat->SetRow(0, cols, a);
     }
     else
     {
        aTmat = new SparseMatrix(dimc, n, dimc);
     }

     HypreParMatrix * aThypre = GenerateHypreParMatrixFromSparseMatrix(cOffsets, uOffsets, aTmat);

     HypreParMatrix * ahypre = aThypre->Transpose();

     Array2D<const HypreParMatrix *> BlockMat(2, 2);
     BlockMat(0, 0) = I;
     BlockMat(0, 1) = ahypre;
     BlockMat(1, 0) = aThypre;
     BlockMat(1, 1) = nullptr;

     dQdy = HypreParMatrixFromBlocks(BlockMat);


     delete aTmat;
     delete ahypre;
     delete aThypre;
     delete I;
  }
  q_cached.SetSize(dimy);

  delete[] cOffsets;
  delete[] uOffsets;
  delete[] dofOffsets;
  delete[] constraintOffsets;
  

}


void Ex4Problem::Q(const Vector & x, const Vector & y, Vector & qeval, int &Qeval_err, bool new_pt) const
{
   if (new_pt)
   {
      qeval = 0.0;
      dQdy->Mult(y, qeval);
      if (dimc > 0)
      {
         qeval(dimu) -= rho;
      }
      q_cached.Set(1.0, qeval);
   }
   else
   {
      qeval.Set(1.0, q_cached);
   }
   Qeval_err = 0;
}

void Ex4Problem::F(const Vector & x, const Vector & y, Vector & feval, int &Feval_err, bool new_pt) const
{
   feval = 0.0;
   Feval_err = 0;
}


HypreParMatrix * Ex4Problem::DxF(const Vector& x, const Vector& y, bool new_pt)
{
  return dFdx;
}

HypreParMatrix * Ex4Problem::DyF(const Vector& x, const Vector& y, bool new_pt)
{
  return dFdy;
}

HypreParMatrix * Ex4Problem::DxQ(const Vector& x, const Vector& y, bool new_pt)
{
  return dQdx;
}

HypreParMatrix * Ex4Problem::DyQ(const Vector& x, const Vector& y, bool new_pt)
{
  return dQdy;
}

Ex4Problem::~Ex4Problem()
{
   delete dFdx;
   delete dFdy;
   delete dQdx;
   delete dQdy;
}


