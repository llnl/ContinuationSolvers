//                         Example Problem 5
//
//
// Compile with: make TestProblem5
//
// Sample runs: mpirun -np 4 ./TestProblem5
//
//
// Description: This example code demonstrates the use of the MFEM based
//              homotopy solver to solve the
//              equality-constrained optimization problem
//
//              minimize_(u \in R^n) E(u) := 1/2 u^T u 
//              subject to c(u) := \sum_i u_i^3/3 - rho = 0
//
//              where the sum over u_i^3 /3 only runs over the dofs on the root process

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "../continuationsolvers/problems/Problems.hpp"
#include "../continuationsolvers/solvers/HomotopySolver.hpp"
#include "../continuationsolvers/utilities.hpp"

using namespace std;
using namespace mfem;


// No constraints
//
// Purpose: test how the solver performs when
// there are not any second order constraint derivatives
//
//
/* NLMCP of the form
 * 0 <= x \perp F(x, y) >= 0
 *              Q(x, y)  = 0
 * with F(x, y) = { empty }
 *           x  = { empty }
 *      Q(x, y) = [ dE / du + (dc / du)^T l ]
 *                [ c(u)                    ]
 *           y = [ u ]
 *               [ l ]
 *      l a single Lagrange multiplier associated to the equality constraint c(u)
 *
 * Here we will approximate dQ / dy as
 *
 *      dQ / dy (approx) = [ d^2 E / du^2,    (dc / du)^T ]
 *                         [ dc / du     ,         0      ]
 */
class Ex5Problem : public GeneralNLMCProblem
{
protected:
   Vector a;
   double rho;
   int dimu, dimc;
   HYPRE_BigInt * uOffsets = nullptr;
   HYPRE_BigInt * cOffsets = nullptr;
   HypreParMatrix * dFdx = nullptr;
   HypreParMatrix * dFdy = nullptr;
   HypreParMatrix * dQdx = nullptr;
   HypreParMatrix * dQdy = nullptr;
   Array<int> y_partition; // y partitioned into [u, l]
public:
   Ex5Problem(int n);
   void F(const Vector &x, const Vector &y, Vector &feval, int &Feval_err, bool new_pt) const;
   void Q(const Vector &x, const Vector &y, Vector &qeval, int &Qeval_err, bool new_pt) const;
   HypreParMatrix * DxF(const Vector &x, const Vector &y, bool new_pt);
   HypreParMatrix * DyF(const Vector &x, const Vector &y, bool new_pt);
   HypreParMatrix * DxQ(const Vector &x, const Vector &y, bool new_pt);
   HypreParMatrix * DyQ(const Vector &x, const Vector &y, bool new_pt);
   virtual ~Ex5Problem();
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



   Ex5Problem problem(n);


   ////OptNLMCProblem problem(&optproblem); 
   int dimx = problem.GetDimx();
   int dimy = problem.GetDimy();

   cout << "dimx = " << dimx << endl;
   cout << "dimy = " << dimy << endl;
   Vector x0(dimx); x0 = 0.0;
   Vector y0(dimy); y0 = 0.0; y0.Randomize();
   Vector xf(dimx); xf = 0.0;
   Vector yf(dimy); yf = 0.0;
   HomotopySolver solver(&problem);
   solver.SetMaxIter(nmcpSolverMaxIter);
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


// Ex5Problem
Ex5Problem::Ex5Problem(int n) : GeneralNLMCProblem() 
{
  MFEM_VERIFY(n >= 1, "Ex5Problem::Ex5Problem -- problem must have nontrivial size");
	
  // generate parallel partition  
  int nprocs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
 
  uOffsets = new HYPRE_BigInt[2];
  cOffsets = new HYPRE_BigInt[2];

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
  
 
  { 
     HYPRE_BigInt * dofOffsets = new HYPRE_BigInt[2];
     HYPRE_BigInt * constraintOffsets = new HYPRE_BigInt[2];
     for (int i = 0; i < 2; i++)
     {
        dofOffsets[i] = uOffsets[i] + cOffsets[i];
     }
     constraintOffsets[0] = 0;
     constraintOffsets[1] = 0;
     Init(constraintOffsets, dofOffsets);
     delete[] dofOffsets;
     delete[] constraintOffsets;
  }
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
}


// TO-DO: complete me!
// Q(x, y = [ u ]) = [ u + (dc / du)^T l ]
//          [ l ]    [ c(u)              ] 
void Ex5Problem::Q(const Vector & x, const Vector & y, Vector & qeval, int &Qeval_err, bool new_pt) const
{
   qeval = 0.0;
   BlockVector yblock(y_partition); yblock.Set(1.0, y);
   BlockVector qblock(y_partition); qblock = 0.0;
   
   qblock.GetBlock(0).Set(1.0, yblock.GetBlock(0));
   // dimc > 0 only for root process
   if (dimc > 0)
   {
      double ceval = 0.0;
      // loop over dofs owned by current MPI process
      for (int i = 0; i < dimu; i++)
      {
         ceval += pow(y(i), 3.0) / 3.0;
	 qblock(i) += yblock.GetBlock(1)(0) * pow(y(i), 2.0);
      }   
      ceval -= rho;
      qblock.GetBlock(1)(0) = ceval;
   }
   qeval.Set(1.0, qblock);

   Qeval_err = 0;
}

void Ex5Problem::F(const Vector & x, const Vector & y, Vector & feval, int &Feval_err, bool new_pt) const
{
   feval = 0.0;
   Feval_err = 0;
}


HypreParMatrix * Ex5Problem::DxF(const Vector& x, const Vector& y, bool new_pt)
{
  return dFdx;
}

HypreParMatrix * Ex5Problem::DyF(const Vector& x, const Vector& y, bool new_pt)
{
  return dFdy;
}

HypreParMatrix * Ex5Problem::DxQ(const Vector& x, const Vector& y, bool new_pt)
{
  return dQdx;
}

// TODO: update here!
HypreParMatrix * Ex5Problem::DyQ(const Vector& x, const Vector& y, bool new_pt)
{
  if (dQdy)
  {
     delete dQdy;
  }
  // dQ / dy = [ I   a ]
  //           [ a^T 0 ]
  {
     // task construct I dimu x dimu
     // construct amatrix = [a] dimu x 1
     // construct transpose
     // use block matrix technology to form dQdy
     Vector one(dimu); one = 1.0;
     //for (int i = 0; i < dimy -1; i++)
     //{
     //   one += y(dimy-1) * 2. * y(i);
     //}
     HypreParMatrix * I = nullptr;
     I = GenerateHypreParMatrixFromDiagonal(uOffsets, one);
     

     // construct [a]^T
     SparseMatrix * aTmat;
     Vector entries(dimu); entries = 0.0;
     if (dimc > 0)
     {
        aTmat = new SparseMatrix(dimc, I->GetGlobalNumCols(), dimu);
        Array<int> cols;
        cols.SetSize(dimu);
        for (int i = 0; i < dimu; i++)
        {
           cols[i] = i;
	   entries(i) = pow(y(i), 2.0);
        }
        aTmat->SetRow(0, cols, entries);
     }
     else
     {
        aTmat = new SparseMatrix(dimc, I->GetGlobalNumCols(), dimc);
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

  return dQdy;
}

Ex5Problem::~Ex5Problem()
{
   delete dFdx;
   delete dFdy;
   delete dQdx;
   delete dQdy;
   delete[] uOffsets;
   delete[] cOffsets;
}


