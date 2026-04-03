//                         Obstacle Problem
//
//
// Compile with: make ParamObstacleProblem
//
// Sample runs: mpirun -np 4 ./ParamObstacleProblem
//
//
// Description: This example code demonstrates the use of MFEM to solve the
//              bound-constrained energy minimization problem
//
//                      minimize (||∇u||² + ||u||²) subject to u ≥ ϕ in H¹.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "../continuationsolvers/problems/ObstacleProblems.hpp"
#include "../continuationsolvers/solvers/IPSolver.hpp"
#include "../continuationsolvers/problems/MPECProblems.hpp"


double dmanufacturedFun(const mfem::Vector &x);
double fRhs(const mfem::Vector &x);
double flat_obstacle(const mfem::Vector &x);




int main(int argc, char *argv[])
{
   // Initialize MPI
   mfem::Mpi::Init();
   int num_procs = mfem::Mpi::WorldSize();
   int myid = mfem::Mpi::WorldRank();
   mfem::Hypre::Init();

   int FEorder = 1; // order of the finite elements
   int maxIPMiters = 30;
   int ref_levels = 3;
   mfem::OptionsParser args(argc, argv);
   args.AddOption(&FEorder, "-o", "--order",\
	 	  "Order of the finite elements.");
   args.AddOption(&maxIPMiters, "-IPMiters", "--IPMiters",\
	 	  "Maximum number of IPM iterations");
   args.AddOption(&ref_levels, "-r", "--mesh_refinement", \
		  "Mesh Refinement");
  
   args.Parse();
   if(!args.Good())
   {
     args.PrintUsage(std::cout);
     return 1;
   }
   else
   {
     if(myid == 0)
     {  
       args.PrintOptions(std::cout);
     }
   }

   // meshing
   const char *meshFile = "meshes/inline-quad.mesh";
   mfem::Mesh mesh(meshFile, 1, 1);
   {
      for (int l = 0; l < ref_levels; l++)
      {
         mesh.UniformRefinement();
      }
   }
   mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

   // finite element-space
   int dim = mesh.Dimension(); // geometric dimension of the meshed domain
   mfem::FiniteElementCollection *fec = new mfem::H1_FECollection(FEorder, dim);
   mfem::ParFiniteElementSpace   *Vh  = new mfem::ParFiniteElementSpace(&pmesh, fec);

   // define parametrized obstacle problem
   ParamObstacleProblem problem(Vh,&fRhs,&flat_obstacle);
  
   int dimU = problem.GetDimU();
   mfem::Vector x0(dimU); x0 = 100.0;
   mfem::Vector xf(dimU); xf = 0.0;

   InteriorPointSolver optimizer(&problem); 
   optimizer.SetTol(1.e-8);
   optimizer.SetMaxIter(maxIPMiters);
   optimizer.Mult(x0, xf);

   mfem::ParGridFunction d_gf(Vh);

   d_gf.SetFromTrueDofs(xf);


   mfem::FunctionCoefficient dm_fc(dmanufacturedFun); // manufactured solution
   mfem::ParGridFunction dm_gf(Vh);
   dm_gf.ProjectCoefficient(dm_fc);
   
   // 17. Save data in the ParaView format
   mfem::ParaViewDataCollection paraview_dc("Obstacle", &pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(FEorder);
   paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("displacement", &d_gf);
   paraview_dc.RegisterField("displacement (manufactured)", &dm_gf);
   paraview_dc.Save();
   

   // define obstacle design problem
   ObstacleDesignProblem designproblem(&problem);
   int dimDesign = designproblem.GetDimU();
   mfem::Vector U0(dimDesign);
   mfem::Vector Uf(dimDesign);
   InteriorPointSolver designoptimizer(&designproblem); 
   designoptimizer.SetTol(1.e-8);
   designoptimizer.SetMaxIter(maxIPMiters);
   designoptimizer.Mult(U0, Uf);
   


   delete Vh;
   delete fec;
   return 0;
}


double dmanufacturedFun(const mfem::Vector &x)
{
  return cos(2*M_PI*x(0)) + 0.2 - 2.0*(pow(x(0),3) - 1.5*pow(x(0),2));
}

double fRhs(const mfem::Vector &x)
{
  double fx = 0.;
  fx = 0.2 - 2.0 * (pow(x(0),3)- 1.5*pow(x(0),2.) - 6 * x(0) + 3.) + (1. + pow(2.*M_PI,2))*cos(2.*M_PI*x(0));
  return fx;
}

double flat_obstacle(const mfem::Vector &x)
{
  return 0.2;
}
