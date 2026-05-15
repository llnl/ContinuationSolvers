//                         Obstacle Design Problem
//
//
// Compile with: make ObstacleDesignProblem
//
// Sample runs: mpirun -np 4 ./ObstacleDesignProblem
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
#include "../continuationsolvers/problems/OptProblems.hpp"
#include "../continuationsolvers/solvers/IPSolver.hpp"
#include "../continuationsolvers/solvers/MPECSolver.hpp"
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

   int maxOuterIter = 100;
   int ref_levels = 3;
   double delta = 1.e-4;
   double outerTol = 1.e-5;
   mfem::OptionsParser args(argc, argv);
   args.AddOption(&ref_levels, "-r", "--mesh_refinement", \
		  "Mesh Refinement");
   args.AddOption(&maxOuterIter, "-outerIter", "--outerIter",\
	 	  "Maximum number of nonlinear iterations");
   args.AddOption(&outerTol, "-outerTol", "--outerTol", \
		  "Nonlinear solver tolerance");
   args.AddOption(&delta, "-delta", "--delta", \
		  "Primal Hessian regularization");
   args.ParseCheck();

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
   int order = 1; // order of the finite elements
   auto fec  = std::make_unique<mfem::H1_FECollection>(order, dim);
   auto fes  = std::make_unique<mfem::ParFiniteElementSpace>(&pmesh, fec.get());
   mfem::Array<int> ess_tdof_list;
   mfem::Array<int> ess_bdr;
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh.bdr_attributes.Max());
      ess_bdr[0] = 1;
      fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   // define parametrized obstacle problem
   auto Vh = fes.get();
   int dimU = Vh->GetTrueVSize();
   int dimM = dimU - ess_tdof_list.Size();
   mfem::Vector uDC(dimU); uDC = 0.0;
   ParamObstacleProblem problem(Vh, &fRhs, &flat_obstacle, ess_tdof_list, uDC);
   ObstacleDesignProblem designproblem(&problem);
   int dimPrimal = designproblem.GetDimU();
   mfem::Vector X0(dimPrimal); X0 = 0.0;
   mfem::Vector Xf(dimPrimal); Xf = 0.0;
   mfem::Vector uf( Xf,  0, dimU);
   mfem::Vector pf( Xf, dimU, dimM);
   mfem::Vector thf(Xf, dimU + dimM, dimM);
   mfem::Vector sf( Xf, dimU + 2 * dimM, dimM);
   mfem::Vector zf( Xf, dimU + 3 * dimM, dimM);
   mfem::ParGridFunction u_gf(Vh);
   mfem::ParaViewDataCollection paraview_dc("ObstacleDesign", &pmesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.RegisterField("displacement", &u_gf);
   {
      X0 = 0.0;
      MPECSolver designoptimizer(&designproblem);
      designoptimizer.SetTol(outerTol);
      designoptimizer.SetBarrierParameter(1.e-3);
      designoptimizer.SetMaxIter(maxOuterIter);
      designoptimizer.CheckLinearSystemResiduals();
      designoptimizer.RegularizePrimalHessian(delta);
      designoptimizer.Mult(X0, Xf);
      auto mu_history = designoptimizer.GetMuHistory();
      if (!myid)
      {
         for (int i = 0; i < mu_history.Size(); i++)
         {
            std::cout << "mu_" << i << " = " << mu_history[i] << std::endl;
         }
      }
   }
   u_gf.SetFromTrueDofs(uf);
   //for (int i = 0; i < dimM; i++)
   //{
   //   std::cout << thf(i) << std::endl;
   //}
   //   p_gf.SetFromTrueDofs(pf);
   //   th_gf.SetFromTrueDofs(thf);
   //   s_gf.SetFromTrueDofs(sf);
   //   z_gf.SetFromTrueDofs(zf);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.Save();



   //// define obstacle and reduced obstacle design problems
   //ReducedParamOptProblem     rproblem(&problem, ess_tdof_list);
   //ObstacleDesignProblem designproblem(&rproblem);
   //
   //int dimPrimal = designproblem.GetDimU();
   //mfem::Vector U0(dimPrimal);
   //mfem::Vector Uf(dimPrimal);
   //
   //int dimUr = rproblem.GetDimU();
   //int dimU  =  problem.GetDimU();
   //mfem::Vector ufr( Uf,      0, dimUr);
   //mfem::Vector pfr( Uf,   dimUr, dimUr);
   //mfem::Vector thfr(Uf, 2*dimUr, dimUr);
   //mfem::Vector sfr( Uf, 3*dimUr, dimUr);
   //mfem::Vector zfr( Uf, 4*dimUr, dimUr);
   //
   //mfem::Vector uf (dimU); uf = 0.0;
   //mfem::Vector pf (dimU); pf = 0.0;
   //mfem::Vector thf(dimU); thf = 0.0;
   //mfem::Vector sf (dimU); sf = 0.0;
   //mfem::Vector zf (dimU); zf = 0.0;
   //
   //mfem::ParGridFunction u_gf(Vh);
   //mfem::ParGridFunction p_gf(Vh);
   //mfem::ParGridFunction th_gf(Vh);
   //mfem::ParGridFunction s_gf(Vh);
   //mfem::ParGridFunction z_gf(Vh);

   //mfem::ParaViewDataCollection paraview_dc("ObstacleDesign", &pmesh);
   //paraview_dc.SetPrefixPath("ParaView");
   //paraview_dc.SetLevelsOfDetail(order);
   //paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
   //paraview_dc.SetHighOrderOutput(true);
   //paraview_dc.RegisterField("displacement", &u_gf);
   //paraview_dc.RegisterField("pressure", &p_gf);
   //paraview_dc.RegisterField("design", &th_gf);
   //paraview_dc.RegisterField("slack", &s_gf);
   //paraview_dc.RegisterField("dual slack", &z_gf);
   //{
   //   U0 = 0.0;
   //   MPECSolver designoptimizer(&designproblem);
   //   designoptimizer.SetTol(outerTol);
   //   designoptimizer.SetBarrierParameter(1.e-3);
   //   designoptimizer.SetMaxIter(maxOuterIter);
   //   designoptimizer.CheckLinearSystemResiduals();
   //   designoptimizer.RegularizePrimalHessian(delta);
   //   designoptimizer.Mult(U0, Uf);
   //   auto mu_history = designoptimizer.GetMuHistory();
   //   if (!myid)
   //   {
   //      for (int i = 0; i < mu_history.Size(); i++)
   //      {
   //         std::cout << "mu_" << i << " = " << mu_history[i] << std::endl;
   //      }
   //   }
   //   rproblem.ProlongateToFullDofs(ufr, uf);
   //   rproblem.ProlongateToFullDofs(pfr, pf);
   //   rproblem.ProlongateToFullDofs(thfr, thf);
   //   rproblem.ProlongateToFullDofs(sfr, sf);
   //   rproblem.ProlongateToFullDofs(zfr, zf);
   //   u_gf.SetFromTrueDofs(uf);
   //   p_gf.SetFromTrueDofs(pf);
   //   th_gf.SetFromTrueDofs(thf);
   //   s_gf.SetFromTrueDofs(sf);
   //   z_gf.SetFromTrueDofs(zf);
   //   paraview_dc.SetCycle(0);
   //   paraview_dc.SetTime(0.0);
   //   paraview_dc.Save();
   //}
   return 0;
}


double dmanufacturedFun(const mfem::Vector &x)
{
   return std::cos(2*M_PI*x(0)) + 0.2 - 2.0*(std::pow(x(0),3) - 1.5*std::pow(x(0),2));
}

double fRhs(const mfem::Vector &x)
{
  double fx = 0.;
  fx = 0.2 - 2.0 * (std::pow(x(0),3)- 1.5*std::pow(x(0),2.) - 6 * x(0) + 3.) + (1. + std::pow(2.*M_PI,2))*std::cos(2.*M_PI*x(0));
  return fx;
}

double flat_obstacle(const mfem::Vector &x)
{
  return 0.0;
}
