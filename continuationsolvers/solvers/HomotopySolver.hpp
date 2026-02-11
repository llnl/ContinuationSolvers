#include "mfem.hpp"
#include "../problems/NLMCProblems.hpp"
#include "../utilities.hpp"

#ifndef HomotopySOLVER
#define HomotopySOLVER

class HomotopySolver
{
protected:
   // problem and sizes of local/global vectors
   GeneralNLMCProblem * problem;
   int dimx, dimy;
   HYPRE_BigInt dimxglb, dimyglb;
   HYPRE_BigInt * dofOffsetsx, * dofOffsetsy; // owned by problem
   
   double tol = 1.e-4;
   bool earlyTermination = true;


   mfem::Array<int> block_offsets_xsy;
   mfem::Array<int> block_offsets_xy;

   // pointers to various HypreParMatrix
   // solver will not own these pointers
   // memory management should be handled by 
   // problem class 
   mfem::HypreParMatrix * dFdx = nullptr;
   mfem::HypreParMatrix * dFdy = nullptr;
   mfem::HypreParMatrix * dQdx = nullptr;
   mfem::HypreParMatrix * dQdy = nullptr;

   mfem::HypreParMatrix * JGxx = nullptr;
   mfem::HypreParMatrix * JGxs = nullptr;
   mfem::HypreParMatrix * JGsx = nullptr;
   mfem::HypreParMatrix * JGss = nullptr;
   mfem::HypreParMatrix * JGsy = nullptr;
   mfem::HypreParMatrix * JGyx = nullptr;
   mfem::HypreParMatrix * JGyy = nullptr;

   mfem::BlockOperator * JGX = nullptr;
   
   mfem::Solver * linSolver = nullptr; 
   // Homotopy variable/parameters (eq 12.)
   double theta0 = 0.9;
   const double p = 1.5;
   const double q = 1.0;
   mfem::Vector gammax, gammay;
   mfem::Vector ax, bx, cx, cy;
   double max_cx_scale = std::numeric_limits<double>::infinity();
   double max_cy_scale = std::numeric_limits<double>::infinity();

   // filter
   mfem::Array<mfem::Vector *> filter;
   double gammaf = 1.e-4;

   const double delta0 = 1.0;
   double delta_MAX = 1.e5;
   const double kappa_delta = 1.e2;
   const double eta1 = 0.2;
   const double eta2 = 0.5;
   // neighborhood parameters
   const double f0 = 100.0;
   const double fbeta = 100.0;
   
   double beta0 = 1.e5;
   double beta1 = fbeta * beta0;
   
   bool useNeighborhood1 = false;
   
   const double alg_nu = 0.75;
   const double alg_rho = 1.75;


   const double epsgrad = 1.e-4;
   const double deltabnd = 1.e10;
   const double feps = 1.e6;
   bool converged;
   int max_outer_iter = 100;
   int jOpt;

   int MyRank;
   bool iAmRoot;
   int print_level = 0;

   std::ostream * hout = &std::cout;
   
   // save data
   bool save_iterates = false;
   mfem::Array<mfem::Vector *> iterates;   
   mfem::Array<int> krylov_its; 
public:
   HomotopySolver(GeneralNLMCProblem * problem_);
   void Mult(const mfem::Vector & x0, const mfem::Vector & y0, mfem::Vector & xf, mfem::Vector & yf);
   void Mult(const mfem::Vector & X0, mfem::Vector & Xf);
   bool GetConverged() const {  return converged;  };
   double E(const mfem::BlockVector & X, int & Eeval_err, bool new_pt=true);
   void G(const mfem::BlockVector & X, const double theta, mfem::BlockVector & GX, int &Geval_err, bool new_pt=true);
   void Residual(const mfem::BlockVector & X, const double theta, mfem::BlockVector & r, int &reval_err, bool new_pt=true);
   void ResidualFromG(const mfem::BlockVector & GX, const double theta, mfem::BlockVector & r);
   void PredictorResidual(const mfem::BlockVector & X, const double theta, const double thetaplus, mfem::BlockVector & r, int & reval_err, bool new_pt = true);
   void JacG(const mfem::BlockVector & X, const double theta, bool new_pt=true);
   void NewtonSolve(mfem::BlockOperator & JkOp, const mfem::BlockVector & rk, mfem::BlockVector & dXN);
   void DogLeg(const mfem::BlockOperator & JkOp, const mfem::BlockVector & gk, const double delta, const mfem::BlockVector & dXN, mfem::BlockVector & dXtr);
   bool FilterCheck(const mfem::Vector & r_comp_norm);
   void UpdateFilter(const mfem::Vector & r_comp_norm);
   void ClearFilter();
   bool NeighborhoodCheck(const mfem::BlockVector & X, const mfem::BlockVector & r, const double theta, const double beta, double & betabar_);
   bool NeighborhoodCheck_1(const mfem::BlockVector & X, const mfem::BlockVector & r, const double theta, const double beta, double & betabar_);
   bool NeighborhoodCheck_2(const mfem::BlockVector & X, const mfem::BlockVector & r, const double theta, const double beta, double & betabar_);
   void SetDeltaMax(const double delta_MAX_)
   {
      delta_MAX = delta_MAX_;
   };
   void SetNeighborhoodParameter(const double beta0_)
   {
      beta0 = beta0_;
      beta1 = fbeta * beta0;
   };
   void SetContinuationParameter(double theta0_)
   {
      theta0 = theta0_;
   };
   void EnableRegularizedNewtonMode()
   {
      max_cx_scale = 0.0;
      max_cy_scale = 0.0;
      theta0 = 1.e-2;
   };
   void SetTol(double tol_) { tol = tol_; };
   void SetMaxIter(int max_outer_iter_) { max_outer_iter = max_outer_iter_; };
   void SetEarlyTermination(bool earlyTermination_) { earlyTermination = earlyTermination_; };
   void SetOutputStream(std::ostream * hout_)
   {
      hout = hout_;
   };
   void SetLinearSolver(mfem::Solver &solver_) { linSolver = &(solver_); };
   mfem::Array<int> & GetKrylovIterations() {return krylov_its;};
   void SetPrintLevel(int print_level_) { print_level = print_level_; };
   void EnableSaveIterates()  { save_iterates = true; };
   void DisableSaveIterates() { save_iterates = false; };  
   mfem::Array<mfem::Vector *> GetIterates() {return iterates;};
   virtual ~HomotopySolver();
};


#endif
