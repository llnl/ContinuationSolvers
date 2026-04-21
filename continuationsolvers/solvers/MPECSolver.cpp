#include "MPECSolver.hpp"



MPECSolver::MPECSolver(MPECProblem * problem_) : InteriorPointSolver(problem_)
{
}

double MPECSolver::E(const mfem::BlockVector &x, const mfem::Vector &l,
                              const mfem::Vector &zl, double mu,
                              bool printEeval) {
  MPECProblem * param_opt_problem = dynamic_cast<MPECProblem*> (problem);
  MFEM_VERIFY(param_opt_problem, "cast failure");
  double mu_used = std::max(target_mu, mu);
  param_opt_problem->SetRegularizationConst(mu_used);
  return InteriorPointSolver::E(x, l, zl, mu_used, printEeval);
}

double MPECSolver::UpdateBarrierParameter(double mu) {
   MPECProblem * param_opt_problem = dynamic_cast<MPECProblem*> (problem);
   MFEM_VERIFY(param_opt_problem, "cast failure");
   double mu_new = std::max(target_mu, mu / 10.);
   param_opt_problem->SetRegularizationConst(mu_new);
   return mu_new;
}

MPECSolver::~MPECSolver()
{
}
