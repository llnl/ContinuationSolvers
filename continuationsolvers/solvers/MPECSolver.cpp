#include "MPECSolver.hpp"



MPECSolver::MPECSolver(MPECProblem * problem_) : InteriorPointSolver(problem_)
{
}

double MPECSolver::E(const mfem::BlockVector &x, const mfem::Vector &l,
                              const mfem::Vector &zl, double mu,
                              bool printEeval) {
  double mu_min = std::max(OptTol / 10., mu);
  MPECProblem * param_opt_problem = dynamic_cast<MPECProblem*> (problem);
  MFEM_VERIFY(param_opt_problem, "cast failure");
  param_opt_problem->SetRegularizationConst(mu_min);
  return InteriorPointSolver::E(x, l, zl, mu_min, printEeval);
}

double MPECSolver::UpdateBarrierParameter(double mu) {
   return std::max(OptTol / 10., mu / 10.);
}

MPECSolver::~MPECSolver()
{
}
