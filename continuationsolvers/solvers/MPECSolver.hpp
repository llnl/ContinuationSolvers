#include "IPSolver.hpp"
#include "../problems/MPECProblems.hpp"

#ifndef MPECSOLVER
#define MPECSOLVER

class MPECSolver : public InteriorPointSolver {
protected:
  MPECProblem * paramoptproblem;
  double target_mu = 1.e-6;
public:
  MPECSolver(MPECProblem *);
  double E(const mfem::BlockVector &, const mfem::Vector &,
           const mfem::Vector &, double, bool) override;
  double UpdateBarrierParameter(double) override;
  virtual ~MPECSolver();
};

#endif // MPECSOLVER
