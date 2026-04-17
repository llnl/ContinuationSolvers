#include "IPSolver.hpp"
#include "../problems/MPECProblems.hpp"

#ifndef MPECSOLVER
#define MPECSOLVER

class MPECSolver : public InteriorPointSolver {
protected:
  MPECProblem * paramoptproblem; 
public:
  MPECSolver(MPECProblem *);
  double E(const mfem::BlockVector &, const mfem::Vector &,
           const mfem::Vector &, double, bool) override;
  virtual ~MPECSolver();
};

#endif // MPECSOLVER
