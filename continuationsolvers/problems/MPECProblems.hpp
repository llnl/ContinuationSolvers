#include "mfem.hpp"
#include "../utilities.hpp"
#include "OptProblems.hpp"
#include "ObstacleProblems.hpp"


#ifndef MPECPROBLEM_DEFS
#define MPECPROBLEM_DEFS

class MPECProblem : public OptEqProblem
{
protected:
  ParamOptProblem *paramopt;
  mfem::Array<int> primal_blockoffsets; // offsets for variable U = (u, p, \theta, s, z)
  mfem::Array<int> constraint_blockoffsets; // offsets for constraints which have sizes like (u, p, s, z)

  double compl_reg_const = 1.e-2; // complementarity regularization 
  mfem::HypreParMatrix * DsPhi = nullptr;
  mfem::HypreParMatrix * DzPhi = nullptr;
public:
  MPECProblem(ParamOptProblem * paramopt_);
  void RegularizedComplementarity(const mfem::Vector& s, const mfem::Vector& z, 
    const double& mu, mfem::Vector & phi);
  mfem::Operator * DsRegularizedComplementarity(const mfem::Vector &s, const mfem::Vector & z,
    const double &mu);
  mfem::Operator * DzRegularizedComplementarity(const mfem::Vector &s, const mfem::Vector & z,
    const double &mu);
  void SetRegularizationConst(const double & reg_const) { compl_reg_const = reg_const; };
  
  //virtual double E(const mfem::Vector & u, const mfem::Vector &p, const mfem::Vector & theta, int & eval_err) = 0;
  //double E(const mfem::Vector & U, int & eval_err) override;
  // TODO: include 1st and 2nd derivative callbacks for energy
  void g(const mfem::Vector &U, mfem::Vector &gU, int & eval_err) override;
  //mfem::Operator * Ddg(const mfem::Vector &U) override;  
};



#endif //MPECPROBLEM_DEFS
