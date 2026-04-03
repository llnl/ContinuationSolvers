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
  mfem::HypreParMatrix * constraintJacobian = nullptr;

  // Hessian energy
  mfem::HypreParMatrix * HE = nullptr; 

  // Hessian blocks (energy)
  mfem::HypreParMatrix * HuuE = nullptr;
  mfem::HypreParMatrix * HupE = nullptr;
  mfem::HypreParMatrix * HuthE = nullptr;
  mfem::HypreParMatrix * HppE = nullptr;
  mfem::HypreParMatrix * HpthE = nullptr;
  mfem::HypreParMatrix * HththE = nullptr;
public:
  MPECProblem(ParamOptProblem * paramopt_);
  void RegularizedComplementarity(const mfem::Vector& s, const mfem::Vector& z, 
    const double& mu, mfem::Vector & phi);
  mfem::Operator * DsRegularizedComplementarity(const mfem::Vector &s, const mfem::Vector & z,
    const double &mu);
  mfem::Operator * DzRegularizedComplementarity(const mfem::Vector &s, const mfem::Vector & z,
    const double &mu);
  void SetRegularizationConst(const double & reg_const) { compl_reg_const = reg_const; };
  
  // objective in terms of u, p, and theta
  virtual double E(const mfem::Vector & u, const mfem::Vector &p, const mfem::Vector & theta, int & eval_err) = 0;
  double E(const mfem::Vector & U, int & eval_err) override;
  /* DuE, DpE, DthE assumed zero, user should implement nonzero ones */
  virtual void DuE(const mfem::Vector &u, const mfem::Vector &p, const mfem::Vector & theta, mfem::Vector& gradE);
  virtual void DpE(const mfem::Vector &u, const mfem::Vector &p, const mfem::Vector & theta, mfem::Vector& gradE);
  virtual void DthE(const mfem::Vector &u, const mfem::Vector &p, const mfem::Vector & theta, mfem::Vector& gradE);
  void DdE(const mfem::Vector &U, mfem::Vector & gradE);

  /* DuuE, DupE, DuthE, DppE, DpthE, DththE assumed zero, user should implement nonzero ones */
  virtual mfem::Operator * DuuE(const mfem::Vector &u, const mfem::Vector &p, const mfem::Vector & theta);
  virtual mfem::Operator * DupE(const mfem::Vector &u, const mfem::Vector &p, const mfem::Vector & theta);
  virtual mfem::Operator * DuthE(const mfem::Vector &u, const mfem::Vector &p, const mfem::Vector & theta);
  virtual mfem::Operator * DppE(const mfem::Vector &u, const mfem::Vector &p, const mfem::Vector & theta);
  virtual mfem::Operator * DpthE(const mfem::Vector &u, const mfem::Vector &p, const mfem::Vector & theta);
  virtual mfem::Operator * DththE(const mfem::Vector &u, const mfem::Vector &p, const mfem::Vector & theta);
  mfem::Operator * DddE(const mfem::Vector & U) override;
  void g(const mfem::Vector &U, mfem::Vector &gU, int & eval_err) override;
  mfem::Operator * Ddg(const mfem::Vector &U) override; 
  virtual ~MPECProblem();
};


class ObstacleDesignProblem : public MPECProblem
{
public:
  ObstacleDesignProblem(ParamOptProblem * paramopt_);
  double E(const mfem::Vector & U, int & eval_err) override;
  double E(const mfem::Vector & u, const mfem::Vector &p, const mfem::Vector & theta, int & eval_err) override;
  void DthE(const mfem::Vector &u, const mfem::Vector &p, const mfem::Vector & theta, mfem::Vector& gradE) override;
  mfem::Operator * DththE(const mfem::Vector &u, const mfem::Vector &p, const mfem::Vector & theta) override;
};
#endif //MPECPROBLEM_DEFS
