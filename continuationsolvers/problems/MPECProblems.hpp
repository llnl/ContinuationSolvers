#include "../utilities.hpp"
#include "ObstacleProblems.hpp"
#include "OptProblems.hpp"
#include "mfem.hpp"

#ifndef MPECPROBLEM_DEFS
#define MPECPROBLEM_DEFS

class MPECProblem : public OptEqProblem {
protected:
  ParamOptProblem *paramopt;
  mfem::Array<int>
      primal_blockoffsets; // offsets for variable U = (u, p, \theta, s, z)
  mfem::Array<int> constraint_blockoffsets; // offsets for constraints which
                                            // have sizes like (u, p, s, z)

  mfem::Vector slackScale;
  double compl_reg_const = 1.e-2; // complementarity regularization
  std::unique_ptr<mfem::HypreParMatrix> DsPhi;
  std::unique_ptr<mfem::HypreParMatrix> DzPhi;
  std::unique_ptr<mfem::HypreParMatrix> DsslPhi;
  std::unique_ptr<mfem::HypreParMatrix> DszlPhi;
  std::unique_ptr<mfem::HypreParMatrix> DzzlPhi;
  std::unique_ptr<mfem::HypreParMatrix> constraintJacobian;
  std::unique_ptr<mfem::HypreParMatrix> constraintHessian;

  // Hessian energy
  std::unique_ptr<mfem::HypreParMatrix> HE;

  // Hessian blocks (energy)
  std::unique_ptr<mfem::HypreParMatrix> HuuE;
  std::unique_ptr<mfem::HypreParMatrix> HupE;
  std::unique_ptr<mfem::HypreParMatrix> HuthE;
  std::unique_ptr<mfem::HypreParMatrix> HppE;
  std::unique_ptr<mfem::HypreParMatrix> HpthE;
  std::unique_ptr<mfem::HypreParMatrix> HththE;

  // Jacobian blocks
  std::unique_ptr<mfem::HypreParMatrix> dg1ds; // d/ds(g(u, theta) - s) = -I
  std::unique_ptr<mfem::HypreParMatrix> dg2dp; // d/dp(p - z) = I
  std::unique_ptr<mfem::HypreParMatrix> dg2dz; // d/dz(p - z) = -I
public:
  MPECProblem(ParamOptProblem *paramopt_);
  void RegularizedComplementarity(const mfem::Vector &s, const mfem::Vector &z,
                                  const double &mu, mfem::Vector &phi);
  mfem::Operator *DsRegularizedComplementarity(const mfem::Vector &s,
                                               const mfem::Vector &z,
                                               const double &mu);
  mfem::Operator *DzRegularizedComplementarity(const mfem::Vector &s,
                                               const mfem::Vector &z,
                                               const double &mu);
  mfem::Operator *DsslRegularizedComplementarity(const mfem::Vector &s,
                                                 const mfem::Vector &z,
                                                 const mfem::Vector &l,
                                                 const double &mu);
  mfem::Operator *DszlRegularizedComplementarity(const mfem::Vector &s,
                                                 const mfem::Vector &z,
                                                 const mfem::Vector &l,
                                                 const double &mu);
  mfem::Operator *DzzlRegularizedComplementarity(const mfem::Vector &s,
                                                 const mfem::Vector &z,
                                                 const mfem::Vector &l,
                                                 const double &mu);
  void SetRegularizationConst(const double &reg_const) {
    compl_reg_const = reg_const;
  };

  // objective in terms of u, p, and theta
  virtual double E(const mfem::Vector &u, const mfem::Vector &p,
                   const mfem::Vector &theta, int &eval_err) = 0;
  double E(const mfem::Vector &U, int &eval_err) override;
  /* DuE, DpE, DthE assumed zero, user should implement nonzero ones */
  virtual void DuE(const mfem::Vector &u, const mfem::Vector &p,
                   const mfem::Vector &theta, mfem::Vector &gradE);
  virtual void DpE(const mfem::Vector &u, const mfem::Vector &p,
                   const mfem::Vector &theta, mfem::Vector &gradE);
  virtual void DthE(const mfem::Vector &u, const mfem::Vector &p,
                    const mfem::Vector &theta, mfem::Vector &gradE);
  void DdE(const mfem::Vector &U, mfem::Vector &gradE) override;

  /* DuuE, DupE, DuthE, DppE, DpthE, DththE assumed zero, user should implement
   * nonzero ones */
  virtual mfem::Operator *DuuE(const mfem::Vector &u, const mfem::Vector &p,
                               const mfem::Vector &theta);
  virtual mfem::Operator *DupE(const mfem::Vector &u, const mfem::Vector &p,
                               const mfem::Vector &theta);
  virtual mfem::Operator *DuthE(const mfem::Vector &u, const mfem::Vector &p,
                                const mfem::Vector &theta);
  virtual mfem::Operator *DppE(const mfem::Vector &u, const mfem::Vector &p,
                               const mfem::Vector &theta);
  virtual mfem::Operator *DpthE(const mfem::Vector &u, const mfem::Vector &p,
                                const mfem::Vector &theta);
  virtual mfem::Operator *DththE(const mfem::Vector &u, const mfem::Vector &p,
                                 const mfem::Vector &theta);
  mfem::Operator *DddE(const mfem::Vector &U) override;
  void g(const mfem::Vector &U, mfem::Vector &gU, int &eval_err) override;
  mfem::Operator *Ddg(const mfem::Vector &U) override;
  mfem::Operator *Dddgl(const mfem::Vector &U, const mfem::Vector &l) override;
  void SetSlackScale(const mfem::Vector sScale) {
    slackScale.Set(1.0, sScale);
  };
  virtual ~MPECProblem();
};

class ObstacleDesignProblem : public MPECProblem {
public:
  ObstacleDesignProblem(ParamOptProblem *paramopt_);
  double E(const mfem::Vector &U, int &eval_err) override;
  double E(const mfem::Vector &u, const mfem::Vector &p,
           const mfem::Vector &theta, int &eval_err) override;
  void DthE(const mfem::Vector &u, const mfem::Vector &p,
            const mfem::Vector &theta, mfem::Vector &gradE) override;
  mfem::Operator *DththE(const mfem::Vector &u, const mfem::Vector &p,
                         const mfem::Vector &theta) override;
};
#endif // MPECPROBLEM_DEFS
