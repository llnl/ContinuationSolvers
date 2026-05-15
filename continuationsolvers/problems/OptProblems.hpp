#include "../utilities.hpp"
#include "mfem.hpp"

#ifndef OPTPROBLEM_DEFS
#define OPTPROBLEM_DEFS

// abstract GeneralOptProblem class
// of the form
// min_(u,m) f(u,m) s.t. c(u,m)=0 and m>=ml
// the primal variable (u, m) is represented as a BlockVector
class GeneralOptProblem {
protected:
  int dimU, dimM, dimC;
  int dimUglb, dimMglb;
  HYPRE_BigInt *dofOffsetsU;
  HYPRE_BigInt *dofOffsetsM;
  mfem::Array<int> block_offsetsx;
  mfem::Vector ml;
  int label;

public:
  GeneralOptProblem();
  virtual void Init(HYPRE_BigInt *dofOffsetsU_, HYPRE_BigInt *dofOffsetsM_);
  virtual double CalcObjective(const mfem::BlockVector &, int &) = 0;
  double CalcObjective(const mfem::BlockVector &);
  virtual void Duf(const mfem::BlockVector &, mfem::Vector &) = 0;
  virtual void Dmf(const mfem::BlockVector &, mfem::Vector &) = 0;
  void CalcObjectiveGrad(const mfem::BlockVector &, mfem::BlockVector &);
  virtual mfem::Operator *Duuf(const mfem::BlockVector &) = 0;
  virtual mfem::Operator *Dumf(const mfem::BlockVector &) = 0;
  virtual mfem::Operator *Dmuf(const mfem::BlockVector &) = 0;
  virtual mfem::Operator *Dmmf(const mfem::BlockVector &) = 0;
  virtual mfem::Operator *Duc(const mfem::BlockVector &) = 0;
  virtual mfem::Operator *Dmc(const mfem::BlockVector &) = 0;
  virtual mfem::Operator *Duucl(const mfem::BlockVector &,
                                const mfem::Vector &) = 0;
  virtual mfem::Operator *Dumcl(const mfem::BlockVector &,
                                const mfem::Vector &) = 0;
  virtual mfem::Operator *Dmucl(const mfem::BlockVector &,
                                const mfem::Vector &) = 0;
  virtual mfem::Operator *Dmmcl(const mfem::BlockVector &,
                                const mfem::Vector &) = 0;
  virtual void c(const mfem::BlockVector &, mfem::Vector &, int &) = 0;
  void c(const mfem::BlockVector &, mfem::Vector &);
  int GetDimU() const { return dimU; };
  int GetDimM() const { return dimM; };
  int GetDimC() const { return dimC; };
  int GetDimUGlb() const { return dimUglb; };
  int GetDimMGlb() const { return dimMglb; };
  HYPRE_BigInt *GetDofOffsetsU() const { return dofOffsetsU; };
  HYPRE_BigInt *GetDofOffsetsM() const { return dofOffsetsM; };
  mfem::Vector Getml() const { return ml; };
  void setProblemLabel(int label_) { label = label_; };
  int getProblemLabel() { return label; };
  ~GeneralOptProblem();
};

// abstract inequality constrained optimization problem
// class of the form
// min_d E(d) s.t. g(d) >= 0
class OptProblem : public GeneralOptProblem {
protected:
  mfem::HypreParMatrix *Ih;

public:
  OptProblem();
  void Init(HYPRE_BigInt *, HYPRE_BigInt *);

  // GeneralOptProblem methods are defined in terms of
  // OptProblem specific methods: E, DdE, DddE, g, Ddg
  double CalcObjective(const mfem::BlockVector &, int &) override;
  void Duf(const mfem::BlockVector &, mfem::Vector &) override;
  void Dmf(const mfem::BlockVector &, mfem::Vector &) override;
  mfem::Operator *Duuf(const mfem::BlockVector &) override;
  mfem::Operator *Dumf(const mfem::BlockVector &) override;
  mfem::Operator *Dmuf(const mfem::BlockVector &) override;
  mfem::Operator *Dmmf(const mfem::BlockVector &) override;
  void c(const mfem::BlockVector &, mfem::Vector &, int &) override;
  mfem::Operator *Duc(const mfem::BlockVector &) override;
  mfem::Operator *Dmc(const mfem::BlockVector &) override;
  mfem::Operator *Duucl(const mfem::BlockVector &,
                        const mfem::Vector &) override;
  mfem::Operator *Dumcl(const mfem::BlockVector &,
                        const mfem::Vector &) override;
  mfem::Operator *Dmucl(const mfem::BlockVector &,
                        const mfem::Vector &) override;
  mfem::Operator *Dmmcl(const mfem::BlockVector &,
                        const mfem::Vector &) override;

  // OptProblem specific methods:

  // energy objective function e(d)
  // input: d an mfem::Vector
  // output: e(d) a double
  virtual double E(const mfem::Vector &d, int &) = 0;
  // gradient of energy objective De / Dd
  // input: d an mfem::Vector,
  //        gradE an mfem::Vector, which will be the gradient of E at d
  // output: none
  virtual void DdE(const mfem::Vector &d, mfem::Vector &gradE) = 0;

  // Hessian of energy objective D^2 e / Dd^2
  // input:  d, an mfem::Vector
  // output: The Hessian of the energy objective at d, a pointer to an
  // mfem::Operator
  virtual mfem::Operator *DddE(const mfem::Vector &d) = 0;

  // Constraint function g(d) >= 0, e.g., gap function
  // input: d, an mfem::Vector,
  //       gd, an mfem::Vector, which upon successfully calling the g method
  //       will be
  //                            the evaluation of the function g at d
  // output: none
  virtual void g(const mfem::Vector &d, mfem::Vector &gd, int &) = 0;
  // Jacobian of constraint function Dg / Dd, e.g., gap function Jacobian
  // input:  d, an mfem::Vector,
  // output: The Jacobain of the constraint function g at d, a pointer to an
  // mfem::Operator
  virtual mfem::Operator *Ddg(const mfem::Vector &) = 0;

  // Hessian of constraint times Lagrange multiplier D^2 (g^T l) Dd^2
  // input: d, an mfem::Vector (primal)
  // input: l, an mfem::Vector (Lagrange multiplier)
  // output: The Hessian of the constraint function times Lagrange multiplier at
  // d, a pointer to an mfem::Operator
  virtual mfem::Operator *Dddgl(const mfem::Vector &, const mfem::Vector &);
  virtual ~OptProblem();
};

// abstract equality-constrained optimization problem
// class of the form
// min_d E(d) s.t. g(d) = 0
class OptEqProblem : public GeneralOptProblem {
protected:
  std::unique_ptr<mfem::HypreParMatrix> dcdm;
  std::unique_ptr<mfem::HypreParMatrix> Hmmf;
  HYPRE_BigInt *dofOffsetsC;

public:
  OptEqProblem();
  void Init(HYPRE_BigInt *, HYPRE_BigInt *);

  // GeneralOptProblem methods are defined in terms of
  // OptProblem specific methods: E, DdE, DddE, g, Ddg
  double CalcObjective(const mfem::BlockVector &, int &) override;
  void Duf(const mfem::BlockVector &, mfem::Vector &) override;
  void Dmf(const mfem::BlockVector &, mfem::Vector &) override;
  mfem::Operator *Duuf(const mfem::BlockVector &) override;
  mfem::Operator *Dumf(const mfem::BlockVector &) override;
  mfem::Operator *Dmuf(const mfem::BlockVector &) override;
  mfem::Operator *Dmmf(const mfem::BlockVector &) override;
  void c(const mfem::BlockVector &, mfem::Vector &, int &) override;
  mfem::Operator *Duc(const mfem::BlockVector &) override;
  mfem::Operator *Dmc(const mfem::BlockVector &) override;
  mfem::Operator *Duucl(const mfem::BlockVector &,
                        const mfem::Vector &) override;
  mfem::Operator *Dumcl(const mfem::BlockVector &,
                        const mfem::Vector &) override;
  mfem::Operator *Dmucl(const mfem::BlockVector &,
                        const mfem::Vector &) override;
  mfem::Operator *Dmmcl(const mfem::BlockVector &,
                        const mfem::Vector &) override;

  // OptProblem specific methods:

  // energy objective function e(d)
  // input: d an mfem::Vector
  // output: e(d) a double
  virtual double E(const mfem::Vector &d, int &) = 0;
  // gradient of energy objective De / Dd
  // input: d an mfem::Vector,
  //        gradE an mfem::Vector, which will be the gradient of E at d
  // output: none
  virtual void DdE(const mfem::Vector &d, mfem::Vector &gradE) = 0;

  // Hessian of energy objective D^2 e / Dd^2
  // input:  d, an mfem::Vector
  // output: The Hessian of the energy objective at d, a pointer to a Operator
  virtual mfem::Operator *DddE(const mfem::Vector &d) = 0;

  // Constraint function g(d) = 0, e.g., (tied) gap function
  // input: d, an mfem::Vector,
  //       gd, an mfem::Vector, which upon successfully calling the g method
  //       will be
  //                            the evaluation of the function g at d
  // output: none
  virtual void g(const mfem::Vector &d, mfem::Vector &gd, int &) = 0;
  // Jacobian of constraint function Dg / Dd, e.g., gap function Jacobian
  // input:  d, an mfem::Vector,
  // output: The Jacobain of the constraint function g at d, a pointer to a
  // Operator
  virtual mfem::Operator *Ddg(const mfem::Vector &) = 0;

  // Hessian of constraint times Lagrange multiplier D^2 (g^T l) Dd^2
  // input: d, an mfem::Vector (primal)
  // input: l, an mfem::Vector (Lagrange multiplier)
  // output: The Hessian of the constraint function times Lagrange multiplier at
  // d, a pointer to an mfem::Operator
  virtual mfem::Operator *Dddgl(const mfem::Vector &, const mfem::Vector &);
  virtual ~OptEqProblem();
};

class ParamOptProblem : public OptProblem {
protected:
  mfem::Vector theta_default; // default value of parameter
  bool theta_initialized = false;
  int dimTheta = 0;
  int dimThetaglb;
  HYPRE_BigInt *dofOffsetsTheta = nullptr;
  std::unique_ptr<mfem::HypreParMatrix>
      Hddgl; // (nabla_(d,d) g)_(i,j,k) x l_i contracting on first index i of
             // (nabla_(d,d) g) in R^(n_c x n_d x n_d)
  std::unique_ptr<mfem::HypreParMatrix>
      Hdthgl; // (nabla_(d,th) g)_(i,j,k) x l_i contracting on first index i of
              // (nabla_(d,th) g) in R^(n_c x n_d x n_th)
  std::unique_ptr<mfem::HypreParMatrix>
      Hththgl; // (nabla_(th,th) g)_(i,j,k) x l_i contracting on first index i
               // of (nabla_(th,th) g) in R^(n_c x n_th x n_th)
  std::unique_ptr<mfem::HypreParMatrix>
      Hddgl2; // (nabla_(d,d) g)_(i,j,k) x l_j contracting on second index j of
              // (nabla_(d,d) g) in R^(n_c x n_d x n_d)
  std::unique_ptr<mfem::HypreParMatrix>
      Hthdgl2; // (nabla_(th,d) g)_(i,j,k) x l_j contracting on second index j
               // of (nabla_(th,d) g) in R^(n_c x n_d x n_th)
  std::unique_ptr<mfem::HypreParMatrix>
      HdddEl; // (nabla_(d,d,d) E)_(i,j,k) x l_k contracting on final index k
              // of (nabla_(d,d,d) E) in R^(n_d x n_d x n_d)
  std::unique_ptr<mfem::HypreParMatrix>
      HthddEl; // (nabla_(th,d,d) E)_(i,j,k) x l_k contracting on final index k
               // of (nabla_(th,d,d) E) in R^(n_th x n_d x n_d)
  std::unique_ptr<mfem::HypreParMatrix>
      HththdEl; // (nabla_(th,th,d) E)_(i,j,k) x l_k contracting on final index
                // k of (nabla_(th,th,d) E) in R^(n_th x n_th x n_d)
		//
  std::unique_ptr<mfem::HypreParMatrix> HddE;
  std::unique_ptr<mfem::HypreParMatrix> HdthE;
  std::unique_ptr<mfem::HypreParMatrix> Jdg;
  std::unique_ptr<mfem::HypreParMatrix> Jthg;

  void InitTheta(const mfem::Vector &theta);

public:
  ParamOptProblem();
  HYPRE_BigInt *GetDofOffsetsTheta() const;
  int GetDimTheta() const { return dimTheta; };
  int GetDimThetaGlb() const { return dimThetaglb; };
  mfem::Vector GetTheta() { return theta_default; };

  // ParamOptProblem specific methods:

  // energy objective function e(d, \theta)
  // input: d an mfem::Vector
  // input: theta an mfem::Vector (design parameter for upper level problem)
  // output: e(d, theta) a double
  virtual double E(const mfem::Vector &d, const mfem::Vector &theta,
                   int &eval_err) = 0;

  // energy objective E(d, theta=theta_default)
  double E(const mfem::Vector &d, int &eval_err) override;

  // gradient of energy objective De(d,\theta) / Dd
  // input: d an mfem::Vector,
  // input: \theta an mfem::Vector
  //        gradE an mfem::Vector, which will be the gradient of E at \theta and
  //        at and w.r.t. d
  // output: none
  virtual void DdE(const mfem::Vector &d, const mfem::Vector &theta,
                   mfem::Vector &gradE) = 0;

  // gradient of energy objective De(d,\theta) / Dd |_{theta = theta_default}
  // input: d an mfem::Vector,
  //        gradE an mfem::Vector, which will be the gradient of E at d
  // output: none
  void DdE(const mfem::Vector &d, mfem::Vector &gradE) override;

  // Hessian of energy objective D^2 e / Dd^2
  // input:  d, an mfem::Vector,
  // input: \theta, an mfem::Vector,
  // output: The Hessian of the energy objective at d, a pointer to an
  // mfem::Operator
  virtual mfem::Operator *DddE(const mfem::Vector &d,
                               const mfem::Vector &theta) = 0;

  // mixed Hessian of energy objective D^2 e / (Dth Dd)
  // input:  d, an mfem::Vector,
  // input: \theta, an mfem::Vector,
  // output: The Hessian of the energy objective at (d, theta), a pointer to an
  // mfem::Operator
  virtual mfem::Operator *DdthE(const mfem::Vector &d,
                                const mfem::Vector &theta) = 0;

  // Hessian of energy objective D^2 e / Dd^2
  // input:  d, an mfem::Vector
  // output: The Hessian of the energy objective at d, a pointer to a Operator
  mfem::Operator *DddE(const mfem::Vector &d) override;

  // Third-derivative contraction energy objective D^3 e / Dd^3 l
  // input:  d, an mfem::Vector,
  // input:  l, an mfem::Vector
  // input: \theta, an mfem::Vector,
  // output: Third-derivative contraction of the energy objective at (d, theta),
  // a pointer to an mfem::Operator
  virtual mfem::Operator *DdddEl(const mfem::Vector &d, const mfem::Vector &l,
                                 const mfem::Vector &theta);

  // Third-derivative contraction energy objective D^3 e / Dth Dd^2 l
  // input:  d, an mfem::Vector,
  // input:  l, an mfem::Vector
  // input: \theta, an mfem::Vector,
  // output: Third-derivative contraction of the energy objective at (d, theta),
  // a pointer to an mfem::Operator
  virtual mfem::Operator *DthddEl(const mfem::Vector &d, const mfem::Vector &l,
                                  const mfem::Vector &theta);

  // Third-derivative contraction energy objective (D^3 e / Dth^2 Dd) l
  // input:  d, an mfem::Vector,
  // input:  l, an mfem::Vector
  // input: \theta, an mfem::Vector,
  // output: Third-derivative contraction of the energy objective at (d, theta),
  // a pointer to an mfem::Operator
  virtual mfem::Operator *DththdEl(const mfem::Vector &d, const mfem::Vector &l,
                                   const mfem::Vector &theta);

  // Constraint function g(d, theta) >= 0, e.g., gap function
  // input: d, an mfem::Vector,
  // input: theta, an mfem::Vector
  //       gd, an mfem::Vector, which upon successfully calling the g method
  //       will be
  //                            the evaluation of the function g at d
  // output: none
  virtual void g(const mfem::Vector &d, const mfem::Vector &theta,
                 mfem::Vector &gd, int &eval_err) = 0;

  // Constraint function g(d, theta_default)
  void g(const mfem::Vector &d, mfem::Vector &gd, int &eval_err) override;
  // Jacobian of constraint function Dg(d,\theta) / Dd, e.g., gap function
  // Jacobian input:  d, an mfem::Vector, input: \theta, an mfem::Vector output:
  // The Jacobain of the constraint function g at (d, \theta), with respect to
  // d, a pointer to an mfem::Operator
  virtual mfem::Operator *Ddg(const mfem::Vector &d,
                              const mfem::Vector &theta) = 0;

  mfem::Operator *Ddg(const mfem::Vector &d) override;

  // Jacobian of constraint with respect to parameter, D{g(d, \theta)}/D\theta
  // input: d, an mfem::Vector
  // input: \theta, an mfem::Vector
  // output: The Jacobian of the constraint function g at (d, \theta), with
  // respect to \theta, a pointer to an mfem::Operator
  virtual mfem::Operator *Dthg(const mfem::Vector &d,
                               const mfem::Vector &theta) = 0;

  virtual mfem::Operator *Dddgl(const mfem::Vector &d, const mfem::Vector &l,
                                const mfem::Vector &theta);

  mfem::Operator *Dddgl(const mfem::Vector &d, const mfem::Vector &l) override;
  virtual mfem::Operator *Ddthgl(const mfem::Vector &d, const mfem::Vector &l,
                                 const mfem::Vector &theta);
  virtual mfem::Operator *Dththgl(const mfem::Vector &d, const mfem::Vector &l,
                                  const mfem::Vector &theta);

  virtual mfem::Operator *Dddgl2(const mfem::Vector &d, const mfem::Vector &l,
                                 const mfem::Vector &theta);
  virtual mfem::Operator *Dthdgl2(const mfem::Vector &d, const mfem::Vector &l,
                                  const mfem::Vector &theta);
  virtual ~ParamOptProblem();
};

// assuming dimU = dimTheta
class ReducedParamOptProblem : public ParamOptProblem {
protected:
  ParamOptProblem * problem;
  // constraint restriction operator
  std::unique_ptr<mfem::HypreParMatrix> Rc;
  // constraint prolongation operator
  std::unique_ptr<mfem::HypreParMatrix> Pc;
  // dof restriction operator
  std::unique_ptr<mfem::HypreParMatrix> Rdof;
  // dof prolongation operator
  std::unique_ptr<mfem::HypreParMatrix> Pdof;
  
  
  mfem::Vector dDC; // Dirichlet dofs
  mfem::Vector dFull; // helper vec
  mfem::Vector gradEFull; // helper vec
  mfem::Vector gFull; // helper vec
  mfem::Vector thetaFull;
public:
  ReducedParamOptProblem(ParamOptProblem * problem_, mfem::Array<int> tdof_list);
  // ParamOptProblem specific methods:

  // energy objective function e(d, \theta)
  // input: d an mfem::Vector
  // input: theta an mfem::Vector (design parameter for upper level problem)
  // output: e(d, theta) a double
  double E(const mfem::Vector &d, const mfem::Vector &theta,
                   int &eval_err) override;

  // gradient of energy objective De(d,\theta) / Dd
  // input: d an mfem::Vector,
  // input: \theta an mfem::Vector
  //        gradE an mfem::Vector, which will be the gradient of E at \theta and
  //        at and w.r.t. d
  // output: none
  void DdE(const mfem::Vector &d, const mfem::Vector &theta,
                   mfem::Vector &gradE) override;

  // Hessian of energy objective D^2 e / Dd^2
  // input:  d, an mfem::Vector,
  // input: \theta, an mfem::Vector,
  // output: The Hessian of the energy objective at d, a pointer to an
  // mfem::Operator
  mfem::Operator *DddE(const mfem::Vector &d,
                               const mfem::Vector &theta) override;

  // mixed Hessian of energy objective D^2 e / (Dth Dd)
  // input:  d, an mfem::Vector,
  // input: \theta, an mfem::Vector,
  // output: The Hessian of the energy objective at (d, theta), a pointer to an
  // mfem::Operator
  mfem::Operator *DdthE(const mfem::Vector &d,
                                const mfem::Vector &theta) override;

  // Constraint function g(d, theta) >= 0, e.g., gap function
  // input: d, an mfem::Vector,
  // input: theta, an mfem::Vector
  //       gd, an mfem::Vector, which upon successfully calling the g method
  //       will be
  //                            the evaluation of the function g at d
  // output: none
  void g(const mfem::Vector &d, const mfem::Vector &theta,
                 mfem::Vector &gd, int &eval_err) override;

  // Jacobian of constraint function Dg(d,\theta) / Dd, e.g., gap function
  // Jacobian input:  d, an mfem::Vector, input: \theta, an mfem::Vector output:
  // The Jacobain of the constraint function g at (d, \theta), with respect to
  // d, a pointer to an mfem::Operator
  mfem::Operator *Ddg(const mfem::Vector &d,
                              const mfem::Vector &theta) override;

  // Jacobian of constraint with respect to parameter, D{g(d, \theta)}/D\theta
  // input: d, an mfem::Vector
  // input: \theta, an mfem::Vector
  // output: The Jacobian of the constraint function g at (d, \theta), with
  // respect to \theta, a pointer to an mfem::Operator
  mfem::Operator *Dthg(const mfem::Vector &d,
                               const mfem::Vector &theta) override;
  
  void ProlongateToFullDofs(const mfem::Vector &, mfem::Vector &);

  virtual ~ReducedParamOptProblem();
};


class ReducedOptProblem : public OptProblem {
protected:
  // constraint restriction operator
  std::unique_ptr<mfem::HypreParMatrix> Rc;
  // constraint prolongation operator
  std::unique_ptr<mfem::HypreParMatrix> Pc;
  // dof restriction operator
  std::unique_ptr<mfem::HypreParMatrix> Rdof;
  // dof prolongation operator
  std::unique_ptr<mfem::HypreParMatrix> Pdof;
  // reduced constraint Jacobain J = Rc Jfull Rdof^T
  std::unique_ptr<mfem::HypreParMatrix> J;
  // reduced energy Hessian H = Rdof Hfull Rdof^T 
  std::unique_ptr<mfem::HypreParMatrix> H; // reduced energy Hessian 
  mfem::Vector dDC; // Dirichlet dofs
  mfem::Vector dFull; // helper vec
  mfem::Vector gradEfull; // helper vec
  mfem::Vector gfull; // helper vec
  OptProblem *problem; // non-owing optimization problem pointer
public:
  ReducedOptProblem(OptProblem *problem, HYPRE_Int *constraintMask);
  ReducedOptProblem(OptProblem *problem, HYPRE_Int *dofMask, HYPRE_Int *constraintMask);
  ReducedOptProblem(OptProblem *problem, mfem::HypreParVector &constraintMask);
  double E(const mfem::Vector &, int &);
  void DdE(const mfem::Vector &, mfem::Vector &);
  mfem::Operator *DddE(const mfem::Vector &);
  void g(const mfem::Vector &, mfem::Vector &, int &);
  mfem::Operator *Ddg(const mfem::Vector &);
  OptProblem *GetProblem() { return problem; }
  void ProlongateToFullDofs(const mfem::Vector &, mfem::Vector &);
  virtual ~ReducedOptProblem();
};



#if 0
// TODO: something better than needing to create the OptEqProblem and then this additional
// wrapper around it
class ReducedOptEqProblem : public OptEqProblem {
protected:
  std::unique_ptr<mfem::HypreParMatrix> Rc; // constraint restriction operator
  std::unique_ptr<mfem::HypreParMatrix> R;  // dof restriction operator
  std::unique_ptr<mfem::HypreParMatrix> J;  // reduced constraint Jacobian
  std::unique_ptr<mfem::HypreParMatrix> H;  // 
  OptEqProblem *problem;
public:
  ReducedOptEqProblem(OptProblem *problem, HYPRE_Int *constraintMask);
  double E(const mfem::Vector &, int &);
  void DdE(const mfem::Vector &, mfem::Vector &);
  mfem::Operator *DddE(const mfem::Vector &);
  void g(const mfem::Vector &, mfem::Vector &, int &);
  mfem::Operator *Ddg(const mfem::Vector &);
  virtual ~ReducedOptEqProblem();
};
#endif // ReducedOptEqProblem


#endif // OPTPROBLEM_DEFS
