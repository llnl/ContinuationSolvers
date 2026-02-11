#include "mfem.hpp"
#include "../utilities.hpp"

#ifndef PARPROBLEM_DEFS
#define PARPROBLEM_DEFS

// abstract GeneralOptProblem class
// of the form
// min_(u,m) f(u,m) s.t. c(u,m)=0 and m>=ml
// the primal variable (u, m) is represented as a BlockVector
class GeneralOptProblem
{
protected:
    int dimU, dimM, dimC;
    int dimUglb, dimMglb;
    HYPRE_BigInt * dofOffsetsU;
    HYPRE_BigInt * dofOffsetsM;
    mfem::Array<int> block_offsetsx;
    mfem::Vector ml;
    int label;
public:
    GeneralOptProblem();
    virtual void Init(HYPRE_BigInt * dofOffsetsU_, HYPRE_BigInt * dofOffsetsM_);
    virtual double CalcObjective(const mfem::BlockVector &, int &) = 0;
    double CalcObjective(const mfem::BlockVector &) ;
    virtual void Duf(const mfem::BlockVector &, mfem::Vector &) = 0;
    virtual void Dmf(const mfem::BlockVector &, mfem::Vector &) = 0;
    void CalcObjectiveGrad(const mfem::BlockVector &, mfem::BlockVector &);
    virtual mfem::Operator * Duuf(const mfem::BlockVector &) = 0;
    virtual mfem::Operator * Dumf(const mfem::BlockVector &) = 0;
    virtual mfem::Operator * Dmuf(const mfem::BlockVector &) = 0;
    virtual mfem::Operator * Dmmf(const mfem::BlockVector &) = 0;
    virtual mfem::Operator * Duc(const mfem::BlockVector &) = 0;
    virtual mfem::Operator * Dmc(const mfem::BlockVector &) = 0;
    virtual void c(const mfem::BlockVector &, mfem::Vector &, int &) = 0;
    void c(const mfem::BlockVector &, mfem::Vector &) ;
    int GetDimU() const { return dimU; };
    int GetDimM() const { return dimM; }; 
    int GetDimC() const { return dimC; };
    int GetDimUGlb() const { return dimUglb; };
    int GetDimMGlb() const { return dimMglb; };
    HYPRE_BigInt * GetDofOffsetsU() const { return dofOffsetsU; };
    HYPRE_BigInt * GetDofOffsetsM() const { return dofOffsetsM; }; 
    mfem::Vector Getml() const { return ml; };
    void setProblemLabel(int label_) { label = label_; };
    int getProblemLabel() { return label; };
    ~GeneralOptProblem();
};


// abstract ContactProblem class
// of the form
// min_d e(d) s.t. g(d) >= 0
class OptProblem : public GeneralOptProblem
{
protected:
    mfem::HypreParMatrix * Ih;
public:
    OptProblem();
    void Init(HYPRE_BigInt *, HYPRE_BigInt *);
    
    // GeneralOptProblem methods are defined in terms of
    // OptProblem specific methods: E, DdE, DddE, g, Ddg
    double CalcObjective(const mfem::BlockVector &, int &) ; 
    void Duf(const mfem::BlockVector &, mfem::Vector &) ;
    void Dmf(const mfem::BlockVector &, mfem::Vector &) ;
    mfem::Operator * Duuf(const mfem::BlockVector &);
    mfem::Operator * Dumf(const mfem::BlockVector &);
    mfem::Operator * Dmuf(const mfem::BlockVector &);
    mfem::Operator * Dmmf(const mfem::BlockVector &);
    void c(const mfem::BlockVector &, mfem::Vector &, int &) ;
    mfem::Operator * Duc(const mfem::BlockVector &);
    mfem::Operator * Dmc(const mfem::BlockVector &);
    
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
    virtual mfem::Operator * DddE(const mfem::Vector &d) = 0;

    // Constraint function g(d) >= 0, e.g., gap function
    // input: d, an mfem::Vector,
    //       gd, an mfem::Vector, which upon successfully calling the g method will be
    //                            the evaluation of the function g at d
    // output: none
    virtual void g(const mfem::Vector &d, mfem::Vector &gd, int &) = 0;
    // Jacobian of constraint function Dg / Dd, e.g., gap function Jacobian
    // input:  d, an mfem::Vector,
    // output: The Jacobain of the constraint function g at d, a pointer to a Operator
    virtual mfem::Operator * Ddg(const mfem::Vector &) = 0;
    virtual ~OptProblem();
};



class ReducedOptProblem : public OptProblem
{
protected:
  mfem::HypreParMatrix *J;
  mfem::HypreParMatrix *P; // projector
  OptProblem  *problem;
public:
  ReducedOptProblem(OptProblem *problem, HYPRE_Int * constraintMask);
  ReducedOptProblem(OptProblem *problem, mfem::HypreParVector & constraintMask);
  double E(const mfem::Vector &, int &);
  void DdE(const mfem::Vector &, mfem::Vector &);
  mfem::Operator * DddE(const mfem::Vector &);
  void g(const mfem::Vector &, mfem::Vector &, int &);
  mfem::Operator * Ddg(const mfem::Vector &);
  OptProblem * GetProblem() {  return problem; }
  virtual ~ReducedOptProblem();
};




#endif
