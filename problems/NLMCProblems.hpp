#include "mfem.hpp"
#include "OptProblems.hpp"

#ifndef PROBLEM_DEFS
#define PROBLEM_DEFS

/* Abstract GeneralNLMCProblem class
 * to describe the nonlinear mixed complementarity problem
 * 0 <= x \perp F(x, y) >= 0
 *              Q(x, y)  = 0
 * where NLMC stands for nonlinear mixed complementarity 
 */
class GeneralNLMCProblem
{
protected:
   int dimx, dimy;
   mfem::Array<int> xyoffsets;
   HYPRE_BigInt dimxglb, dimyglb;
   HYPRE_BigInt * dofOffsetsx;
   HYPRE_BigInt * dofOffsetsy;
   int label;
public:
   GeneralNLMCProblem();
   virtual void Init(HYPRE_BigInt * dofOffsetsx_, HYPRE_BigInt * dofOffsetsy_);
   virtual void F(const mfem::Vector &x, const mfem::Vector &y, mfem::Vector &feval, int &eval_err, bool new_pt=true) const = 0;
   virtual void Q(const mfem::Vector &x, const mfem::Vector &y, mfem::Vector &qeval, int &eval_err, bool new_pt=true) const = 0;
   virtual mfem::Operator * DxF(const mfem::Vector &x, const mfem::Vector &y, bool new_pt=true) = 0;
   virtual mfem::Operator * DyF(const mfem::Vector &x, const mfem::Vector &y, bool new_pt=true) = 0;
   virtual mfem::Operator * DxQ(const mfem::Vector &x, const mfem::Vector &y, bool new_pt=true) = 0;
   virtual mfem::Operator * DyQ(const mfem::Vector &x, const mfem::Vector &y, bool new_pt=true) = 0;
   int GetDimx() const { return dimx; };
   int GetDimy() const { return dimy; }; 
   HYPRE_BigInt GetDimxGlb() const { return dimxglb; };
   HYPRE_BigInt GetDimyGlb() const { return dimyglb; };
   HYPRE_BigInt * GetDofOffsetsx() const { return dofOffsetsx; };
   HYPRE_BigInt * GetDofOffsetsy() const { return dofOffsetsy; }; 
   void setProblemLabel(int label_) { label = label_; };
   int getProblemLabel() { return label; };
   mfem::BlockVector GetOptimizationVariable() {
      mfem::BlockVector temp(xyoffsets);
      temp = 0.0;
      return temp;
   };
   virtual ~GeneralNLMCProblem();
};


class OptNLMCProblem : public GeneralNLMCProblem
{
protected:
   OptProblem * optproblem;
   mfem::HypreParMatrix * dFdx = nullptr;
   mfem::HypreParMatrix * dFdy = nullptr;
   mfem::HypreParMatrix * dQdx = nullptr;
   mfem::HypreParMatrix * dQdy = nullptr;
public:
   OptNLMCProblem(OptProblem * problem_);
   void F(const mfem::Vector &x, const mfem::Vector &y, mfem::Vector &feval, int &eval_err, bool new_pt=true) const override;
   void Q(const mfem::Vector &x, const mfem::Vector &y, mfem::Vector &qeval, int &eval_err, bool new_pt=true) const override;
   mfem::Operator * DxF(const mfem::Vector &x, const mfem::Vector &y, bool new_pt) override;
   mfem::Operator * DyF(const mfem::Vector &x, const mfem::Vector &y, bool new_pt) override;
   mfem::Operator * DxQ(const mfem::Vector &x, const mfem::Vector &y, bool new_pt) override;
   mfem::Operator * DyQ(const mfem::Vector &x, const mfem::Vector &y, bool new_pt) override;
   OptProblem * GetOptProblem() { return optproblem;  };
   virtual ~OptNLMCProblem();    
};


class EqualityConstrainedHomotopyProblem : public GeneralNLMCProblem
{
protected:
   mfem::HypreParMatrix * dFdx = nullptr;
   mfem::HypreParMatrix * dFdy = nullptr;
   mfem::HypreParMatrix * dQdx = nullptr;
   mfem::HypreParMatrix * dQdy = nullptr;
   mutable mfem::Vector q_cache;
   int dimu_;
   int dimuglb_;
   int dimufull_;
   int dimc_;
   int dimcglb_;
   mfem::Array<int> y_partition;
   HYPRE_BigInt * uOffsets_ = nullptr;
   HYPRE_BigInt * cOffsets_ = nullptr;
   mfem::Solver * adjoint_solver = nullptr;
   bool own_adjoint_solver = true;
   bool adjoint_is_symmetric = false;
   bool set_sizes = false;

   mfem::Array<int> fixed_tdof_list_;
   mfem::Array<int> disp_tdof_list_;
   bool has_essential_dofs = false;
   std::unique_ptr<mfem::HypreParMatrix> restriction_;
   std::unique_ptr<mfem::HypreParMatrix> prolongation_;
   std::unique_ptr<mfem::HypreParMatrix> disp_restriction_;
   std::unique_ptr<mfem::HypreParMatrix> disp_prolongation_;
   mutable mfem::Vector ufull_;
   mfem::Vector uDC_;
public:
   EqualityConstrainedHomotopyProblem();
   void EqualityConstrainedHomotopyInit();
   EqualityConstrainedHomotopyProblem(mfem::Array<int> fixed_tdof_list, mfem::Array<int> disp_tdof_list, const mfem::Vector uDC); 
   void SetSizes(int dimu, int dimc);
   void SetSizes(HYPRE_BigInt * uOffsets, HYPRE_BigInt * cOffsets);
   virtual mfem::Vector residual(const mfem::Vector & u, bool new_pt) const = 0;
   virtual mfem::Vector constraintJacobianTvp(const mfem::Vector &u, const mfem::Vector & l, bool new_pt, bool new_deriv) const = 0;
   virtual mfem::HypreParMatrix * residualJacobian(const mfem::Vector & u, bool new_pt, bool new_deriv) = 0; 
   virtual mfem::Vector constraint(const mfem::Vector & u, bool new_pt) const = 0;
   virtual mfem::HypreParMatrix * constraintJacobian(const mfem::Vector & u, bool new_pt, bool new_deriv) = 0;
   void F(const mfem::Vector &x, const mfem::Vector &y, mfem::Vector &feval, int &eval_err, bool new_pt = true) const override;
   void Q(const mfem::Vector &x, const mfem::Vector &y, mfem::Vector &qeval, int &eval_err, bool new_pt = true) const override;
   mfem::Operator * DxF(const mfem::Vector &/*x*/, const mfem::Vector &/*y*/, bool /*new_pt*/ = true) override { return dFdx; };
   mfem::Operator * DyF(const mfem::Vector &/*x*/, const mfem::Vector &/*y*/, bool /*new_pt*/ = true) override { return dFdy; };
   mfem::Operator * DxQ(const mfem::Vector &/*x*/, const mfem::Vector &/*y*/, bool /*new_pt*/ = true) override { return dQdx; };
   mfem::Operator * DyQ(const mfem::Vector &x, const mfem::Vector &y, bool new_pt=true) override;
   mfem::Vector GetDisplacement(mfem::Vector &Xf);
   mfem::Vector GetLagrangeMultiplier(mfem::Vector &Xf);
   int GetDisplacementDim() const { return dimu_; };
   int GetMultiplierDim() const { return dimc_; };
   int GetGlobalDisplacementDim() const { return dimuglb_; };
   int GetGlobalMultiplierDim() const { return dimcglb_; };
   void SetAdjointSolver(mfem::Solver * adjoint_solver_);
   void SetSymmetricAdjoint(bool symmetric) { adjoint_is_symmetric = symmetric; };
   void AdjointSolve(const mfem::Vector & evaluation_u_point, const mfem::Vector & adjoint_load, 
      mfem::Vector & adjoint);
   virtual ~EqualityConstrainedHomotopyProblem();
};




#endif
