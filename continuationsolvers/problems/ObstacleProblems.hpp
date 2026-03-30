#include "OptProblems.hpp"

#ifndef OBSTACLEPROBLEM_DEFS
#define OBSTACLEPROBLEM_DEFS

typedef double (*mfem_fun_ptr_type)(const mfem::Vector &);

class ObstacleProblem : public OptProblem
{
protected:
   // data to define energy objective function e(d) = 0.5 d^T K d - f^T d, g(d) = d >= \psi
   // stiffness matrix used to define objective
   mfem::ParBilinearForm *Kform = nullptr;
   mfem::ParLinearForm   *fform = nullptr;
   mfem::Array<int> ess_tdof_list; // needed for calls to FormSystemMatrix
   mfem::HypreParMatrix  K;
   mfem::HypreParMatrix* J = nullptr;
   mfem::ParFiniteElementSpace* Vh = nullptr;
   mfem::Vector f;
   mfem::Vector psi;
public :
   ObstacleProblem(mfem::ParFiniteElementSpace*, mfem::ParFiniteElementSpace*, mfem_fun_ptr_type fSource);
   ObstacleProblem(mfem::ParFiniteElementSpace*, mfem::ParFiniteElementSpace*, mfem_fun_ptr_type fSource, mfem_fun_ptr_type obstacleSource, mfem::Array<int> tdof_list, mfem::Vector &);
   double E(const mfem::Vector &, int &);
   void DdE(const mfem::Vector &, mfem::Vector &);
   mfem::Operator* DddE(const mfem::Vector &);
   void g(const mfem::Vector &, mfem::Vector &, int &);
   mfem::Operator* Ddg(const mfem::Vector &);
   virtual ~ObstacleProblem();
};


class ParamObstacleProblem : public ParamOptProblem
{
protected:
   // data to define energy objective function e(d, \theta) = 0.5 d^T K d - f^T d, g(d, \theta) = d - \theta >= 0
   // stiffness matrix used to define objective
   mfem::ParBilinearForm *Kform = nullptr;
   mfem::ParLinearForm   *fform = nullptr;
   mfem::Array<int> ess_tdof_list; // needed for calls to FormSystemMatrix
   mfem::HypreParMatrix  K;
   mfem::HypreParMatrix* Jd = nullptr;
   mfem::HypreParMatrix* Jth = nullptr;
   mfem::HypreParMatrix* Hddgl = nullptr;  // Hessian of (gap^T Lagrange multiplier) D^2 / Dd^2
   mfem::HypreParMatrix* Hthdgl = nullptr; // Hessian of (gap^T Lagrange multiplier) D^2 / (Dth Dd)
   mfem::HypreParMatrix* HdthE = nullptr; // mixed Hessian of (objective) D^2 / (Dd Dth)    
   mfem::ParFiniteElementSpace* Vh = nullptr;
   mfem::Vector f;
public :
   ParamObstacleProblem(mfem::ParFiniteElementSpace*, mfem_fun_ptr_type fSource, mfem_fun_ptr_type obstacleSource);
   double E(const mfem::Vector &d, const mfem::Vector &theta, int &eval_err) override;
   void DdE(const mfem::Vector &d, const mfem::Vector &theta, mfem::Vector &gradE) override;
   mfem::Operator* DddE(const mfem::Vector &d, const mfem::Vector &theta) override;
   mfem::Operator* DdthE(const mfem::Vector &d, const mfem::Vector & theta) override;
   void g(const mfem::Vector &d, const mfem::Vector &theta, mfem::Vector &gd, int &eval_err) override;
   mfem::Operator* Ddg(const mfem::Vector &d, const mfem::Vector &theta) override;
   mfem::Operator* Dthg(const mfem::Vector &d, const mfem::Vector &theta) override;
   mfem::Operator* Dddgl(const mfem::Vector & d, const mfem::Vector &l, const mfem::Vector &theta) override;
   mfem::Operator* Dthdgl(const mfem::Vector & d, const mfem::Vector &l, const mfem::Vector &theta);
   virtual ~ParamObstacleProblem();
};


#endif //OBSTACLEPROBLEM_DEFS
