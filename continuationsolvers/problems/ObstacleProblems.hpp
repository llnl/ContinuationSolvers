#include "OptProblems.hpp"

#ifndef OBSTACLEPROBLEM_DEFS
#define OBSTACLEPROBLEM_DEFS


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
   ObstacleProblem(mfem::ParFiniteElementSpace*, mfem::ParFiniteElementSpace*, double (*fSource)(const mfem::Vector &));
   ObstacleProblem(mfem::ParFiniteElementSpace*, mfem::ParFiniteElementSpace*, double (*fSource)(const mfem::Vector &), double (*obstacleSource)(const mfem::Vector &), mfem::Array<int> tdof_list, mfem::Vector &);
   double E(const mfem::Vector &, int &);
   void DdE(const mfem::Vector &, mfem::Vector &);
   mfem::Operator* DddE(const mfem::Vector &);
   void g(const mfem::Vector &, mfem::Vector &, int &);
   mfem::Operator* Ddg(const mfem::Vector &);
   virtual ~ObstacleProblem();
};

#endif //OBSTACLEPROBLEM_DEFS
