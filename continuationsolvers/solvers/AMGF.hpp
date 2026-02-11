#include "mfem.hpp"
#include <fstream>
#include <iostream>


#ifndef AMGF_HPP
#define AMGF_HPP


/// AMG with filtering
class AMGF : public mfem::Solver
{
private:
    MPI_Comm comm;
    int numProcs, myid;
    const mfem::HypreParMatrix * A = nullptr;
    const mfem::HypreParMatrix * Pfiltered = nullptr;
    mfem::HypreBoomerAMG * amg = nullptr;
    mfem::HypreParMatrix * Afiltered = nullptr;
    mfem::Solver * Mfiltered = nullptr;
    bool additive = false;
    int relax_type = 8;
    void Init(MPI_Comm comm_);
    void InitAMG();
    void InitFilteredSpaceSolver();
public:
    AMGF(MPI_Comm comm_);
    AMGF(const mfem::Operator & Op, const mfem::Operator & P_);
    void SetOperator(const mfem::Operator &op);
    void SetFilteredTransferMap(const mfem::Operator & P);
    void EnableAdditiveCoupling() { additive = true; }
    void EnableMultiplicativeCoupling() { additive = false; }
    void SetAMGRelaxType(int relax_type_) { relax_type = relax_type_; }

    virtual void Mult(const mfem::Vector & y, mfem::Vector & x) const;

    ~AMGF()
    {
        delete amg;
        delete Mfiltered;
        delete Afiltered;
    }
};

#endif // AMGF_HPP
