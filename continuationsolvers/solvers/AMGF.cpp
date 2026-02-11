#include "AMGF.hpp"
#include "../utilities.hpp"


AMGF::AMGF(MPI_Comm comm_): mfem::Solver()
{
    Init(comm_);
}

AMGF::AMGF(const mfem::Operator & Op, const mfem::Operator & P_)
: mfem::Solver()
{
   auto APtr = dynamic_cast<const mfem::HypreParMatrix *>(&Op);
   MFEM_VERIFY(APtr, "Operator: Not a compatible matrix type");
   Init(APtr->GetComm());
   auto PPtr = dynamic_cast<const mfem::HypreParMatrix *>(&P_);
   MFEM_VERIFY(PPtr, "Transfer Map: not a compatible matrix type");

   SetOperator(Op);
   SetFilteredTransferMap(P_);
}

void AMGF::Init(MPI_Comm comm_)
{
    comm=comm_;
    MPI_Comm_size(comm, &numProcs);
    MPI_Comm_rank(comm, &myid);
}

void AMGF::SetOperator(const mfem::Operator & Op)
{
    A = dynamic_cast<const mfem::HypreParMatrix *>(&Op);
    MFEM_VERIFY(A, "Operator: Not a compatible matrix type");
    height = A->Height();
    width = A->Width();
    InitAMG();
    if (Pfiltered)
    {
        InitFilteredSpaceSolver();
    }
}

void AMGF::SetFilteredTransferMap(const mfem::Operator & P)
{
    Pfiltered = dynamic_cast<const mfem::HypreParMatrix *>(&P);
    MFEM_VERIFY(Pfiltered, "Transfer Map: not a compatible matrix type");
    if (A)
    {
        InitFilteredSpaceSolver();
    }
}

void AMGF::InitAMG()
{
    if (amg)
    {
       delete amg;
    }
    amg = new mfem::HypreBoomerAMG(*A);
    amg->SetPrintLevel(0);
    amg->SetSystemsOptions(3);
    amg->SetRelaxType(relax_type);
}

void AMGF::InitFilteredSpaceSolver()
{
    if (Afiltered)
    {
       delete Afiltered;
    }
    if (Mfiltered)
    {
       delete Mfiltered;
    }
    Afiltered = mfem::RAP(A, Pfiltered);
    Mfiltered = new DirectSolver(*Afiltered);
}


void AMGF::Mult(const mfem::Vector & b, mfem::Vector & x) const
{
    MFEM_VERIFY(b.Size() == x.Size(), "Inconsistent x and y size");

    x = 0.0;
    mfem::Vector z(x);
    amg->Mult(b, z);
    x+=z;
    mfem::Vector rf(Pfiltered->Width());
    mfem::Vector xf(Pfiltered->Width());
    if (additive)
    {
        Pfiltered->MultTranspose(b,rf);
        Mfiltered->Mult(rf,xf);
        Pfiltered->Mult(xf,z);
    }
    else
    {
        mfem::Vector r(b.Size());
        // 2. Compute Residual r = b - A x
        A->Mult(x,r);
        r.Neg(); r+=b;
        // 3. Restrict to filtered subspace
        Pfiltered->MultTranspose(r,rf);
        // 4. Solve on the filtered subspace
        Mfiltered->Mult(rf,xf);
        // 5. Transfer to fine space
        Pfiltered->Mult(xf,z);
        // 6. Update Correction
        x+=z;
        // 7. Compute Residual r = b - A x
        A->Mult(x,r);
        r.Neg(); r+=b;
        // 8. Post V-Cycle 
        amg->Mult(r, z);
    }
    x+= z;
}

