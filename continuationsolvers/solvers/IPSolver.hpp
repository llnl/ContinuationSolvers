#include "mfem.hpp"
#include "../problems/OptProblems.hpp"


#ifndef PARIPSOLVER 
#define PARIPSOLVER

class InteriorPointSolver
{
protected:
    GeneralOptProblem* problem;
    double OptTol;
    int  max_iter;
    double mu_k; // \mu_k
    mfem::Vector lk, zlk;

    double sMax, kSig, tauMin, eta, thetaMin, delta, sTheta, sPhi, kMu, thetaMu;
    double thetaMax, kSoc, gTheta, gPhi, kEps;
	
    // filter
    mfem::Array<double> F1, F2;
	
    // quantities computed in lineSearch
    double alpha, alphaz;
    double thx0, thxtrial;
    double phx0, phxtrial;
    bool descentDirection, switchCondition, sufficientDecrease, lineSearchSuccess, inFilterRegion;
    double Dxphi0_xhat;

    int dimU, dimM, dimC;
    int dimUGlb, dimMGlb, dimCGlb;
    mfem::Array<int> block_offsetsumlz, block_offsetsuml, block_offsetsx;
    mfem::Vector ml;

    mfem::HypreParMatrix * Huu, * Hum, * Hmu, * Hmm, * Wmm, *D, * Ju, * Jm, * JuT, * JmT;
   
    mfem::Solver * linSolver; 
    int jOpt;
    bool converged;
    
    int MyRank;
    bool iAmRoot;

    bool saveLogBarrierIterates;
    bool savedLogBarrierSol;
    double muLogBarrierSol;
    mfem::Vector uLogBarrierSol, mLogBarrierSol, lLogBarrierSol, zlLogBarrierSol;

    bool initializedm;
    bool initializedl;
    bool initializedzl;
    mfem::Vector minit, linit, zlinit;
    double linSolveTol;
    std::ostream * ipout = &std::cout;
public:
    InteriorPointSolver(GeneralOptProblem*);
    double MaxStepSize(mfem::Vector& , mfem::Vector& , mfem::Vector& , double);
    double MaxStepSize(mfem::Vector& , mfem::Vector& , double);
    void Mult(const mfem::BlockVector& , mfem::BlockVector&);
    void Mult(const mfem::Vector&, mfem::Vector &); 
    void GetLagrangeMultiplier(mfem::Vector &);
    void FormIPNewtonMat(mfem::BlockVector& , mfem::Vector& , mfem::Vector& , mfem::BlockOperator &);
    void IPNewtonSolve(mfem::BlockVector& , mfem::Vector& , mfem::Vector& , mfem::Vector&, mfem::BlockVector& , double);
    void lineSearch(mfem::BlockVector& , mfem::BlockVector& , double);
    void projectZ(const mfem::Vector & , mfem::Vector &, double);
    void filterCheck(double, double);
    double E(const mfem::BlockVector &, const mfem::Vector &, const mfem::Vector &, double, bool);
    double E(const mfem::BlockVector &, const mfem::Vector &, const mfem::Vector &, bool);
    bool GetConverged() const;
    double theta(const mfem::BlockVector &, int &);
    double phi(const mfem::BlockVector &, double, int &);
    double theta(const mfem::BlockVector &);
    double phi(const mfem::BlockVector &, double);
    void Dxphi(const mfem::BlockVector &, double, mfem::BlockVector &);
    double L(const mfem::BlockVector &, const mfem::Vector &, const mfem::Vector &);
    void DxL(const mfem::BlockVector &, const mfem::Vector &, const mfem::Vector &, mfem::BlockVector &);
    void SetTol(double);
    void SetMaxIter(int);
    void SetBarrierParameter(double);
    void GetNumIterations(int &);    
    void SaveLogBarrierHessianIterates(bool);
    void SetLinearSolveTol(double);
    void InitializeM(mfem::Vector &);
    void InitializeL(mfem::Vector &);
    void InitializeZl(mfem::Vector &);
    void GetLogBarrierU(mfem::Vector &);
    void GetLogBarrierM(mfem::Vector &);
    void GetLogBarrierL(mfem::Vector &);
    void GetLogBarrierZl(mfem::Vector &);
    void GetLogBarrierMu(double &);
    void SetLogBarrierMu(double);
    void SetOutputStream(std::ostream * ipout_)
    {
       ipout = ipout_;
    };
    void SetLinearSolver(mfem::Solver &solver_) { linSolver = &(solver_); };
    virtual ~InteriorPointSolver();
};

#endif
