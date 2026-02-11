#include "mfem.hpp"
#include "../problems/OptProblems.hpp"
#include "../utilities.hpp"
#include "AMGF.hpp"

#ifndef CONDENSEDHOMOTOPYSOLVER 
#define CONDENSEDHOMOTOPYSOLVER

class CondensedHomotopySolver : public mfem::Solver
{
protected:
	mfem::HypreParMatrix* Areduced = nullptr;
	mfem::Solver* AreducedSolver = nullptr;
        AMGF * amgf = nullptr;	
	mfem::HypreParMatrix * P = nullptr;
	mfem::Array<int> blockOffsets;
	mfem::Vector scale00;
	mfem::Vector scale01;
	mfem::Vector scale10;
	mfem::Vector scale11;
	const mfem::HypreParMatrix* A12;
	const mfem::HypreParMatrix* A20;
	bool use_amgf = true;
public:
	CondensedHomotopySolver() = default;

	/// Set the solver for the reduced system.
	void SetPreconditioner(mfem::Solver *solver) { AreducedSolver = (solver); };
	void SetPreconditioner(mfem::Solver &solver) { AreducedSolver = &(solver); };
	mfem::Solver* GetPreconditioner() { return AreducedSolver; };

	/**
	 * @brief Sets the linear system to be solved.
	 *
	 * This builds a reduced system and the blocks needed to form the reduced RHS.
	 * Also, it updates the solver for the reduced system.
	 * The reduced system is:
	 *     A22 + A20 * A00^{-1} * A01 * (A11 - A10 * A00^{-1} * A01)^{-1} * A12
	 * 
	 * @param op The operator is expected to be a BlockOperator of the form:
	 *                            [A00 A01 0  ]
	 *                            [A10 A11 A12]
	 *                            [A20 0   A22]
	 *           where A00, A01, A10, and A11 are diagonal matrices.
	 */
	void SetOperator(const mfem::Operator& op) override;

	void Mult(const mfem::Vector&, mfem::Vector &) const override; 
	void Mult(const mfem::BlockVector& , mfem::BlockVector&) const;
	void SetUseAMGF(bool use_amgf_) { use_amgf = use_amgf_; };

	mfem::Solver * GetReducedSolver() { return AreducedSolver; };
	virtual ~CondensedHomotopySolver();
};

#endif // CONDENSEDHOMOTOPYSOLVER
