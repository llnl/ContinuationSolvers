#include "IPSolver.hpp"
#ifdef MFEM_USE_STRUMPACK
#include <StrumpackOptions.hpp>
#include <mfem/linalg/strumpack.hpp>
#endif

InteriorPointSolver::InteriorPointSolver(GeneralOptProblem* problem_)
    : problem(problem_),
      block_offsetsumlz(5),
      block_offsetsuml(4),
      block_offsetsx(3),
      Ju(nullptr),
      Jm(nullptr),
      linSolver(nullptr),
      saveLogBarrierIterates(false)
{
  OptTol = 1.e-2;
  max_iter = 20;
  mu_k = 1.0;

  sMax = 1.e2;
  kSig = 1.e10;      // control deviation from primal Hessian
  tauMin = 0.99;     // control rate at which iterates can approach the boundary
  eta = 1.e-4;       // backtracking constant
  thetaMin = 1.e-4;  // allowed violation of the equality constraints

  // constants in line-step A-5.4
  delta = 1.0;
  sTheta = 1.1;
  sPhi = 2.3;

  // control the rate at which the penalty parameter is decreased
  kMu = 0.2;
  thetaMu = 1.5;

  thetaMax = 1.e6;  // maximum constraint violation
  // data for the second order correction
  kSoc = 0.99;

  // equation (18)
  gTheta = 1.e-5;
  gPhi = 1.e-5;

  kEps = 1.e1;

  dimU = problem->GetDimU();
  dimM = problem->GetDimM();
  dimC = problem->GetDimC();

  block_offsetsumlz[0] = 0;
  block_offsetsumlz[1] = dimU;  // u
  block_offsetsumlz[2] = dimM;  // m
  block_offsetsumlz[3] = dimC;  // lambda
  block_offsetsumlz[4] = dimM;  // zl
  block_offsetsumlz.PartialSum();

  MPI_Allreduce(&dimU, &dimUGlb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dimM, &dimMGlb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dimC, &dimCGlb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  for (int i = 0; i < block_offsetsuml.Size(); i++) {
    block_offsetsuml[i] = block_offsetsumlz[i];
  }
  for (int i = 0; i < block_offsetsx.Size(); i++) {
    block_offsetsx[i] = block_offsetsuml[i];
  }

  ml = problem->Getml();

  lk.SetSize(dimC);
  lk = 0.0;
  zlk.SetSize(dimM);
  zlk = 0.0;

  Mulump.SetSize(dimU);
  Mmlump.SetSize(dimM);
  Mclump.SetSize(dimC);
  problem->getLumpedMassWeights(Mulump, Mmlump, Mclump);
  savedLogBarrierSol = false;
  muLogBarrierSol = 1.e-4;
  uLogBarrierSol.SetSize(dimU);
  mLogBarrierSol.SetSize(dimM);
  lLogBarrierSol.SetSize(dimC);
  zlLogBarrierSol.SetSize(dimM);
  initializedm = false;
  initializedl = false;
  initializedzl = false;
  minit.SetSize(dimM);
  linit.SetSize(dimC);
  zlinit.SetSize(dimM);

  MyRank = mfem::Mpi::WorldRank();
  iAmRoot = MyRank == 0 ? true : false;
}

double InteriorPointSolver::MaxStepSize(mfem::Vector& x, mfem::Vector& xl, mfem::Vector& xhat, double tau)
{
  double alphaMaxloc = 1.0;
  double alphaTmp;
  for (int i = 0; i < x.Size(); i++) {
    if (xhat(i) < 0.) {
      alphaTmp = -1. * tau * (x(i) - xl(i)) / xhat(i);
      alphaMaxloc = std::min(alphaMaxloc, alphaTmp);
    }
  }

  // alphaMaxloc is the local maximum step size which is
  // distinct on each MPI process. Need to compute
  // the global maximum step size
  double alphaMaxglb;
  MPI_Allreduce(&alphaMaxloc, &alphaMaxglb, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  return alphaMaxglb;
}

double InteriorPointSolver::MaxStepSize(mfem::Vector& x, mfem::Vector& xhat, double tau)
{
  mfem::Vector zero(x.Size());
  zero = 0.0;
  return MaxStepSize(x, zero, xhat, tau);
}

void InteriorPointSolver::Mult(const mfem::Vector& x0, mfem::Vector& xf)
{
  mfem::BlockVector x0block(block_offsetsx);
  x0block = 0.0;
  x0block.GetBlock(0).Set(1.0, x0);
  if (!(initializedm)) {
    OptProblem* optproblem = dynamic_cast<OptProblem*>(problem);
    //    if (dimM > 0)
    {
      if (!optproblem) {
        // fixed initialization
        x0block.GetBlock(1) = 1.e2;
        x0block.GetBlock(1).Add(1.0, ml);
      } else {
        // use c(u, m) = g(u) - m
        // if it is possible (with respect to the constraint m >= min_minit)
        // choose m such that c(u, m) = 0
        double min_minit = 1.0;
        x0block.GetBlock(1) = 0.0;
        x0block.GetBlock(1).Add(1.0, ml);
        mfem::Vector c0(dimC);
        c0 = 0.0;
        problem->c(x0block, c0);
        mfem::Vector dm(dimM);
        dm = 0.0;
        for (int i = 0; i < dimM; i++) {
          dm(i) = std::max(min_minit - x0block(dimU + i), c0(i));
        }
        x0block.GetBlock(1).Add(1.0, dm);
      }
    }
  }
  mfem::BlockVector xfblock(block_offsetsx);
  xfblock = 0.0;
  Mult(x0block, xfblock);
  xf.Set(1.0, xfblock.GetBlock(0));
}

void InteriorPointSolver::Mult(const mfem::BlockVector& x0, mfem::BlockVector& xf)
{
  converged = false;

  mfem::BlockVector xk(block_offsetsx), xhat(block_offsetsx);
  xk = 0;
  xhat = 0.0;
  mfem::BlockVector Xk(block_offsetsumlz), Xhat(block_offsetsumlz);
  Xk = 0.0;
  Xhat = 0.0;
  mfem::BlockVector Xhatuml(block_offsetsuml);
  Xhatuml = 0.0;
  mfem::Vector zlhat(dimM);
  zlhat = 0.0;

  xk.GetBlock(0).Set(1.0, x0.GetBlock(0));
  xk.GetBlock(1).Set(1.0, x0.GetBlock(1));

  // running estimate of the final values of the Lagrange multipliers
  if (initializedl) {
    lk.Set(1.0, linit);
  } else {
    lk = 0.0;
  }
  if (initializedzl) {
    zlk.Set(1.0, zlinit);
  } else {
    for (int i = 0; i < dimM; i++) {
      zlk(i) = 1.e1 * mu_k / (xk(i + dimU) - ml(i));
    }
  }

  Xk.GetBlock(0).Set(1.0, xk.GetBlock(0));
  Xk.GetBlock(1).Set(1.0, xk.GetBlock(1));
  Xk.GetBlock(2).Set(1.0, lk);
  Xk.GetBlock(3).Set(1.0, zlk);

  /* set theta0 = theta(x0)
   *     thetaMin
   *     thetaMax
   * when theta(xk) < thetaMin and the switching condition holds
   * then we ask for the Armijo sufficient decrease of the barrier
   * objective to be satisfied, in order to accept the trial step length alphakl
   *
   * thetaMax controls how the filter is initialized for each log-barrier
   * subproblem F0 = {(th, phi) s.t. th > thetaMax} that is the filter does not
   * allow for iterates where the constraint violation is larger than that of
   * thetaMax
   */
  double theta0 = theta(xk);
  thetaMin = 1.e-4 * std::max(1.0, theta0);
  thetaMax = 1.e4 * std::max(1.0, theta0);

  double Eeval, maxBarrierSolves, Eevalmu0;
  bool printOptimalityError;  // control optimality error print to console for
                              // log-barrier subproblems

  maxBarrierSolves = 10;

  for (jOpt = 0; jOpt < max_iter; jOpt++) {
    if (iAmRoot && printLevel > 1) {
      *ipout << "IP solver iteration#" << jOpt << std::endl;
    }
    // A-2. Check convergence of overall optimization problem
    printOptimalityError = false;
    Eevalmu0 = E(xk, lk, zlk, printOptimalityError);
    if (Eevalmu0 < OptTol) {
      converged = true;
      if (iAmRoot && printLevel > 0) {
        *ipout << "solved optimization problem to absolute tolerance " << OptTol << "\n";
      }
      break;
    }

    if (jOpt > 0) {
      maxBarrierSolves = 1;
    }

    for (int i = 0; i < maxBarrierSolves; i++) {
      // Check convergence of the barrier subproblem
      printOptimalityError = true;
      Eeval = E(xk, lk, zlk, mu_k, printOptimalityError);
      if (iAmRoot && printLevel > 1) {
        *ipout << "Optimality error = " << Eevalmu0 << std::endl;
        *ipout << "Subproblem error = " << Eeval << ", mu = " << mu_k << std::endl;
      }
      if (Eeval < kEps * mu_k) {
        // Update the barrier parameter
        double mu_k_new = UpdateBarrierParameter(mu_k);
        if (mu_k_new < muLogBarrierSol && !(savedLogBarrierSol)) {
          uLogBarrierSol.Set(1.0, xk.GetBlock(0));
          mLogBarrierSol.Set(1.0, xk.GetBlock(1));
          lLogBarrierSol.Set(1.0, lk);
          zlLogBarrierSol.Set(1.0, zlk);
          savedLogBarrierSol = true;
          muLogBarrierSol = mu_k;
        }
        if (iAmRoot && printLevel > 2) {
          *ipout << "solved barrier subproblem, for mu = " << mu_k << std::endl;
          *ipout << "updating barrier parameter to " << mu_k_new << std::endl;
        }
        if (mu_k != mu_k_new) {
          // Re-initialize the filter for the new subproblem
          F1.DeleteAll();
          F2.DeleteAll();
          mu_k = mu_k_new;
        } else if (iAmRoot && printLevel > 2) {
          *ipout << "not re-initializing the filter " << std::endl;
        }
      } else {
        break;
      }
    }

    // Compute the IP-Newton search direction
    // solve for (uhat, mhat, lhat)
    if (iAmRoot && printLevel > 3) {
      *ipout << "\n** IP-Newton solve **\n";
    }
    zlhat = 0.0;
    Xhatuml = 0.0;
    bool passedCurvatureTest = false;
    IPNewtonSolve(xk, lk, zlk, zlhat, Xhatuml, mu_k, passedCurvatureTest);
    if (!passedCurvatureTest) {
      if (iAmRoot && printLevel > 3) {
        *ipout << "curvature test failed\n";
      }
      int maxCurvatureTests = 30;
      double deltaReg = 0.0;
      if (deltaRegLast < deltaRegMin) {
        deltaReg = deltaReg0;
      } else {
        deltaReg = std::max(deltaRegMin, kRegMinus * deltaRegLast);
      }

      // solve regularized IP-Newton linear system
      if (iAmRoot && printLevel > 3) {
        *ipout << "deltaReg = " << deltaReg << "\n";
      }
      zlhat = 0.0;
      Xhatuml = 0.0;
      IPNewtonSolve(xk, lk, zlk, zlhat, Xhatuml, mu_k, passedCurvatureTest, deltaReg);
      for (int numCurvatureTests = 0; numCurvatureTests < maxCurvatureTests; numCurvatureTests++) {
        if (passedCurvatureTest) {
          deltaRegLast = deltaReg;
          break;
        } else {
          if (deltaRegLast < deltaRegMin) {
            deltaReg *= kRegBarPlus;
          } else {
            deltaReg *= kRegPlus;
          }
        }
        // solve regularized IP-Newton linear system with updated regularization
        // param
        if (iAmRoot && printLevel > 3) {
          *ipout << "deltaReg = " << deltaReg << "\n";
        }
        zlhat = 0.0;
        Xhatuml = 0.0;
        IPNewtonSolve(xk, lk, zlk, zlhat, Xhatuml, mu_k, passedCurvatureTest, deltaReg);
      }
    }
    mu_history.Append(mu_k);

    // assign data stack, X = (u, m, l, zl)
    Xk = 0.0;
    Xk.GetBlock(0).Set(1.0, xk.GetBlock(0));
    Xk.GetBlock(1).Set(1.0, xk.GetBlock(1));
    Xk.GetBlock(2).Set(1.0, lk);
    Xk.GetBlock(3).Set(1.0, zlk);

    // assign data stack, Xhat = (uhat, mhat, lhat, zlhat)
    Xhat = 0.0;
    for (int i = 0; i < 3; i++) {
      Xhat.GetBlock(i).Set(1.0, Xhatuml.GetBlock(i));
    }
    Xhat.GetBlock(3).Set(1.0, zlhat);

    // Backtracking filter line search.
    if (iAmRoot && printLevel > 3) {
      *ipout << "\n** Linesearch **\n";
      *ipout << "mu = " << mu_k << std::endl;
    }
    lineSearch(Xk, Xhat, mu_k);

    if (lineSearchSuccess) {
      if (iAmRoot && printLevel > 3) {
        *ipout << "Linesearch successful\n";
      }
      if (!switchCondition || !sufficientDecrease) {
        F1.Append((1. - gTheta) * thx0);
        F2.Append(phx0 - gPhi * thx0);
      }
      // ----- A-6: Accept the trial point
      xk.GetBlock(0).Add(alpha, Xhat.GetBlock(0));
      xk.GetBlock(1).Add(alpha, Xhat.GetBlock(1));
      lk.Add(alpha, Xhat.GetBlock(2));
      zlk.Add(alphaz, Xhat.GetBlock(3));
      projectZ(xk, zlk, mu_k);
    } else {
      if (iAmRoot && printLevel > 3) {
        *ipout << "Linesearch not successful\n";
        *ipout << "attempting feasibility restoration with theta = " << thx0 << std::endl;
        *ipout << "no feasibility restoration implemented, exiting now \n";
      }
      break;
    }
    if (iAmRoot && jOpt + 1 == max_iter && printLevel > 3) {
      *ipout << "maximum optimization iterations reached, exiting without "
                "converging\n";
    }
  }
  // done with optimization routine, just reassign data to xf reference so
  // that the application code has access to the optimal point
  xf = 0.0;
  xf.GetBlock(0).Set(1.0, xk.GetBlock(0));
  xf.GetBlock(1).Set(1.0, xk.GetBlock(1));
}

void InteriorPointSolver::FormIPNewtonMat(mfem::BlockVector& x, mfem::Vector& l, mfem::Vector& zl,
                                          mfem::BlockOperator& Ak, const double& delta)
{
  // MFEM_VERIFY(!fullLagrangianHessian,
  //             "only supporting partial Lagrangian of the Hessian");
  auto Huuf = dynamic_cast<mfem::HypreParMatrix*>(problem->Duuf(x));
  auto Humf = dynamic_cast<mfem::HypreParMatrix*>(problem->Dumf(x));
  auto Hmuf = dynamic_cast<mfem::HypreParMatrix*>(problem->Dmuf(x));
  auto Hmmf = dynamic_cast<mfem::HypreParMatrix*>(problem->Dmmf(x));

  mfem::HypreParMatrix* Huucl = nullptr;
  mfem::HypreParMatrix* Humcl = nullptr;
  mfem::HypreParMatrix* Hmucl = nullptr;
  mfem::HypreParMatrix* Hmmcl = nullptr;
  if (fullLagrangianHessian) {
    Huucl = dynamic_cast<mfem::HypreParMatrix*>(problem->Duucl(x, l));
    Humcl = dynamic_cast<mfem::HypreParMatrix*>(problem->Dumcl(x, l));
    Hmucl = dynamic_cast<mfem::HypreParMatrix*>(problem->Dmucl(x, l));
    Hmmcl = dynamic_cast<mfem::HypreParMatrix*>(problem->Dmmcl(x, l));
  }

  mfem::Vector duu(dimU);
  duu.Set(1.0, Mulump);
  duu *= (hessRegularization + delta);
  // duu = hessRegularization + delta;
  std::unique_ptr<mfem::HypreParMatrix> Duu;
  Duu.reset(GenerateHypreParMatrixFromDiagonal(problem->GetDofOffsetsU(), duu));
  std::unique_ptr<mfem::HypreParMatrix> Huu2;
  // (0,0), Hessian block Hmm
  if (!fullLagrangianHessian || !Huucl) {
    if (Huuf) {
      Huu.reset(ParAdd(Huuf, Duu.get()));
    } else {
      Huu.reset(new mfem::HypreParMatrix(*Duu.get()));
    }
  } else {
    if (Huuf) {
      Huu2.reset(ParAdd(Huucl, Duu.get()));
      Huu.reset(ParAdd(Huuf, Huu2.get()));
    } else {
      Huu.reset(ParAdd(Huucl, Duu.get()));
    }
  }

  mfem::Vector DiagLogBar(dimM);
  DiagLogBar = 0.0;
  for (int ii = 0; ii < dimM; ii++) {
    DiagLogBar(ii) = zl(ii) / (x(ii + dimU) - ml(ii)) + (delta + hessRegularization) * Mmlump(ii);
  }
  std::unique_ptr<mfem::HypreParMatrix> Dmm;
  Dmm.reset(GenerateHypreParMatrixFromDiagonal(problem->GetDofOffsetsM(), DiagLogBar));
  std::unique_ptr<mfem::HypreParMatrix> Hmm2;
  // (1,1), Hessian block Hmm
  if (!fullLagrangianHessian || !Hmmcl) {
    if (Hmmf) {
      Hmm.reset(ParAdd(Hmmf, Dmm.get()));
    } else {
      Hmm.reset(new mfem::HypreParMatrix(*Dmm.get()));
    }
  } else {
    if (Hmmf) {
      Hmm2.reset(ParAdd(Hmmcl, Dmm.get()));
      Hmm.reset(ParAdd(Hmmf, Hmm2.get()));
    } else {
      Hmm.reset(ParAdd(Hmmcl, Dmm.get()));
    }
  }

  // (0,1) Hessian block, Hum
  if (!fullLagrangianHessian || !Humcl) {
    if (Humf) {
      Hum.reset(new mfem::HypreParMatrix(*Humf));  // deep-copy
    } else {
      Hum.reset(nullptr);  // deep-copy
    }
  } else {
    if (Humf) {
      Hum.reset(ParAdd(Humf, Humcl));
    } else {
      Hum.reset(new mfem::HypreParMatrix(*Humcl));
    }
  }

  // (1, 0) Hessian block, Hmu
  if (!fullLagrangianHessian || !Hmucl) {
    if (Hmuf) {
      Hmu.reset(new mfem::HypreParMatrix(*Hmuf));  // deep-copy
    } else {
      Hmu.reset(nullptr);  // deep-copy
    }
  } else {
    if (Hmuf) {
      Hmu.reset(ParAdd(Hmuf, Hmucl));
    } else {
      Hmu.reset(new mfem::HypreParMatrix(*Hmucl));
    }
  }
  Ju = dynamic_cast<mfem::HypreParMatrix*>(problem->Duc(x));
  JuT.reset(Ju->Transpose());
  Jm = dynamic_cast<mfem::HypreParMatrix*>(problem->Dmc(x));
  JmT.reset(Jm->Transpose());
  //         IP-Newton system matrix
  //    Ak = [[H_(u,u)  H_(u,m)   J_u^T]
  //          [H_(m,u)  W_(m,m)   J_m^T]
  //          [ J_u      J_m       0  ]]

  Ak.SetBlock(0, 0, Huu.get());
  Ak.SetBlock(0, 2, JuT.get());
  Ak.SetBlock(1, 1, Hmm.get());
  Ak.SetBlock(1, 2, JmT.get());
  Ak.SetBlock(2, 0, Ju);
  Ak.SetBlock(2, 1, Jm);
  if (Hum.get()) {
    Ak.SetBlock(0, 1, Hum.get());
    Ak.SetBlock(1, 0, Hmu.get());
  }
}

// perturbed KKT system solve
// determine the search direction
void InteriorPointSolver::IPNewtonSolve(mfem::BlockVector& x, mfem::Vector& l, mfem::Vector& zl, mfem::Vector& zlhat,
                                        mfem::BlockVector& Xhat, const double& mu, bool& passedCurvatureTest,
                                        const double& delta)
{
  bool linSolveConvergence = false;
  int nKrylovIts = -1;
  // solve A x = b, where A is the IP-Newton matrix
  mfem::BlockOperator A(block_offsetsuml, block_offsetsuml);
  mfem::BlockVector b(block_offsetsuml);
  b = 0.0;
  FormIPNewtonMat(x, l, zl, A, delta);

  //       [grad_u phi + Ju^T l]
  // b = - [grad_m phi + Jm^T l]
  //       [          c        ]
  mfem::BlockVector gradphi(block_offsetsx);
  gradphi = 0.0;
  mfem::BlockVector JTl(block_offsetsx);
  JTl = 0.0;
  Dxphi(x, mu, gradphi);

  (A.GetBlock(0, 2)).Mult(l, JTl.GetBlock(0));
  (A.GetBlock(1, 2)).Mult(l, JTl.GetBlock(1));

  for (int ii = 0; ii < 2; ii++) {
    b.GetBlock(ii).Set(1.0, gradphi.GetBlock(ii));
    b.GetBlock(ii).Add(1.0, JTl.GetBlock(ii));
  }
  problem->c(x, b.GetBlock(2));
  b *= -1.0;
  Xhat = 0.0;

  if (linSolver) {
    linSolver->SetOperator(A);
    linSolver->Mult(b, Xhat);
  } else  // default direct solver
  {
    mfem::Array2D<const mfem::HypreParMatrix*> ABlockMatrix(3, 3);
    for (int ii = 0; ii < 3; ii++) {
      for (int jj = 0; jj < 3; jj++) {
        if (!A.IsZeroBlock(ii, jj)) {
          ABlockMatrix(ii, jj) =
              dynamic_cast<mfem::HypreParMatrix*>(const_cast<mfem::Operator*>(&(A.GetBlock(ii, jj))));
        } else {
          ABlockMatrix(ii, jj) = nullptr;
        }
      }
    }
    std::unique_ptr<mfem::HypreParMatrix> Ah;
    Ah.reset(HypreParMatrixFromBlocks(ABlockMatrix));
    /* direct solve of the 3x3 IP-Newton linear system */
    linSolver = new DirectSolver(*Ah.get());
    linSolver->Mult(b, Xhat);
    delete linSolver;
    linSolver = nullptr;
  }

  {
    mfem::BlockVector residual(block_offsetsuml);
    residual = 0.0;
    A.Mult(Xhat, residual);
    residual.Add(-1.0, b);
    double res_norm = mfem::GlobalLpNorm(2, residual.Norml2(), MPI_COMM_WORLD);
    double relative_res_norm = res_norm / mfem::GlobalLpNorm(2, b.Norml2(), MPI_COMM_WORLD);
    if (iAmRoot && checkLinearSysResiduals) {
      *ipout << "|| A x - b ||_2 = " << res_norm << "\n";
      *ipout << "|| A x - b ||_2 / || b ||_2 = " << relative_res_norm << "\n";
    }
    if (res_norm > 1.e-4 || relative_res_norm > 1.e-4) {
      linSolveConvergence = false;
    } else {
      linSolveConvergence = true;
    }
  }

  passedCurvatureTest = CurvatureTest(A, Xhat, l, b, delta);
  if (iAmRoot && printLevel > 3) {
    *ipout << "curvature check passed? " << passedCurvatureTest << " \n";
    *ipout << "linear solver converged? " << linSolveConvergence << " \n";
  }
  if (!linSolveConvergence) {
    passedCurvatureTest = linSolveConvergence;
  }

  /* backsolve to determine zlhat */
  for (int ii = 0; ii < dimM; ii++) {
    zlhat(ii) = -1. * (zl(ii) + (zl(ii) * Xhat(ii + dimU) - mu) / (x(ii + dimU) - ml(ii)));
  }
}

// here Xhat, X will be mfem::BlockVectors w.r.t. the 4 partitioning X = (u, m,
// l, zl)

void InteriorPointSolver::lineSearch(mfem::BlockVector& X0, mfem::BlockVector& Xhat, double mu)
{
  // double tau  = max(tauMin, 1.0 - mu);
  double tau = tauMin;
  mfem::Vector u0 = X0.GetBlock(0);
  mfem::Vector m0 = X0.GetBlock(1);
  mfem::Vector l0 = X0.GetBlock(2);
  mfem::Vector z0 = X0.GetBlock(3);
  mfem::Vector uhat = Xhat.GetBlock(0);
  mfem::Vector mhat = Xhat.GetBlock(1);
  mfem::Vector lhat = Xhat.GetBlock(2);
  mfem::Vector zhat = Xhat.GetBlock(3);
  double alphaMax = MaxStepSize(m0, ml, mhat, tau);
  double alphaMaxz = MaxStepSize(z0, zhat, tau);
  alphaz = alphaMaxz;

  mfem::BlockVector x0(block_offsetsx);
  x0 = 0.0;
  x0.GetBlock(0).Set(1.0, u0);
  x0.GetBlock(1).Set(1.0, m0);

  mfem::BlockVector xhat(block_offsetsx);
  xhat = 0.0;
  xhat.GetBlock(0).Set(1.0, uhat);
  xhat.GetBlock(1).Set(1.0, mhat);

  mfem::BlockVector xtrial(block_offsetsx);
  xtrial = 0.0;
  mfem::BlockVector Dxphi0(block_offsetsx);
  Dxphi0 = 0.0;
  int maxBacktrack = 20;
  alpha = alphaMax;

  mfem::Vector ck0(dimC);
  ck0 = 0.0;
  mfem::Vector zhatsoc(dimM);
  zhatsoc = 0.0;
  mfem::BlockVector Xhatumlsoc(block_offsetsuml);
  Xhatumlsoc = 0.0;
  mfem::BlockVector xhatsoc(block_offsetsx);
  xhatsoc = 0.0;
  mfem::Vector uhatsoc(dimU);
  uhatsoc = 0.0;
  mfem::Vector mhatsoc(dimM);
  mhatsoc = 0.0;

  Dxphi(x0, mu, Dxphi0);

  int th_eval_err;
  int ph_eval_err;
  Dxphi0_xhat = InnerProduct(MPI_COMM_WORLD, Dxphi0, xhat);
  double xhat_norm = sqrt(InnerProduct(MPI_COMM_WORLD, xhat, xhat));
  double Dxphi0_norm = sqrt(InnerProduct(MPI_COMM_WORLD, Dxphi0, Dxphi0));
  descentDirection = Dxphi0_xhat < 0. ? true : false;
  if (iAmRoot && printLevel > 3) {
    *ipout << "grad(phi)^T xhat / (||grad(phi)|| * ||xhat||) = " << Dxphi0_xhat / (xhat_norm * Dxphi0_norm)
           << std::endl;
    *ipout << "|grad(phi)^T xhat| = " << abs(Dxphi0_xhat) << std::endl;
    if (descentDirection) {
      *ipout << "is a descent direction for the log-barrier objective\n";
    } else {
      *ipout << "is not a descent direction for the log-barrier objective\n";
    }
  }
  thx0 = theta(x0);
  phx0 = phi(x0, mu);

  lineSearchSuccess = false;
  for (int i = 0; i < maxBacktrack; i++) {
    if (iAmRoot && printLevel > 3) {
      *ipout << "\n linesearch step length = " << alpha << " ---------\n";
    }
    // ----- Compute trial point: xtrial = x0 + alpha_i xhat
    xtrial.Set(1.0, x0);
    xtrial.Add(alpha, xhat);

    // ------ If not in filter region go to A.5.4 otherwise go to A-5.5.
    thxtrial = theta(xtrial, th_eval_err);
    phxtrial = phi(xtrial, mu, ph_eval_err);
    if (iAmRoot && printLevel > 3) {
      *ipout << "| grad(phi)^xhat - (phi(x0 + alpha xhat) - phi(x0)) / alpha | = "
             << abs(Dxphi0_xhat - (phxtrial - phx0) / alpha) << ", alpha = " << alpha << std::endl;
    }

    if (!(th_eval_err == 0 && ph_eval_err == 0)) {
      if (iAmRoot && printLevel > 3) {
        *ipout << "BAD STEP: reducing step length\n";
      }
      alpha *= 0.5;
      continue;
    }
    filterCheck(thxtrial, phxtrial);
    if (!inFilterRegion) {
      if (iAmRoot && printLevel > 3) {
        *ipout << "not in filter region :)\n";
      }
      // ------ Check sufficient decrease
      if (!descentDirection) {
        switchCondition = false;
      } else {
        switchCondition = (alpha * pow(abs(Dxphi0_xhat), sPhi) > delta * pow(thx0, sTheta)) ? true : false;
      }
      if (iAmRoot && printLevel > 3) {
        *ipout << "theta(x0) = " << thx0 << ", thetaMin = " << thetaMin << std::endl;
        *ipout << "theta(xtrial) = " << thxtrial << ", (1-gTheta) *theta(x0) = " << (1. - gTheta) * thx0 << std::endl;
        *ipout << "phi(xtrial) = " << phxtrial << ", phi(x0) = " << phx0
               << ", phi(x0) - gPhi *theta(x0) = " << phx0 - gPhi * thx0 << std::endl;
      }
      // Case I
      if (thx0 <= thetaMin && switchCondition) {
        sufficientDecrease = (phxtrial <= phx0 + eta * alpha * Dxphi0_xhat) ? true : false;
        if (sufficientDecrease) {
          if (iAmRoot && printLevel > 3) {
            *ipout << "Line search successful: sufficient decrease in "
                      "log-barrier objective.\n";
          }
          // accept the trial step
          lineSearchSuccess = true;
          break;
        } else {
          if (iAmRoot && printLevel > 3) {
            *ipout << "sufficient decrease not achieved in log-barrier "
                      "objective.\n";
          }
        }
      } else {
        if (thxtrial <= (1. - gTheta) * thx0 || phxtrial <= phx0 - gPhi * thx0) {
          if (iAmRoot && printLevel > 3) {
            *ipout << "Line search successful: infeasibility or log-barrier "
                      "objective decreased.\n";
          }
          // accept the trial step
          lineSearchSuccess = true;
          break;
        }
      }
    } else {
      if (iAmRoot && printLevel > 3) {
        *ipout << "in filter region, reducing step length\n";
      }
    }
    alpha *= 0.5;
  }
}

void InteriorPointSolver::projectZ(const mfem::Vector& x, mfem::Vector& z, double mu)
{
  double zi;
  double mudivmml;
  for (int i = 0; i < dimM; i++) {
    zi = z(i);
    mudivmml = mu / (x(i + dimU) - ml(i));
    z(i) = std::max(std::min(zi, kSig * mudivmml), mudivmml / kSig);
  }
}

void InteriorPointSolver::filterCheck(double th, double ph)
{
  inFilterRegion = false;
  if (th > thetaMax) {
    inFilterRegion = true;
  } else {
    for (int i = 0; i < F1.Size(); i++) {
      if (th >= F1[i] && ph >= F2[i]) {
        inFilterRegion = true;
        break;
      }
    }
  }
}

double InteriorPointSolver::E(const mfem::BlockVector& x, const mfem::Vector& l, const mfem::Vector& zl, double mu,
                              bool printEeval)
{
  double E1, E2, E3;
  double sc, sd;
  mfem::BlockVector gradL(block_offsetsx);
  gradL = 0.0;  // stationarity grad L = grad f + J^T l - z
  mfem::Vector cx(dimC);
  cx = 0.0;  // feasibility c = c(x)
  mfem::Vector comp(dimM);
  comp = 0.0;  // complementarity M Z - mu 1

  DxL(x, l, zl, gradL);
  if (massWeightedNorms) {
    mfem::BlockVector MinvgradL(block_offsetsx);
    MinvgradL.Set(1.0, gradL);
    MinvgradL.GetBlock(0) /= Mulump;
    MinvgradL.GetBlock(1) /= Mmlump;
    E1 = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, MinvgradL, gradL));
  } else {
    E1 = mfem::GlobalLpNorm(mfem::infinity(), gradL.Normlinf(), MPI_COMM_WORLD);
  }
  problem->c(x, cx);
  if (massWeightedNorms) {
    mfem::Vector Mcx(dimC);
    Mcx = 0.0;
    Mcx.Set(1.0, cx);
    Mcx *= Mclump;
    E2 = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, Mcx, cx));
  } else {
    E2 = mfem::GlobalLpNorm(mfem::infinity(), cx.Normlinf(), MPI_COMM_WORLD);
  }
  for (int ii = 0; ii < dimM; ii++) {
    comp(ii) = x(dimU + ii) * zl(ii) - mu;
  }
  E3 = mfem::GlobalLpNorm(mfem::infinity(), comp.Normlinf(), MPI_COMM_WORLD);
  if (iAmRoot && printEeval && printLevel > 2) {
    *ipout << "evaluating optimality error for mu = " << mu << std::endl;
    *ipout << "stationarity measure = " << E1 << std::endl;
    *ipout << "feasibility measure  = " << E2 << std::endl;
    *ipout << "complimentarity measure = " << E3 << std::endl;
  }
  return std::max(std::max(E1, E2), E3);
  // double ll1, zl1;

  // zl1 = mfem::GlobalLpNorm(1, zl.Norml1(), MPI_COMM_WORLD);
  // ll1 = mfem::GlobalLpNorm(1, l.Norml1(), MPI_COMM_WORLD);
  // sc = std::max(sMax, zl1 / (double(dimMGlb))) / sMax;
  // sd = std::max(sMax, (ll1 + zl1) / (double(dimCGlb + dimMGlb))) / sMax;
  // if (iAmRoot && printEeval && printLevel > 2) {
  //   *ipout << "evaluating optimality error for mu = " << mu << std::endl;
  //   *ipout << "stationarity measure = " << E1 / sd << std::endl;
  //   *ipout << "feasibility measure  = " << E2 << std::endl;
  //   *ipout << "complimentarity measure = " << E3 / sc << std::endl;
  // }
  // return std::max(std::max(E1 / sd, E2), E3 / sc);
}

double InteriorPointSolver::E(const mfem::BlockVector& x, const mfem::Vector& l, const mfem::Vector& zl,
                              bool printEeval)
{
  return E(x, l, zl, 0.0, printEeval);
}

double InteriorPointSolver::theta(const mfem::BlockVector& x, int& eval_err)
{
  mfem::Vector cx(dimC);
  cx = 0.0;
  problem->c(x, cx, eval_err);
  if (massWeightedNorms) {
    mfem::Vector Mcx(dimC);
    Mcx = 0.0;
    Mcx.Set(1.0, cx);
    Mcx *= Mclump;
    return sqrt(InnerProduct(MPI_COMM_WORLD, Mcx, cx));
  } else {
    return mfem::GlobalLpNorm(2, cx.Norml2(), MPI_COMM_WORLD);
  }
}

double InteriorPointSolver::theta(const mfem::BlockVector& x)
{
  int eval_err;  // throw away
  return theta(x, eval_err);
}

// log-barrier objective
double InteriorPointSolver::phi(const mfem::BlockVector& x, double mu, int& eval_err)
{
  double fx = problem->CalcObjective(x, eval_err);
  double logBarrierLoc = 0.0;
  for (int i = 0; i < dimM; i++) {
    logBarrierLoc += log(x(dimU + i) - ml(i));
  }
  double logBarrierGlb;
  MPI_Allreduce(&logBarrierLoc, &logBarrierGlb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return fx - mu * logBarrierGlb;
}

double InteriorPointSolver::phi(const mfem::BlockVector& x, double mu)
{
  int eval_err;  // throw away
  return phi(x, mu, eval_err);
}

// gradient of log-barrier objective with respect to x = (u, m)
void InteriorPointSolver::Dxphi(const mfem::BlockVector& x, double mu, mfem::BlockVector& y)
{
  problem->CalcObjectiveGrad(x, y);

  for (int i = 0; i < dimM; i++) {
    y(dimU + i) -= mu / (x(dimU + i));
  }
}

// Lagrangian function evaluation
// L(x, l, zl) = f(x) + l^T c(x) - zl^T m
double InteriorPointSolver::L(const mfem::BlockVector& x, const mfem::Vector& l, const mfem::Vector& zl)
{
  double fx = problem->CalcObjective(x);
  mfem::Vector cx(dimC);
  problem->c(x, cx);
  return (fx + InnerProduct(MPI_COMM_WORLD, cx, l) - InnerProduct(MPI_COMM_WORLD, x.GetBlock(1), zl));
}

void InteriorPointSolver::DxL(const mfem::BlockVector& x, const mfem::Vector& l, const mfem::Vector& zl,
                              mfem::BlockVector& y)
{
  // evaluate the gradient of the objective with respect to the primal variables
  // x = (u, m)
  mfem::BlockVector gradxf(block_offsetsx);
  gradxf = 0.0;
  problem->CalcObjectiveGrad(x, gradxf);

  mfem::HypreParMatrix *Jacu, *Jacm, *JacuT, *JacmT;
  Jacu = dynamic_cast<mfem::HypreParMatrix*>(problem->Duc(x));
  Jacm = dynamic_cast<mfem::HypreParMatrix*>(problem->Dmc(x));
  Jacu->MultTranspose(l, y.GetBlock(0));
  Jacm->MultTranspose(l, y.GetBlock(1));

  y.Add(1.0, gradxf);
  (y.GetBlock(1)).Add(-1.0, zl);
}

bool InteriorPointSolver::GetConverged() const { return converged; }

void InteriorPointSolver::SetTol(double Tol) { OptTol = Tol; }

void InteriorPointSolver::SetMaxIter(int max_it) { max_iter = max_it; }

double InteriorPointSolver::UpdateBarrierParameter(double mu)
{
  return std::max(OptTol / 10., std::min(kMu * mu, pow(mu, thetaMu)));
}

void InteriorPointSolver::SetBarrierParameter(double mu_0) { mu_k = mu_0; }

void InteriorPointSolver::SaveLogBarrierHessianIterates(bool save)
{
  MFEM_ASSERT(MyRank == 0 || save == false, "currently can only save logbarrier hessian in serial codes");
  saveLogBarrierIterates = save;
}

void InteriorPointSolver::GetLagrangeMultiplier(mfem::Vector& y)
{
  y.SetSize(dimM);
  y = 0.;
  y.Set(1.0, zlk);
}

void InteriorPointSolver::InitializeM(mfem::Vector& m0)
{
  minit.Set(1.0, m0);
  initializedm = true;
}

void InteriorPointSolver::InitializeL(mfem::Vector& l0)
{
  linit.Set(1.0, l0);
  initializedl = true;
}

void InteriorPointSolver::InitializeZl(mfem::Vector& z0)
{
  zlinit.Set(1.0, z0);
  initializedzl = true;
}

void InteriorPointSolver::GetLogBarrierU(mfem::Vector& uLogBar) { uLogBar.Set(1.0, uLogBarrierSol); }

void InteriorPointSolver::GetLogBarrierM(mfem::Vector& mLogBar) { mLogBar.Set(1.0, mLogBarrierSol); }

void InteriorPointSolver::GetLogBarrierL(mfem::Vector& lLogBar) { lLogBar.Set(1.0, lLogBarrierSol); }

void InteriorPointSolver::GetLogBarrierZl(mfem::Vector& zlLogBar) { zlLogBar.Set(1.0, zlLogBarrierSol); }

void InteriorPointSolver::GetNumIterations(int& its) { its = jOpt; }

void InteriorPointSolver::GetLogBarrierMu(double& mu) { mu = muLogBarrierSol; }

void InteriorPointSolver::SetLogBarrierMu(double mu) { muLogBarrierSol = mu; }

// curvature test
// dk^T Wk dk + max{ -(lk + lhat)^T ck, 0.0} >= alpha * dk^T dk
// see "An Inertia-Free Filter Line-search Algorithm for
// Large-scale Nonlinear Programming" by Nai-Yuan Chiang and
// Victor M Zavala, Computational Optimization and Applications (2016)
bool InteriorPointSolver::CurvatureTest(const mfem::BlockOperator& A, const mfem::BlockVector& Xhat,
                                        const mfem::Vector& l, const mfem::BlockVector& b, const double& delta)
{
  // current Lagrange multiplier + Lagrange multiplier update
  mfem::Vector lplus(dimC);
  lplus.Set(1.0, l);
  lplus.Add(1.0, Xhat.GetBlock(2));

  double dWd = 0.0;
  double dd = 0.0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (!A.IsZeroBlock(i, j)) {
        mfem::Vector temp(A.GetBlock(i, j).Height());
        temp = 0.0;
        A.GetBlock(i, j).Mult(Xhat.GetBlock(j), temp);
        dWd += mfem::InnerProduct(MPI_COMM_WORLD, Xhat.GetBlock(i), temp);
      }
    }
    if (i == 0) {
      mfem::Vector temp(dimU);
      temp.Set(1.0, Xhat.GetBlock(i));
      temp *= Mulump;
      dd += mfem::InnerProduct(MPI_COMM_WORLD, Xhat.GetBlock(i), temp);
    } else {
      mfem::Vector temp(dimM);
      temp.Set(1.0, Xhat.GetBlock(i));
      temp *= Mmlump;
      dd += mfem::InnerProduct(MPI_COMM_WORLD, Xhat.GetBlock(i), temp);
    }
    // dd +=
    //     mfem::InnerProduct(MPI_COMM_WORLD, Xhat.GetBlock(i), Xhat.GetBlock(i));
  }
  double lplusTck = -1.0 * mfem::InnerProduct(MPI_COMM_WORLD, lplus, b.GetBlock(2));

  bool passed = (dWd + std::fmax(-lplusTck, 0.0) >= alphaCurvatureTest * dd);
  return passed;
}

InteriorPointSolver::~InteriorPointSolver()
{
  F1.DeleteAll();
  F2.DeleteAll();
  block_offsetsx.DeleteAll();
  block_offsetsumlz.DeleteAll();
  block_offsetsuml.DeleteAll();
  ml.SetSize(0);
}
