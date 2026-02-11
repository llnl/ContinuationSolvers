#include "mfem.hpp"
#include "HomotopySolver.hpp"
#include "CondensedHomotopySolver.hpp"
#include "AMGF.hpp"


HomotopySolver::HomotopySolver(GeneralNLMCProblem * problem_) : problem(problem_), block_offsets_xsy(4), block_offsets_xy(3), filter(0), krylov_its(0), iterates(0)
{
   dimx = problem->GetDimx();
   dimy = problem->GetDimy();
   dimxglb = problem->GetDimxGlb();
   dimyglb = problem->GetDimyGlb();
   dofOffsetsx = problem->GetDofOffsetsx();
   dofOffsetsy = problem->GetDofOffsetsy();

   block_offsets_xsy[0] = 0;
   block_offsets_xsy[1] = dimx; // x
   block_offsets_xsy[2] = dimx; // s
   block_offsets_xsy[3] = dimy; // y
   block_offsets_xsy.PartialSum();

   block_offsets_xy[0] = 0;
   block_offsets_xy[1] = dimx;
   block_offsets_xy[2] = dimy;
   block_offsets_xy.PartialSum();

   gammax.SetSize(dimx); gammax = 1.0;
   gammay.SetSize(dimy); gammay = 1.0;
   ax.SetSize(dimx); ax = 1.0;
   bx.SetSize(dimx); bx = 1.e-6;
   cx.SetSize(dimx); cx = 0.0;
   cy.SetSize(dimy); cy = 0.0;

   converged = false; 
    
   MyRank = mfem::Mpi::WorldRank();
   iAmRoot = MyRank == 0 ? true : false;

}


void HomotopySolver::Mult(const mfem::Vector & X0, mfem::Vector & Xf)
{
   MFEM_VERIFY(X0.Size() == dimx + dimy && Xf.Size() == dimx + dimy, "one or more input vectors of incorrect size");
   mfem::Vector X0copy(dimx + dimy); // non-const copy
   X0copy.Set(1.0, X0);
   mfem::Vector x0(X0copy, 0, dimx);
   mfem::Vector y0(X0copy, dimx, dimy);
   mfem::Vector xf(Xf, 0, dimx);
   mfem::Vector yf(Xf, dimx, dimy);
   Mult(x0, y0, xf, yf);
}


void HomotopySolver::Mult(const mfem::Vector & x0, const mfem::Vector & y0, mfem::Vector & xf, mfem::Vector & yf)
{
   mfem::BlockVector  Xk(block_offsets_xsy);  Xk = 0.0;
   mfem::BlockVector  Xtrial(block_offsets_xsy); Xtrial = 0.0;
   mfem::BlockVector dXNk(block_offsets_xsy); dXNk = 0.0;
   mfem::BlockVector dXtrk(block_offsets_xsy); dXtrk = 0.0;
   mfem::BlockVector GX(block_offsets_xsy); GX = 0.0;
   mfem::BlockVector GX0(block_offsets_xsy); GX0 = 0.0;
   mfem::BlockVector rk(block_offsets_xsy); rk = 0.0;
   mfem::BlockVector rktrial(block_offsets_xsy); rktrial = 0.0;
   mfem::BlockVector rklinear(block_offsets_xsy); rklinear = 0.0;
   mfem::BlockVector gk(block_offsets_xsy); gk = 0.0; // grad(mk) = grad_z(||rk + Jk * z||)|_{z=0} = 2 Jk^T rk
   
   Xk.GetBlock(0).Set(1.0, x0);
   Xk.GetBlock(1).Set(0.0, x0); // s0 = 0
   Xk.GetBlock(2).Set(1.0, y0);
   

   jOpt = 0;
   double theta = theta0;
   double delta = delta0;
   double rhok  = 0.0;
   bool inFilterRegion = false;
   bool inNeighborhood = false;
   const int max_tr_centering = 30;
   bool tr_centering;
   bool new_pt = true;

   int Geval_err;
   int reval_err;
   int Feval_err;
   int Qeval_err;
   int Eeval_err;
   
   mfem::Vector F0(dimx); F0 = 0.0;
   mfem::Vector Q0(dimy); Q0 = 0.0;
   problem->F(Xk.GetBlock(0), Xk.GetBlock(2), F0, Feval_err, new_pt);
   problem->Q(Xk.GetBlock(0), Xk.GetBlock(2), Q0, Qeval_err, new_pt);
   MFEM_VERIFY(Feval_err == 0 && Qeval_err == 0, "unsuccessful evaluation of F and/or Q at initial point of Homotopy solver");

   double cx_scale = mfem::GlobalLpNorm(2, F0.Norml2(), MPI_COMM_WORLD);
   double cy_scale = mfem::GlobalLpNorm(2, Q0.Norml2(), MPI_COMM_WORLD);
   cx_scale = std::max(1.0, sqrt(cx_scale));
   cy_scale = std::max(1.0, sqrt(cy_scale));
   cx_scale = std::min(cx_scale, max_cx_scale);
   cy_scale = std::min(cy_scale, max_cy_scale);
   cx = cx_scale;
   cy = cy_scale;
   
   double opt_err; // optimality error
   double betabar;
   while (jOpt < max_outer_iter)
   {
      new_pt = false;
      opt_err = E(Xk, Eeval_err, new_pt);
      if (save_iterates)
      {
         mfem::BlockVector xy(block_offsets_xy);
         xy.GetBlock(0).Set(1.0, Xk.GetBlock(0));
         xy.GetBlock(1).Set(1.0, Xk.GetBlock(2));
         mfem::Vector * iterate = new mfem::Vector(dimx + dimy);
         iterate->Set(1.0, xy);
         iterates.Append(iterate);
      }

      MFEM_VERIFY(Eeval_err == 0, "error in evaluation of optimality error E, should not occur\n");
      if (iAmRoot && print_level > 0)
      {
         *hout << "-----------------\n";
         *hout << "jOpt = " << jOpt << std::endl;
         *hout << "optimality error = " << opt_err << std::endl;
      }
      if (opt_err < tol)
      {
         if (iAmRoot && print_level > 0)
	 {
            *hout << "NMCP solver converged!\n";
	 }
	 converged = true;
	 break;
      }
      tr_centering = true;
      
      // not new X
      G(Xk, theta, GX, Geval_err, new_pt);
      JacG(Xk, theta, new_pt);
      ResidualFromG(GX, theta, rk); 
      
      if (iAmRoot && print_level > 0)
      {
	 *hout << "delta = " << delta << std::endl;
	 *hout << "||rk||_2 = " << rk.Norml2() << ", (theta = " << theta << ")\n";
      }
      NewtonSolve(*JGX, rk, dXNk); // Newton direction, associated to equation rk = 0
     
      JGX->MultTranspose(rk, gk); gk *= 2.0; // gradient of quadratic-model (\nabla_{dX}(||rk + Jk * dX||_2)^2)_{|dX=0}= 2 Jk^T rk
      DogLeg(*JGX, gk, kappa_delta * delta_MAX, dXNk, dXtrk);
      
      // compute trial point
      Xtrial.Set(1.0, Xk);
      Xtrial.Add(1.0, dXtrk);

      // new_x
      new_pt = true;
      Residual(Xtrial, theta, rktrial, reval_err, new_pt);
      mfem::Vector rktrial_comp_norm(3); rktrial_comp_norm = 0.0;
      for (int i = 0; i < 3; i++)
      {
         rktrial_comp_norm(i) = mfem::GlobalLpNorm(2, rktrial.GetBlock(i).Norml2(), MPI_COMM_WORLD);
      }
      inFilterRegion = FilterCheck(rktrial_comp_norm);
      inNeighborhood = NeighborhoodCheck(Xtrial, rktrial, theta, beta1, betabar);
      if (iAmRoot && print_level > 0)
      {
         if (inFilterRegion)
         {
            *hout << "cenGN -- trial point in filter region\n";
         }
         if (inNeighborhood)
         {
            *hout << "cenGN -- trial point in beta1 neighborhood\n";
         }
	 if (!inFilterRegion && inNeighborhood)
	 {
	    *hout << "cenGN -- skipping TR-centering\n";
	 }
      }
      
      if (!inFilterRegion && inNeighborhood && reval_err == 0)
      {
	 UpdateFilter(rktrial_comp_norm);
	 tr_centering = false;
	 Xk.Set(1.0, Xtrial);
      }

      for (int itr_centering = 0; itr_centering < max_tr_centering; itr_centering++)
      {
	 MFEM_VERIFY(delta > 1.e-30, "loss of accuracy in dog-leg (TR radius too small)");
         if (!tr_centering)
	 {
	    break;
	 }
	 DogLeg(*JGX, gk, delta, dXNk, dXtrk);
	 
	 
	 Xtrial.Set(1.0, Xk);
	 Xtrial.Add(1.0, dXtrk);
         // new_x
	 Residual(Xtrial, theta, rktrial, reval_err, new_pt);
	 if (reval_err > 0)
	 {
	    if (iAmRoot && print_level > 0)
	    {
	       *hout << "TRcen -- bad evaluation of residual (reducing trust-region radius)\n";
	    }   
	    delta *= 0.5;
	    continue;
	 }
         
	 // linearized residual, rk + Jk * dX 
	 JGX->Mult(dXtrk, rklinear);
         rklinear.Add(1.0, rk);
         /* evaluate the reduction in the objective
	  * (|| r(x) ||_2)^2
	  * and the reduction predicted from the
	  * linearized form
	  * (|| rk + Jk * dx||_2)^2 
	  */	 
	 double rk_sqrnorm       = mfem::InnerProduct(MPI_COMM_WORLD, rk, rk);
         double rktrial_sqrnorm  = mfem::InnerProduct(MPI_COMM_WORLD, rktrial, rktrial);
         double rklinear_sqrnorm = mfem::InnerProduct(MPI_COMM_WORLD, rklinear, rklinear);
         double pred_decrease   = rk_sqrnorm - rklinear_sqrnorm; // || rk ||_2^2 - || rk + Jk * dx||_2^2
	 double actual_decrease = rk_sqrnorm - rktrial_sqrnorm;  // || r(Xk) ||_2^2 - || r(Xk + dX) ||_2^2
	 rhok = actual_decrease / pred_decrease;
	 if (iAmRoot && print_level > 0)
	 {
	    *hout << "-*-*-*-*-*-*-*-*-*\n";
	    *hout << "TRcen -- delta = " << delta << std::endl;
	    *hout << "TRcen -- predicted decrease = " << pred_decrease << std::endl;
	    *hout << "TRcen -- actual decrease = " << actual_decrease << std::endl;
	 }
	 MFEM_VERIFY(pred_decrease > 0., "Loss of accuracy in dog-leg");

	 if (rhok < eta1)
	 {
	    for (int i = 0; i < 3; i++)
	    {
	       rktrial_comp_norm(i) = mfem::GlobalLpNorm(2, rktrial.GetBlock(i).Norml2(), MPI_COMM_WORLD);
	    }
            inFilterRegion = FilterCheck(rktrial_comp_norm);
	    inNeighborhood = NeighborhoodCheck(Xtrial, rktrial, theta, beta1, betabar);
            if (iAmRoot && print_level > 0)
            {
	       if (inFilterRegion)
	       {
	          *hout << "TRcen -- in filter region\n";
	       }
	       if (!inNeighborhood)
	       {
	          *hout << "TRcen -- not in beta1 neighborhood\n";
	       }
            }
	    if (!inFilterRegion && inNeighborhood && reval_err == 0)
	    {
	       UpdateFilter(rktrial_comp_norm);
	       delta *= 0.5;
	       Xk.Set(1.0, Xtrial);
	       if (iAmRoot && print_level > 0)
	       {
	          *hout << "TRcen -- accepted trial point, decreasing TR-radius\n";
	       }
	       break;
	    }
	    else
	    {
	       delta *= 0.5;
	       if (iAmRoot && print_level > 0)
	       {
	          *hout << "TRcen -- rejected trial point, decreasing TR-radius\n";
	       }
	       continue;
	    }
	 }
	 else if (rhok < eta2)
	 {
	    Xk.Set(1.0, Xtrial);
	    if (iAmRoot && print_level > 0)
	    {
	       *hout << "TRcen -- accepted trial point\n";
	    }	       
	    break;
	 }
	 else
	 {
	    if (iAmRoot && print_level > 0)
	    {
	       *hout << "TRcen -- accepted trial point, potentially increasing TR-radius\n";
	    }
	    delta = std::min(2.0 * delta, delta_MAX);
	    Xk.Set(1.0, Xtrial);
	    break;
	 }
      } 
      
      jOpt += 1;
      JacG(Xk, theta);

      Residual(Xk, theta, rk, reval_err);
      MFEM_VERIFY(reval_err == 0, "bad residual evaluation, this should have been caught\n");
      // Centrality management and targeting predictor step
      // zk \in N(theta, beta0)
      inNeighborhood = NeighborhoodCheck(Xk, rk, theta, beta0, betabar);
      if (inNeighborhood)
      {
         if (iAmRoot && print_level > 0)
	 {
	    *hout << "CenManagement -- reducing homotopy parameter\n";
	 }
	 double thetaplus = std::min(alg_nu * theta, pow(theta, alg_rho)); 
	 double t = 1.0;
	 double theta_t;
	 // compute predictor direction
         mfem::BlockVector rkp(block_offsets_xsy); rkp = 0.0;
	 mfem::BlockVector dXp(block_offsets_xsy); dXp = 0.0;
	 mfem::BlockVector Xtrialp(block_offsets_xsy); Xtrialp = 0.0;
	 
	 PredictorResidual(Xk, theta, thetaplus, rkp, reval_err);
	 MFEM_VERIFY(reval_err == 0, "bad residual evaluation, this should have been caught\n");
         NewtonSolve(*JGX, rkp, dXp);
         // line search
	 bool linesearch_converged = false;
	 int max_linesearch_steps = 60;
	 int i_linesearch = 0;
	 while ( !(linesearch_converged) && i_linesearch < max_linesearch_steps)
	 {
	    theta_t = (1.0 - t) * theta + t * thetaplus;
	    if (t < 1.e-8)
	    {
	       if (iAmRoot && print_level > 0)
	       {      
	          *hout << "CenManagement -- predictor step length too small\n";
	       }
	       theta = 0.9 * theta;
	       break;
	    }
	    Xtrialp.Set(1.0, Xk);
	    Xtrialp.Add(t, dXp);
            // new_x
	    Residual(Xtrialp, theta_t, rktrial, reval_err);
	    if (reval_err > 0)
	    {
	       if (iAmRoot && print_level > 0)
	       {
	           *hout << "CenManagement -- bad evaluation of residual\n";
	       }   
	       t = 0.995 * pow(t, 3.0);
	       i_linesearch += 1;
	       continue;
	    }
	    inNeighborhood = NeighborhoodCheck(Xtrialp, rktrial, theta_t, beta1, betabar);
	    if (inNeighborhood) 
	    {
	       // accept the trial point
	       linesearch_converged = true;
	       Xk.Set(1.0, Xtrialp);
	       theta = theta_t;
	       delta = delta_MAX;
	       ClearFilter();
	       if (iAmRoot && print_level > 0)
	       {
	          *hout << "CenManagement -- accepted linesearch trial point\n";
	          *hout << "CenManagement -- theta = " << theta << std::endl;
	       }
	    }
	    else
	    {
	       t = 0.995 * pow(t, 3.0);
	       if (iAmRoot && print_level > 0)
	       {
	          *hout << "CenManagement -- not in neighborhood\n";
	          *hout << "CenManagement -- reducing t\n";
	          *hout << "CenManagement -- t = " << t << std::endl;
	          *hout << "CenManagement -- thetaplus = " << thetaplus << std::endl;
	       }
	    }
	    i_linesearch += 1;
	 }
      }
      else
      {
         if (iAmRoot && print_level > 0)
	 {
	    *hout << "CenManagement -- skipping\n";
	    *hout << "CenManagement -- applying heuristics for quick termination resolution\n";
	 }
	 beta0 = fbeta * betabar;
	 beta1 = fbeta * beta0;
	 // compute grad(||r||^2) = ...
         JGX->MultTranspose(rk, gk); gk *= 2.0; // gradient of quadratic-model (\nabla_{dX}(||rk + Jk * dX||_2)^2)_{|dX=0}= 2 Jk^T rk
         double gk_norm = mfem::GlobalLpNorm(2, gk.Norml2(), MPI_COMM_WORLD);
	 if (gk_norm < theta * epsgrad)
	 {
            if (iAmRoot && print_level > 0)
            {
               if (earlyTermination)
	       {
	          *hout << "Exiting -- converged to a local stationary point of ||rk||_2^2\n";
	       }
               else
               {
	          *hout << "Warning -- apparent convergence to a local stationary point of ||rk||_2^2\n";
               }
            }
            if (earlyTermination)
	    {
	       break;
            }
	 }
	 else 
	 {	 
	    double Xk_norm = mfem::GlobalLpNorm(mfem::infinity(), Xk.Normlinf(), MPI_COMM_WORLD);
	    if (Xk_norm > deltabnd)
            {
	       if (iAmRoot && print_level > 0)
               {
                  if (earlyTermination)
	          {
	             *hout << "Exiting -- iterates are unbounded\n";
	          }
	          else
	          {
	             *hout << "Warning -- iterates appear to be unbounded\n";
	          }
               }
               if (earlyTermination)
	       {
	          break;
	       }
	    }
	    else
	    {
	       G(Xk, 0., GX0, Geval_err);
	       double GX0_norm = mfem::GlobalLpNorm(2, GX0.Norml2(), MPI_COMM_WORLD);
	       if (GX0_norm > feps * tol && theta < tol)
	       {
	          if (iAmRoot && print_level > 0)
                  {
                     if (earlyTermination)
		     {
		        *hout << "Exiting -- convergence to a non-interior point\n";
		     }
                     else
                     {
		        *hout << "Exiting -- convergence to a non-interior point\n";
                     }
          
                  }
                  if (earlyTermination)
		  {
		     break;
		  }
	       }
	    }
	 }

      }
   }
   xf.Set(1.0, Xk.GetBlock(0));
   yf.Set(1.0, Xk.GetBlock(2));
}



/* 
 *                     [ x + s - \sqrt( (x - s)^2 + 4(\theta * a)^q) ]
 * G(x, s, y; theta) = [ s - (F(x, y) + (\theta * \gamma_x)^p x)     ]
 *                     [ Q(x, y) + (\theta * \gamma_y)^p y           ] 
 */


void HomotopySolver::G(const mfem::BlockVector & X, const double theta, mfem::BlockVector & GX, int &Geval_err, bool new_pt)
{
   int Feval_err = 0;
   int Qeval_err = 0;
   mfem::Vector tempx(dimx); tempx = 0.0;
   // compute sqrt((x-s)^2 + 4(theta a)^q) term
   for (int i = 0; i < dimx; i++)
   {
      tempx(i) = std::sqrt(std::pow(X(i) - X(i+dimx), 2) + 4.0 * std::pow(theta * ax(i), q)); 
   }
   GX.GetBlock(0).Set( 1.0, X.GetBlock(0));
   GX.GetBlock(0).Add( 1.0, X.GetBlock(1));
   GX.GetBlock(0).Add(-1.0, tempx);

   tempx = 0.0;
   problem->F(X.GetBlock(0), X.GetBlock(2), tempx, Feval_err, new_pt);
   for (int i = 0; i < dimx; i++)
   {
      tempx(i) += std::pow(theta * gammax(i), p) * X(i);
   }
   GX.GetBlock(1).Set( 1.0, X.GetBlock(1));
   GX.GetBlock(1).Add(-1.0, tempx);
   
   mfem::Vector tempy(dimy); tempy = 0.0;
   problem->Q(X.GetBlock(0), X.GetBlock(2), tempy, Qeval_err, new_pt);
   for (int i = 0; i < dimy; i++)
   {
      tempy(i) += std::pow(theta * gammay(i), p) * X.GetBlock(2)(i);
   }
   GX.GetBlock(2).Set(1.0, tempy);
   Geval_err = std::max(Feval_err, Qeval_err);
}

double HomotopySolver::E(const mfem::BlockVector &X, int & Eeval_err, bool new_pt)
{
   mfem::Vector x(dimx); x = 0.0;
   mfem::Vector s(dimx); s = 0.0;
   x.Set(1.0, X.GetBlock(0));
   s.Set(1.0, X.GetBlock(1));
   mfem::Vector xsc(dimx); xsc = 0.0;
   mfem::Vector ssc(dimx); ssc = 0.0;
   for (int i = 0; i < dimx; i++)
   {
      xsc(i) = std::max(1.0, abs(x(i)) / f0);
      ssc(i) = std::max(1.0, abs(s(i)) / f0);
   }
   double xsc_1norm = mfem::GlobalLpNorm(1, xsc.Norml1(), MPI_COMM_WORLD);
   double ssc_1norm = mfem::GlobalLpNorm(1, ssc.Norml1(), MPI_COMM_WORLD);

   double xsc_infnorm = mfem::GlobalLpNorm(mfem::infinity(), xsc.Normlinf(), MPI_COMM_WORLD);
   double ssc_infnorm = mfem::GlobalLpNorm(mfem::infinity(), ssc.Normlinf(), MPI_COMM_WORLD);
   double Msc = std::max(xsc_infnorm, ssc_infnorm);
   double fz = 1.0;
   if( dimxglb > 0 ) {
     fz = std::max(xsc_1norm, ssc_1norm) / ( static_cast<double>(dimxglb) ); 
   }

   mfem::BlockVector r0(block_offsets_xsy); r0 = 0.0;
   Residual(X, 0.0, r0, Eeval_err, new_pt); // residual at \theta = 0

   mfem::Array<double> r0_infnorms(3);
   for (int i = 0; i < 3; i++)
   {
      r0_infnorms[i] = mfem::GlobalLpNorm(mfem::infinity(), r0.GetBlock(i).Normlinf(), MPI_COMM_WORLD);
   }
   mfem::Vector xs(dimx); xs = 0.0;
   xs.Set(1.0, x);
   xs *= s;
   double xs_infnorm = mfem::GlobalLpNorm(mfem::infinity(), xs.Normlinf(), MPI_COMM_WORLD);
   
   double Err = 0.;
   if (dimxglb > 0)
   {
      Err = std::max(std::min(r0_infnorms[0] / fz, xs_infnorm / Msc), std::max(r0_infnorms[1], r0_infnorms[2]) / fz);
   }
   else
   {
      Err = r0_infnorms[2];
   }
   return Err;
}



void HomotopySolver::Residual(const mfem::BlockVector & X, const double theta, mfem::BlockVector & r, int & reval_err, bool new_pt)
{
   G(X, theta, r, reval_err, new_pt);
   r.GetBlock(0).Add(-theta, bx);
   r.GetBlock(1).Add(-theta, cx);
   r.GetBlock(2).Add(-theta, cy);
}

void HomotopySolver::ResidualFromG(const mfem::BlockVector & GX, const double theta, mfem::BlockVector & r)
{
   r.Set(1.0, GX);
   r.GetBlock(0).Add(-theta, bx);
   r.GetBlock(1).Add(-theta, cx);
   r.GetBlock(2).Add(-theta, cy);
}


void HomotopySolver::PredictorResidual(const mfem::BlockVector & X, const double theta, const double thetaplus, mfem::BlockVector & r, int & reval_err, bool new_pt)
{
   G(X, theta, r, reval_err, new_pt);
   r.GetBlock(0).Add(-thetaplus, bx);
   r.GetBlock(1).Add(-thetaplus, cx);
   r.GetBlock(2).Add(-thetaplus, cy);

   mfem::Vector tempx(dimx); tempx = 0.0;
   for (int i = 0; i < dimx; i++)
   {
      tempx(i) = 2.0 * q * pow(theta * ax(i), q - 1.0);
   }
   mfem::Vector temps(dimx); temps = 0.0;
   for (int i = 0; i < dimx; i++)
   {
      temps(i) = p * pow(theta * gammax(i), p - 1) * X.GetBlock(0)(i);
   }
   mfem::Vector tempy(dimy); tempy = 0.0;
   for (int i = 0; i < dimy; i++)
   {
      tempy(i) = -p * pow(theta * gammay(i), p - 1) * X.GetBlock(2)(i);
   }
   r.GetBlock(0).Add(theta - thetaplus, tempx);
   r.GetBlock(1).Add(theta - thetaplus, temps);
   r.GetBlock(2).Add(theta - thetaplus, tempy);
}



/*                       [dG_(1,1)   dG_(1,2)   0       ]
 * \nabla G_t(x, s, y) = [dG_(2,1)   dG_(2,2)   dG_(2,3)]
 *                       [dG_(3,1)     0        dG_(3,3)]
 */
void HomotopySolver::JacG(const mfem::BlockVector &X, const double theta, bool new_pt)
{
   // I - diag( (x - s) / sqrt( (x - s)^2 + 4 * (\theta a)^q)
   mfem::Vector diagJacGxx(dimx); diagJacGxx = 0.0;
   for (int i = 0; i < dimx; i++)
   {
      diagJacGxx(i) = 1.0 - (X(i) - X(i+dimx)) / std::sqrt(
		     std::pow(X(i) - X(i+dimx), 2) + 4. * std::pow(theta * ax(i), q)); 
   }
   if (JGxx)
   {
      delete JGxx; JGxx = nullptr;
   }
   JGxx = GenerateHypreParMatrixFromDiagonal(dofOffsetsx, diagJacGxx);

   // I + diag( (x - s) / sqrt( (x - s)^2 + 4 (\theta a)^q)
   // = 2 I - [I - diag( (x - s) / sqrt( (x - s)^2 + 4 * (\theta a)^q)]
   mfem::Vector diagJacGxs(dimx);
   diagJacGxs = 2.0;
   diagJacGxs.Add(-1.0, diagJacGxx);
   if (JGxs)
   {
      delete JGxs; JGxs = nullptr;
   }
   JGxs = GenerateHypreParMatrixFromDiagonal(dofOffsetsx, diagJacGxs);
   
   // d / dx (G_t)_2 = -dF/dx - (t \gamma_x)^p
   if (JGsx)
   {
      delete JGsx; JGsx = nullptr;
   }
   dFdx = dynamic_cast<mfem::HypreParMatrix*>(problem->DxF(X.GetBlock(0), X.GetBlock(2)));
   mfem::Vector diagtgx(dimx); diagtgx = 0.0;
   for (int i = 0; i < dimx; i++)
   {
      diagtgx(i) = pow(theta * gammax(i), p);
   }
   mfem::HypreParMatrix * Dtgx = GenerateHypreParMatrixFromDiagonal(dofOffsetsx, diagtgx);
   JGsx = mfem::Add(-1.0, *dFdx, -1.0, *Dtgx);
   delete Dtgx;

   // d / ds (G_t)_2 = I
   mfem::Vector one(dimx); one = 1.0;
   if (JGss)
   {
      delete JGss;
   }
   JGss = GenerateHypreParMatrixFromDiagonal(dofOffsetsx, one);

   // d / dy (G_t)_2 = - dF / dy
   if (JGsy)
   {
      delete JGsy;
   }
   mfem::HypreParMatrix * temp_mat;
   temp_mat = dynamic_cast<mfem::HypreParMatrix *>(problem->DyF(X.GetBlock(0), X.GetBlock(2)));
   // make a copy that we manipulate
   JGsy = new mfem::HypreParMatrix(*temp_mat);
   one = -1.0;
   JGsy->ScaleRows(one);
   one = 1.0;

   // d / dx (G_t)_3 = dQ / dx
   if (JGyx)
   {
      delete JGyx;
   }
   temp_mat = dynamic_cast<mfem::HypreParMatrix *>(problem->DxQ(X.GetBlock(0), X.GetBlock(2)));
   JGyx = new mfem::HypreParMatrix(*temp_mat);


   // d / dy (G_t)_3 = dQ / dy + (t * \gamma_y)^p
   if (JGyy)
   {
      delete JGyy;   
   }
   dQdy = dynamic_cast<mfem::HypreParMatrix*>(problem->DyQ(X.GetBlock(0), X.GetBlock(2)));
   mfem::Vector diagtgy(dimy); diagtgy = 0.0;
   for (int i = 0; i < dimy; i++)
   {
      diagtgy(i) = std::pow(theta * gammay(i), p);
   }
   mfem::HypreParMatrix * Dtgy = GenerateHypreParMatrixFromDiagonal(dofOffsetsy, diagtgy);
   JGyy = mfem::Add(1.0, *dQdy, 1.0, *Dtgy);
   delete Dtgy; 

   if (JGX)
   {
      delete JGX;
   }
   JGX = new mfem::BlockOperator(block_offsets_xsy, block_offsets_xsy);
   JGX->SetBlock(0, 0, JGxx); JGX->SetBlock(0, 1, JGxs);
   JGX->SetBlock(1, 0, JGsx); JGX->SetBlock(1, 1, JGss); JGX->SetBlock(1, 2, JGsy);
   JGX->SetBlock(2, 0, JGyx);                            JGX->SetBlock(2, 2, JGyy);
}




// solve J dX_N = - rk
void HomotopySolver::NewtonSolve(mfem::BlockOperator & JkOp, const mfem::BlockVector & rk, mfem::BlockVector & dXN)
{
   if (linSolver)
   {
      linSolver->SetOperator(JkOp);
      linSolver->Mult(rk, dXN);
      dXN *= -1.0;
      mfem::IterativeSolver * itSolver = dynamic_cast<mfem::IterativeSolver *>(linSolver);
      if (!itSolver)
      {
         CondensedHomotopySolver * condensedhomotopysolver = dynamic_cast<CondensedHomotopySolver*>(linSolver);
         if (condensedhomotopysolver)
	 {
	    itSolver = dynamic_cast<mfem::IterativeSolver *>(condensedhomotopysolver->GetReducedSolver()); 
	 }
      }

      if (itSolver)
      {
	 if (itSolver->GetConverged())
	 {
	    krylov_its.Append(itSolver->GetNumIterations());
	 }
	 else
	 {
	    krylov_its.Append(-1);
	 }
      }
      else
      {
         krylov_its.Append(-1);
      }
   }
   else
   {
      int num_row_blocks = JkOp.NumRowBlocks();
      int num_col_blocks = JkOp.NumColBlocks();
      mfem::Array2D<const mfem::HypreParMatrix *> JkBlockMat(num_row_blocks, num_col_blocks);
      for(int i = 0; i < num_row_blocks; i++)
      {
         for(int j = 0; j < num_col_blocks; j++)
         {
            if(!JkOp.IsZeroBlock(i, j))
            {
               JkBlockMat(i, j) = dynamic_cast<mfem::HypreParMatrix *>(&(JkOp.GetBlock(i, j)));
	            MFEM_VERIFY(JkBlockMat(i, j), "dynamic cast failure");
            }
            else
            {
               JkBlockMat(i, j) = nullptr;
            }
         }
      }
      
      mfem::HypreParMatrix * Jk = mfem::HypreParMatrixFromBlocks(JkBlockMat);
      /* direct solve of the 3x3 IP-Newton linear system */
      DirectSolver directSolver(*Jk);
      directSolver.Mult(rk, dXN);
      dXN *= -1.0;
      delete Jk;
   }
}

void HomotopySolver::DogLeg(const mfem::BlockOperator & JkOp, const mfem::BlockVector & gk, const double delta, const mfem::BlockVector & dXN, mfem::BlockVector & dXtr)
{
   double dXN_norm = mfem::GlobalLpNorm(2, dXN.Norml2(), MPI_COMM_WORLD);
   if (dXN_norm <= delta)
   {
      dXtr.Set(1.0, dXN);
      if (iAmRoot && print_level > 0)
      {
         *hout << "dog-leg using Newton direction\n";
      }
   }
   else
   {
      mfem::BlockVector dXsd(block_offsets_xsy); dXsd = 0.0;
      mfem::BlockVector Jkgk(block_offsets_xsy); Jkgk = 0.0;
      JkOp.Mult(gk, Jkgk);
      double gk_norm = mfem::GlobalLpNorm(2, gk.Norml2(), MPI_COMM_WORLD);
      double Jkgk_norm = mfem::GlobalLpNorm(2, Jkgk.Norml2(), MPI_COMM_WORLD);
      dXsd.Set(-0.5 * std::pow(gk_norm, 2) / std::pow(Jkgk_norm, 2), gk);

      // || dXsd || = 0.5 * || gk ||^3 / || Jk gk||^2
      double dXsd_norm = 0.5 * pow(gk_norm, 3) / pow(Jkgk_norm, 2);
      if (dXsd_norm >= delta)
      {
         dXtr.Set(-delta / gk_norm, gk);
	 if (iAmRoot && print_level > 0)
	 {
	    *hout << "dog-leg using steepest descent direction\n";
	 }
      }
      else
      {
	 double t_star;
	 double a, b, c;
	 double dXsdTdXN = mfem::InnerProduct(MPI_COMM_WORLD, dXN, dXsd);
	 a = pow(dXN_norm, 2) - 2.0 * dXsdTdXN + pow(dXsd_norm, 2);
	 b = 2.0 * (dXsdTdXN - pow(dXsd_norm, 2));
	 c = pow(dXsd_norm, 2) - pow(delta, 2);
	 double discr = pow(b, 2) - 4.0 * a * c;
	 MFEM_VERIFY(discr >= 0. && a >= 0., "loss of accuracy: Gauss-Newton model not convex?!?");
	 if (b > 0.)
	 {
	    t_star = (-2. * c) / (b + sqrt(discr));
	 }
	 else
	 {
	    t_star = (-b + sqrt(discr)) / (2. * a);
	 }
         dXtr.Set(t_star, dXN);
	 dXtr.Add((1.0 - t_star), dXsd);
	 if (iAmRoot && print_level > 0)
	 {
	    *hout << "dog-leg using combination of Newton and SD directions\n";
	 }
      }
   }
}

bool HomotopySolver::FilterCheck(const mfem::Vector & r_comp_norm)
{
   // for each index j = 0, 1, 2 see if 
   // ||r_j(Xtrial)||_2 < alpha_j - gamma_f * || [alpha0; alpha1; alpha2;] ||_2
   // for all (alpha0, alpha1, alpha2) in the filter
   mfem::Array<bool> reductionJ;  reductionJ.SetSize(3);
   for (int j = 0; j < 3; j++)
   {
      reductionJ[j] = true;
   }

   for (int i = 0; i < filter.Size(); i++)
   {
      MFEM_VERIFY(filter[i]->Size() == 3, "each element of filter must be a 3-vector");
      for (int j = 0; j < 3; j++)
      {
         if (r_comp_norm(j) >= filter[i]->Elem(j) - gammaf * filter[i]->Norml2())
         {
            reductionJ[j] = false;
            continue;
         }
      }
   }
   
   // the region that the filter does not permit are those
   // points for which there does not exist and i such that
   // for each (alpha1, alpha2, alpha) in F that
   // reduction is achieved ||r_theta_i|| < alpha_i - gamma_f ||(alpha1, alpha2, alpha3)||
   // if reduction is achieved then we are not in the filtered region
   // and we say that the given point is not in F 
   bool inFilteredRegion = true;
   for (int j = 0; j < 3; j++)
   {
      if (reductionJ[j])
      {
         inFilteredRegion = false;
	 break;
      }
   }
   reductionJ.SetSize(0);
   return inFilteredRegion;
}

void HomotopySolver::UpdateFilter(const mfem::Vector & r_comp_norm)
{
   filter.Append(new mfem::Vector(3));
   for (int i = 0; i < 3; i++)
   {
      filter.Last()->Elem(i) = r_comp_norm(i);
   }
}

void HomotopySolver::ClearFilter()
{
   for (int i = 0; i < filter.Size(); i++)
   {
      delete filter[i];
   }
   filter.SetSize(0);
}


bool HomotopySolver::NeighborhoodCheck(const mfem::BlockVector & X, const mfem::BlockVector & r, const double theta, const double beta, double & betabar_)
{
   if (useNeighborhood1)
   {
      return NeighborhoodCheck_1(X, r, theta, beta, betabar_);
   }
   else
   {
      return NeighborhoodCheck_2(X, r, theta, beta, betabar_);
   }
}


bool HomotopySolver::NeighborhoodCheck_1(const mfem::BlockVector & /*X*/, const mfem::BlockVector & r, const double theta, const double beta, double & betabar_)
{
   double r_inf_norm = mfem::GlobalLpNorm(mfem::infinity(), r.Normlinf(), MPI_COMM_WORLD);
   bool inNeighborhood = (r_inf_norm <= beta * theta);
   betabar_ = r_inf_norm / theta;
   return inNeighborhood;
}


// see isInNeigh_2 method of numerial experiments branch of mHICOp
bool HomotopySolver::NeighborhoodCheck_2(const mfem::BlockVector & X, const mfem::BlockVector & r, const double theta, const double beta, double & betabar_)
{
   double x_inf_norm = mfem::GlobalLpNorm(mfem::infinity(), X.GetBlock(0).Normlinf(), MPI_COMM_WORLD);
   double s_inf_norm = mfem::GlobalLpNorm(mfem::infinity(), X.GetBlock(1).Normlinf(), MPI_COMM_WORLD);
   double xs_inf_norm = std::max(x_inf_norm, s_inf_norm);


   mfem::Vector r_inf_norms(3); r_inf_norms = 0.0;
   for (int i = 0; i < 3; i++)
   {
      r_inf_norms(i) = mfem::GlobalLpNorm(mfem::infinity(), r.GetBlock(i).Normlinf(), MPI_COMM_WORLD);
   }


   bool inNeighborhood = ((r_inf_norms(0) <= beta * theta * xs_inf_norm) && (std::max(r_inf_norms(1), r_inf_norms(2)) <= beta * theta));
   betabar_ = std::max(r_inf_norms(0) / (theta * xs_inf_norm), std::max(r_inf_norms(1), r_inf_norms(2)) / theta);
   return inNeighborhood;
}



HomotopySolver::~HomotopySolver()
{
   block_offsets_xsy.DeleteAll();
   block_offsets_xy.DeleteAll();
   for (int i = 0; i < iterates.Size(); i++)
   {
     delete iterates[i];
   }


   gammax.SetSize(0);
   gammay.SetSize(0);
   ax.SetSize(0);
   bx.SetSize(0);
   cx.SetSize(0);
   cy.SetSize(0);
   if (JGxx)
   {
      delete JGxx;
   }
   if (JGxs)
   {
      delete JGxs;
   }
   if (JGsx)
   {
      delete JGsx;
   }
   if (JGss)
   {
      delete JGss;
   }
   if (JGsy)
   {
      delete JGsy;
   }
   if (JGyx)
   {
      delete JGyx;
   }
   if (JGyy)
   {
      delete JGyy;
   }
   if (JGX)
   {
      delete JGX;
   }
   ClearFilter();
}
