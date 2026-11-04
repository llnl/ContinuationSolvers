#include "spline_contact_design_problem.hpp"

#include <cmath>
#include <memory>
#include <utility>

SplineContactDesignProblem::SplineContactDesignProblem(
   ParamOptProblem *contact_problem,
   const mfem::Vector &theta_reference,
   const mfem::HypreParMatrix &design_stiffness,
   mfem::real_t displacement_weight,
   mfem::real_t design_weight,
   mfem::real_t design_smoothing_weight,
   mfem::real_t max_pressure_weight,
   mfem::real_t max_pressure_smoothing,
   mfem::real_t pressure_uniformity_weight,
   bool use_slacks)
   : MPECProblem(contact_problem, use_slacks),
     theta_reference_(theta_reference),
     design_stiffness_(design_stiffness),
     displacement_weight_(displacement_weight),
     design_weight_(design_weight),
     design_smoothing_weight_(design_smoothing_weight),
     max_pressure_weight_(max_pressure_weight),
     max_pressure_smoothing_(max_pressure_smoothing),
     pressure_uniformity_weight_(pressure_uniformity_weight),
     contact_problem_(contact_problem)
{
   MFEM_VERIFY(contact_problem != nullptr,
               "A parameterized contact problem is required.");
   MFEM_VERIFY(theta_reference_.Size() == contact_problem->GetDimTheta(),
               "The reference design has an incompatible local size.");
   MFEM_VERIFY(design_stiffness_.Height() == contact_problem->GetDimTheta() &&
                  design_stiffness_.Width() == contact_problem->GetDimTheta() &&
                  design_stiffness_.GetGlobalNumRows() ==
                     contact_problem->GetDimThetaGlb() &&
                  design_stiffness_.GetGlobalNumCols() ==
                     contact_problem->GetDimThetaGlb(),
               "The design stiffness matrix has incompatible dimensions.");
   MFEM_VERIFY(std::isfinite(displacement_weight_) &&
                  displacement_weight_ >= 0.0 &&
                  std::isfinite(design_weight_) && design_weight_ > 0.0 &&
                  std::isfinite(design_smoothing_weight_) &&
                  design_smoothing_weight_ >= 0.0,
               "Objective weights must be finite; the displacement weight "
               "and design smoothing weight must be nonnegative, and the "
               "design weight must be positive.");

   mfem::Vector displacement_diagonal(contact_problem->GetDimU());
   displacement_diagonal = displacement_weight_;
   HuuE.reset(GenerateHypreParMatrixFromDiagonal(
      contact_problem->GetDofOffsetsU(), displacement_diagonal));

   mfem::Vector design_diagonal(contact_problem->GetDimTheta());
   design_diagonal = design_weight_;
   std::unique_ptr<mfem::HypreParMatrix> design_identity(
      GenerateHypreParMatrixFromDiagonal(
         contact_problem->GetDofOffsetsTheta(), design_diagonal));
   if (design_smoothing_weight_ == 0.0)
   {
      HththE = std::move(design_identity);
   }
   else
   {
      HththE.reset(mfem::Add(1.0, *design_identity,
                             design_smoothing_weight_,
                             design_stiffness_));
   }
   design_change_.SetSize(contact_problem->GetDimTheta());
   design_change_ = 0.0;
}

double SplineContactDesignProblem::E(
   const mfem::Vector &u, const mfem::Vector & p,
   const mfem::Vector &theta, int &eval_err)
{
   MFEM_VERIFY(u.Size() == paramopt->GetDimU() &&
                  theta.Size() == paramopt->GetDimTheta(),
               "An upper-level objective argument has an incompatible "
               "local size.");
   mfem::Vector design_change(theta);
   design_change_.Set(1.0, theta);
   design_change_.Add(-1.0, theta_reference_);
   const MPI_Comm comm = design_stiffness_.GetComm();
   mfem::real_t smoothing_energy = 0.0;
   if (design_smoothing_weight_ > 0.0)
   {
      mfem::Vector stiffness_theta(theta.Size());
      design_stiffness_.Mult(theta, stiffness_theta);
      smoothing_energy = 0.5 * design_smoothing_weight_ *
         mfem::InnerProduct(comm, theta, stiffness_theta);
   }

   // smooth max pressure
   // ks(p) = max_pressure_weight_ * log{ \sum_i exp(max_pressure_smoothing_ * p_i) }
   // for numerical purposes factor exp(max_pressure_smoothing_ * max_pressre) from sum
   mfem::real_t smooth_weighted_max_pressure = 0.0;
   if (max_pressure_weight_ > 0.0)
   {
      mfem::real_t local_max_pressure = p.Max();
      mfem::real_t max_pressure = 0.0;
      MPI_Allreduce(&local_max_pressure, &max_pressure, 1, mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MAX, comm);
      mfem::real_t local_exp_ks = 0.0;
      mfem::real_t exp_ks = 0.0;
      for (int i = 0; i < p.Size(); i++)
      {
         local_exp_ks += std::exp(max_pressure_smoothing_ * (p(i) - max_pressure));         
      }   
      MPI_Allreduce(&local_exp_ks, &exp_ks, 1, mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_SUM, comm);
      smooth_weighted_max_pressure = max_pressure + std::log(exp_ks) / max_pressure_smoothing_;
      smooth_weighted_max_pressure *= max_pressure_weight_; 
   }
   mfem::real_t pressure_variance = 0.0;
   if (pressure_uniformity_weight_ > 0.0)
   {
      mfem::real_t local_pressure_sum = p.Sum();
      mfem::real_t pressure_sum = 0.0;
      MPI_Allreduce(&local_pressure_sum, &pressure_sum, 1, mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_SUM, comm);
      mfem::real_t pressure_mean = pressure_sum / ((mfem::real_t) contact_problem_->GetDimMGlb());
      mfem::real_t local_pressure_variance = 0.0;
      for (int i = 0; i < p.Size(); i++)
      {
         local_pressure_variance += std::pow(p(i) - pressure_mean, 2);
      }
      local_pressure_variance *= 1. / (((mfem::real_t) contact_problem_->GetDimMGlb()) - 1.);
      MPI_Allreduce(&local_pressure_variance, &pressure_variance, 1, mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_SUM, comm);
   }
   mfem::real_t weighted_pressure_variance = pressure_uniformity_weight_ * pressure_variance; 


   eval_err = 0; // no reason to error out
   return 0.5 * displacement_weight_ *
             mfem::InnerProduct(comm, u, u) +
          0.5 * design_weight_ *
             mfem::InnerProduct(comm, design_change_,
                                design_change_) +
          smoothing_energy + smooth_weighted_max_pressure + 
          weighted_pressure_variance;
}

void SplineContactDesignProblem::DuE(
   const mfem::Vector &u, const mfem::Vector & /*p*/,
   const mfem::Vector & /*theta*/, mfem::Vector &gradient)
{
   gradient.SetSize(u.Size());
   gradient = u;
   gradient *= displacement_weight_;
}

void SplineContactDesignProblem::DpE(
   const mfem::Vector & /*u*/, const mfem::Vector & p,
   const mfem::Vector &/*theta*/, mfem::Vector &gradient)
{
   gradient.SetSize(p.Size());
   gradient = 0.0;
   mfem::real_t smooth_weighted_max_pressure = 0.0;
   const MPI_Comm comm = design_stiffness_.GetComm();
   if (max_pressure_weight_ > 0.0)
   {
      mfem::real_t local_max_pressure = p.Max();
      mfem::real_t max_pressure = 0.0;
      MPI_Allreduce(&local_max_pressure, &max_pressure, 1, mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MAX, comm);
      mfem::real_t local_exp_ks = 0.0;
      mfem::real_t exp_ks = 0.0;
      for (int i = 0; i < p.Size(); i++)
      {
         local_exp_ks += std::exp(max_pressure_smoothing_ * (p(i) - max_pressure));         
      }   
      MPI_Allreduce(&local_exp_ks, &exp_ks, 1, mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_SUM, comm);
      for (int i = 0; i < p.Size(); i++)
      {
         gradient(i) = std::exp(max_pressure_smoothing_ * (p(i) - max_pressure));
      }
      gradient *= (max_pressure_weight_ / exp_ks);
   }
   if (pressure_uniformity_weight_ > 0.0)
   {
      mfem::real_t local_pressure_sum = p.Sum();
      mfem::real_t pressure_sum = 0.0;
      MPI_Allreduce(&local_pressure_sum, &pressure_sum, 1, mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_SUM, comm);
      mfem::real_t pressure_mean = pressure_sum / ((mfem::real_t) contact_problem_->GetDimMGlb());
      mfem::Vector temp(p.Size()); temp = 0.0;
      for (int i = 0; i < p.Size(); i++)
      {
         temp(i) = 2. / (((mfem::real_t) contact_problem_->GetDimMGlb()) - 1.) * (p(i) - pressure_mean);
      }
      gradient.Add(pressure_uniformity_weight_, temp);
   }



}

void SplineContactDesignProblem::DthE(
   const mfem::Vector & /*u*/, const mfem::Vector & /*p*/,
   const mfem::Vector &theta, mfem::Vector &gradient)
{
   gradient.SetSize(theta.Size());
   gradient = theta;
   gradient -= theta_reference_;
   gradient *= design_weight_;
   if (design_smoothing_weight_ > 0.0)
   {
      mfem::Vector stiffness_theta(theta.Size());
      design_stiffness_.Mult(theta, stiffness_theta);
      gradient.Add(design_smoothing_weight_, stiffness_theta);
   }
}

mfem::Operator *SplineContactDesignProblem::DuuE(
   const mfem::Vector & /*u*/, const mfem::Vector & /*p*/,
   const mfem::Vector & /*theta*/)
{
   return HuuE.get();
}

// diagonal approximation
mfem::Operator *SplineContactDesignProblem::DppE(
   const mfem::Vector & /*u*/, const mfem::Vector &p,
   const mfem::Vector & /*theta*/)
{
   mfem::Vector diagonal(p.Size());
   diagonal = 0.0;
   const MPI_Comm comm = design_stiffness_.GetComm();
   if (max_pressure_weight_ > 0.0)
   {
      mfem::real_t local_max_pressure = p.Max();
      mfem::real_t max_pressure = 0.0;
      MPI_Allreduce(&local_max_pressure, &max_pressure, 1, mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MAX, comm);
      mfem::real_t local_exp_ks = 0.0;
      mfem::real_t exp_ks = 0.0;
      for (int i = 0; i < p.Size(); i++)
      {
         local_exp_ks += std::exp(max_pressure_smoothing_ * (p(i) - max_pressure));         
      }   
      MPI_Allreduce(&local_exp_ks, &exp_ks, 1, mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_SUM, comm);
      for (int i = 0; i < p.Size(); i++)
      {
         diagonal(i) = std::exp(max_pressure_smoothing_ * (p(i) - max_pressure));
      }
      diagonal *= (max_pressure_weight_ * max_pressure_smoothing_ / exp_ks);
   }
   if (pressure_uniformity_weight_ > 0.0)
   {
      diagonal += 2. * pressure_uniformity_weight_ / (((mfem::real_t) contact_problem_->GetDimMGlb()));
   }


   HppE.reset(GenerateHypreParMatrixFromDiagonal(
      contact_problem_->GetDofOffsetsM(), diagonal));
   return HppE.get();
}


mfem::Operator *SplineContactDesignProblem::DththE(
   const mfem::Vector & /*u*/, const mfem::Vector & /*p*/,
   const mfem::Vector & /*theta*/)
{
   return HththE.get();
}
