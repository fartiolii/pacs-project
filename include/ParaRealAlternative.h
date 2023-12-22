#include "GradientFlow.h"
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>

//#include "gnuplot-iostream.h"

using namespace dealii;

struct VectorTypes{
    using ArrayType = std::array<Vector<double>, 2>;
    using VectorArrayType = std::vector<ArrayType>;
    using TupleType = std::tuple<ArrayType, bool>;
    using FutureArrayType = Utilities::MPI::Future<ArrayType>;
    using FutureTupleType = Utilities::MPI::Future<TupleType>;

};


const unsigned int root = 0;

class ParaRealBase
{
public:

  ParaRealBase();
  virtual void run() = 0;
  void set_num_inner_it(const unsigned int n_it);

protected:

  std::unique_ptr<GradientFlowBase> create_GF(const GFStepType GFMethod);

  MPI_Comm                    mpi_communicator;

  const unsigned int          n_mpi_processes;
  const unsigned int          this_mpi_process;

  unsigned int                n_inner_iter;

  bool                        converged;

};

ParaRealBase::ParaRealBase()
:     mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , converged(false)
{}

std::unique_ptr<GradientFlowBase> ParaRealBase::create_GF(const GFStepType GFMethod)
{
  switch(GFMethod)
  {
        case GFStepType::EULER:
            return std::make_unique<GradientFlowEuler>();

        case GFStepType::ADAM:
            return std::make_unique<GradientFlowAdam>();

        default:
            std::cerr << "Algorithm not implemented.\n";
            return nullptr;
  }
}



void ParaRealBase::set_num_inner_it(const unsigned int n_it)
{
    n_inner_iter = n_it;
    std::cout << " n_inner_iter: " << n_inner_iter << std::endl;
}



class ParaReal_Root: public ParaRealBase
{
public:

  using VT = VectorTypes;

  ParaReal_Root(const GFStepType OuterGFMethod, const GFStepType InnerGFMethod);
  void set_outer_step_size(const double outer_it=1.0);
  void set_inner_step_size(const double step_size=0.1);
  virtual void run() override;

private:

  bool check_convergence();

  double                            delta_G;
  double                            delta_F;

  std::unique_ptr<GradientFlowBase> gf_F;
  std::unique_ptr<GradientFlowBase> gf_G;

  unsigned int                      n_outer_iter=1;

  VT::VectorArrayType               F_vectors; //to receive
  VT::VectorArrayType               G_old_vectors;
  VT::VectorArrayType               G_new_vectors;
  VT::VectorArrayType               new_yu_vectors; //to send

  std::vector<bool>                 converged_vec;
  unsigned int                      converged_rank;

};

ParaReal_Root::ParaReal_Root(const GFStepType OuterGFMethod, const GFStepType InnerGFMethod)
:     gf_F(create_GF(InnerGFMethod))
    , gf_G(create_GF(OuterGFMethod))
    , F_vectors(n_mpi_processes)
    , G_old_vectors(n_mpi_processes)
    , G_new_vectors(n_mpi_processes)
    , new_yu_vectors(n_mpi_processes)
    , converged_vec(n_mpi_processes, false)
{}



void ParaReal_Root::set_outer_step_size(const double outer_it)
{
  assert(n_mpi_processes != 0);

  delta_G = outer_it;
  gf_G->set_step_size(delta_G);

  // New method: we just make one iteration of the coarse operator
  //n_outer_iter = static_cast<unsigned int>(global_T/(n_mpi_processes*outer_it));
  n_outer_iter = 1;
  std::cout << " n_outer_iter: " << n_outer_iter << std::endl;

}

void ParaReal_Root::set_inner_step_size(const double step_size)
{
    assert(n_mpi_processes != 0);

    delta_F = step_size;
    gf_F->set_step_size(delta_F);
}

bool ParaReal_Root::check_convergence()
{
    for (unsigned int i=0; i<n_mpi_processes; i++)
      if (converged_vec[i] == true)
      {
          converged_rank = i;
          return true;
      }
    return false;
}

void ParaReal_Root::run()
{
  std::cout << "Initialization" << std::endl;
  // Initialization
  for (unsigned int i=0; i<n_mpi_processes; i++)
  {
    gf_G->run(n_outer_iter);

    Vector<double> y = gf_G->get_y_vec();
    Vector<double> u = gf_G->get_u_vec();

    G_old_vectors[i][0] = y;
    G_old_vectors[i][1] = u;
    new_yu_vectors[i][0] = y;
    new_yu_vectors[i][1] = u;

    gf_G->output_iteration_results();
  }
  G_new_vectors = G_old_vectors;
  gf_G->output_results_vectors();

  unsigned int it=0;
  while(!converged)
  {
      std::cout << "Iteration n: " << it << std::endl;
      // Send Results to all ranks
      for (unsigned int rank=1; rank<n_mpi_processes; rank++)
      {
        Utilities::MPI::isend(new_yu_vectors[rank-1], mpi_communicator, rank, 0);
      }

      // Root computes its own F operator in the first interval [t0, t1]: i.e. computes F at t1
      gf_F->run(n_inner_iter);
      F_vectors[0][0] = gf_F->get_y_vec();
      F_vectors[0][1] = gf_F->get_u_vec();
      converged_vec[0] = gf_F->converged();

      // Receive F operator and local convergence results from all ranks
      for (unsigned int rank=1; rank<n_mpi_processes; rank++)
      {
        VT::FutureTupleType future = Utilities::MPI::irecv<VT::TupleType>(mpi_communicator, rank, 0);
        VT::TupleType fut_tuple = future.get();
        F_vectors[rank] = std::get<0>(fut_tuple);
        converged_vec[rank] = std::get<1>(fut_tuple);
      }

      // Compute G_k+1
      G_old_vectors = G_new_vectors;
      for (unsigned int rank=1; rank<n_mpi_processes; rank++)
      {
        //gf_G->set_initial_vectors(std::get<0>(F_vectors[rank-1]), std::get<1>(F_vectors[rank-1]));
        gf_G->set_initial_vectors(F_vectors[rank-1][0], F_vectors[rank-1][1]);
        gf_G->run(n_outer_iter);
        G_new_vectors[rank][0] = gf_G->get_y_vec();
        G_new_vectors[rank][1] = gf_G->get_u_vec();
        //gf_G->output_iteration_results();
      }

      // Compute new y and u (used as initial conditions of inner processes)
      for (unsigned int rank=0; rank<n_mpi_processes; rank++)
      {
        Vector<double> y_temp(G_new_vectors[rank][0]);

        y_temp -= G_old_vectors[rank][0];
        y_temp += F_vectors[rank][0];

        Vector<double> u_temp(G_new_vectors[rank][1]);
        u_temp -= G_old_vectors[rank][1];
        u_temp += F_vectors[rank][1];

        new_yu_vectors[rank][0] = y_temp; //std::make_tuple(y_temp, u_temp);
        new_yu_vectors[rank][1] = u_temp;
      }

      // Output results of this iteration process
      for (unsigned int rank=0; rank<n_mpi_processes; rank++)
      {
        gf_G->set_initial_vectors(new_yu_vectors[rank][0], new_yu_vectors[rank][1]);
        gf_G->output_iteration_results();
      }
      gf_G->output_results_vectors();

      // Compute convergence and send result to other ranks
      converged = check_convergence();
      for (unsigned int rank=1; rank<n_mpi_processes; rank++)
      {
        Utilities::MPI::isend(converged, mpi_communicator, rank, 1);
      }

      it++;
  }

  // Output vectors
  std::cout << "Final results " << std::endl;
  gf_G->set_initial_vectors(new_yu_vectors[converged_rank][0], new_yu_vectors[converged_rank][1]);
  gf_G->output_iteration_results();
  gf_G->output_results_vectors();
}

class ParaReal_Rank_n: public ParaRealBase
{
public:

  using VT = VectorTypes;

  ParaReal_Rank_n(const GFStepType InnerGFMethod);
  void set_inner_step_size(const double step_size=0.1);
  virtual void run() override;

private:

  double                      delta_F;

  std::unique_ptr<GradientFlowBase> gf_F;

  VT::ArrayType               initial_time_vectors; //to receive
  VT::ArrayType               final_time_vectors; //to send

  bool                        local_conv;
};

ParaReal_Rank_n::ParaReal_Rank_n(const GFStepType InnerGFMethod)
:     gf_F(create_GF(InnerGFMethod))
,     local_conv(false)
{}

void ParaReal_Rank_n::set_inner_step_size(const double step_size)
{
      delta_F = step_size;
      gf_F->set_step_size(delta_F);
}

void ParaReal_Rank_n::run()
{

  //for (unsigned int it=0; it < n_pr_iter; it++)
  while(!converged)
  {
      // Receive initial conditions from root
      VT::FutureArrayType future = Utilities::MPI::irecv<VT::ArrayType>(mpi_communicator, root, 0);
      initial_time_vectors = future.get();

      // Set received vectors as initial conditions of inner gf
      gf_F->set_initial_vectors(initial_time_vectors[0], initial_time_vectors[1]);

      gf_F->run(n_inner_iter);
      local_conv = gf_F->converged();
      final_time_vectors[0] = gf_F->get_y_vec();
      final_time_vectors[1] = gf_F->get_u_vec();

      VT::TupleType tuple_to_send = std::make_tuple(final_time_vectors, local_conv);

      // Send tuple with the final vectors to the root and local convergence results
      Utilities::MPI::isend(tuple_to_send, mpi_communicator, root, 0);

      // Receive convergence result from root
      Utilities::MPI::Future<bool> future1 = Utilities::MPI::irecv<bool>(mpi_communicator, root, 1);
      converged = future1.get();
  }

}
