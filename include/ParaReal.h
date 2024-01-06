#include "GradientFlow.h"
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>

///@note: split non-template object into header and source


//#include "gnuplot-iostream.h"

using namespace dealii;

template<int dim>
struct VectorTypes{
    using ArrayType = typename std::array<Vector<double>, dim>;
    using VectorArrayType = std::vector<ArrayType>;
    using TupleType = std::tuple<ArrayType, bool>;
    using FutureArrayType = Utilities::MPI::Future<ArrayType>;
    using FutureTupleType = Utilities::MPI::Future<TupleType>;

};


const unsigned int root = 0;

template<int dim>
class ParaRealBase
{
public:

  ParaRealBase();
  virtual void run() = 0;
  void set_num_inner_it(const unsigned int n_it);
  void set_final_time(const double T);

protected:

  std::unique_ptr<GradientFlowBase<dim>> create_GF(const GFStepType GFMethod);

  MPI_Comm                    mpi_communicator;

  const unsigned int          n_mpi_processes;
  const unsigned int          this_mpi_process;

  unsigned int                n_inner_iter;

  double                      global_T;

  bool                        converged;

};

template<int dim>
ParaRealBase<dim>::ParaRealBase()
:     mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , converged(false)
{}

template<int dim>
std::unique_ptr<GradientFlowBase<dim>> ParaRealBase<dim>::create_GF(const GFStepType GFMethod)
{
  switch(GFMethod)
  {
        case GFStepType::EULER:
            return std::make_unique<GradientFlowEuler<dim>>();

        case GFStepType::ADAM:
            return std::make_unique<GradientFlowAdam<dim>>();

        case GFStepType::RMSPROP:
            return std::make_unique<GradientFlowRMSProp<dim>>();

        default:
            std::cerr << "Algorithm not implemented.\n" << std::endl;
            return nullptr;
  }
}


template<int dim>
void ParaRealBase<dim>::set_num_inner_it(const unsigned int n_it)
{
    n_inner_iter = n_it;
    std::cout << " n_inner_iter: " << n_inner_iter << std::endl;
}

template<int dim>
void ParaRealBase<dim>::set_final_time(const double T)
{
    global_T = T;
}


template<int dim>
class ParaReal_Root: public ParaRealBase<dim>
{
public:

  using VT = VectorTypes<dim>;

  ParaReal_Root(const GFStepType OuterGFMethod, const GFStepType InnerGFMethod);
  void set_outer_step_size(const double outer_it=1.0);
  void set_inner_step_size(const double step_size=0.1);
  virtual void run() override;

private:

  bool check_convergence();

  double                                    delta_G;
  double                                    delta_F;

  std::unique_ptr<GradientFlowBase<dim>>    gf_F;
  std::unique_ptr<GradientFlowBase<dim>>    gf_G;

  unsigned int                              n_outer_iter=1;

  typename VT::VectorArrayType                       F_vectors; //to receive
  typename VT::VectorArrayType                       G_old_vectors;
  typename VT::VectorArrayType                       G_new_vectors;
  typename VT::VectorArrayType                       new_yu_vectors; //to send

  std::vector<bool>                         converged_vec;
  unsigned int                              converged_rank;

};

template<int dim>
ParaReal_Root<dim>::ParaReal_Root(const GFStepType OuterGFMethod, const GFStepType InnerGFMethod)
:     gf_F(this->create_GF(InnerGFMethod))
    , gf_G(this->create_GF(OuterGFMethod))
    , F_vectors(this->n_mpi_processes)
    , G_old_vectors(this->n_mpi_processes)
    , G_new_vectors(this->n_mpi_processes)
    , new_yu_vectors(this->n_mpi_processes)
    , converged_vec(this->n_mpi_processes, false)
{}

template<int dim>
void ParaReal_Root<dim>::set_outer_step_size(const double outer_it)
{
  assert(this->n_mpi_processes != 0);

  delta_G = outer_it;
  gf_G->set_step_size(delta_G);
  n_outer_iter = static_cast<unsigned int>(this->global_T/(this->n_mpi_processes*outer_it));
  std::cout << " n_outer_iter: " << n_outer_iter << std::endl;

  }

template<int dim>
void ParaReal_Root<dim>::set_inner_step_size(const double step_size)
{
    assert(this->n_mpi_processes != 0);

    delta_F = step_size;
    gf_F->set_step_size(delta_F);
    this->n_inner_iter = static_cast<unsigned int>(this->global_T/(this->n_mpi_processes*step_size));
    std::cout << " n_inner_iter: " << this->n_inner_iter << std::endl;
}

template<int dim>
bool ParaReal_Root<dim>::check_convergence()
{
    for (unsigned int i=0; i<this->n_mpi_processes; i++)
      if (converged_vec[i] == true)
      {
          converged_rank = i;
          return true;
      }
    return false;
}

template<int dim>
void ParaReal_Root<dim>::run()
{
  std::cout << "N outer iter: " << n_outer_iter << std::endl;
  std::cout << "N inner iter: " << this->n_inner_iter << std::endl;
  std::cout << "Initialization" << std::endl;
  // Initialization
  for (unsigned int i=0; i<this->n_mpi_processes; i++)
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
  while(!this->converged)
  {
      std::cout << "Iteration n: " << it << std::endl;
      // Send Results to all ranks
      for (unsigned int rank=1; rank<this->n_mpi_processes; rank++)
      {
        Utilities::MPI::isend(new_yu_vectors[rank-1], this->mpi_communicator, rank, 0);
      }

      // Root computes its own F operator in the first interval [t0, t1]: i.e. computes F at t1
      gf_F->run(this->n_inner_iter);
      F_vectors[0][0] = gf_F->get_y_vec();
      F_vectors[0][1] = gf_F->get_u_vec();
      converged_vec[0] = gf_F->converged();

      // Update the value of y and u obtained from root (coarse operator cancels out)
      new_yu_vectors[0][0] = F_vectors[0][0];
      new_yu_vectors[0][1] = F_vectors[0][1];

      // Receive F operator and local convergence results from all ranks
      for (unsigned int rank=1; rank<this->n_mpi_processes; rank++)
      {
        typename VT::FutureTupleType future = Utilities::MPI::irecv<typename VT::TupleType>(this->mpi_communicator, rank, 0);
        typename VT::TupleType fut_tuple = future.get();
        F_vectors[rank] = std::get<0>(fut_tuple);
        converged_vec[rank] = std::get<1>(fut_tuple);
      }

      // Compute convergence and send result to other ranks
      this->converged = check_convergence();
      for (unsigned int rank=1; rank<this->n_mpi_processes; rank++)
      {
        Utilities::MPI::isend(this->converged, this->mpi_communicator, rank, 1);
      }

      if(!this->converged)
      {
        // Compute G_k+1
        G_old_vectors = G_new_vectors;
        for (unsigned int rank=1; rank<this->n_mpi_processes; rank++)
        {
          // Update G
          gf_G->set_initial_vectors(new_yu_vectors[rank-1][0], new_yu_vectors[rank-1][1]);
          gf_G->run(n_outer_iter);
          G_new_vectors[rank][0] = gf_G->get_y_vec();
          G_new_vectors[rank][1] = gf_G->get_u_vec();
          //gf_G->output_iteration_results();

          // Update values of y and u
          Vector<double> y_temp(G_new_vectors[rank][0]);

          y_temp -= G_old_vectors[rank][0];
          y_temp += F_vectors[rank][0];

          Vector<double> u_temp(G_new_vectors[rank][1]);
          u_temp -= G_old_vectors[rank][1];
          u_temp += F_vectors[rank][1];

          new_yu_vectors[rank][0] = y_temp;
          new_yu_vectors[rank][1] = u_temp;
        }
        /*
        // Compute new y and u (used as initial conditions of inner processes)
        for (unsigned int rank=0; rank<this->n_mpi_processes; rank++)
        {
          Vector<double> y_temp(G_new_vectors[rank][0]);

          y_temp -= G_old_vectors[rank][0];
          y_temp += F_vectors[rank][0];

          Vector<double> u_temp(G_new_vectors[rank][1]);
          u_temp -= G_old_vectors[rank][1];
          u_temp += F_vectors[rank][1];

          new_yu_vectors[rank][0] = y_temp; //std::make_tuple(y_temp, u_temp);
          new_yu_vectors[rank][1] = u_temp;
        }*/

        // Output results of this iteration process
        for (unsigned int rank=0; rank<this->n_mpi_processes; rank++)
        {
          gf_G->set_initial_vectors(new_yu_vectors[rank][0], new_yu_vectors[rank][1]);
          gf_G->output_iteration_results();
        }
        gf_G->output_results_vectors();
      }
      else
      {
        // Output last iteration results
        for (unsigned int rank=0; rank<this->n_mpi_processes; rank++)
        {
          gf_G->set_initial_vectors(F_vectors[rank][0], F_vectors[rank][1]);
          gf_G->output_iteration_results();
        }
        gf_G->output_results_vectors();

      }

      it++;
  }

  // Output vectors
  std::cout << "Final results " << std::endl;
  std::cout << "Result obtained in: " << (n_outer_iter*this->n_mpi_processes + this->n_inner_iter)*it << " iterations" << std::endl;
  gf_G->set_initial_vectors(F_vectors[converged_rank][0], F_vectors[converged_rank][1]);
  gf_G->output_iteration_results();
  gf_G->output_results_vectors();
}

template <int dim>
class ParaReal_Rank_n: public ParaRealBase<dim>
{
public:

  using VT = VectorTypes<dim>;

  ParaReal_Rank_n(const GFStepType InnerGFMethod);
  void set_inner_step_size(const double step_size=0.1);
  virtual void run() override;

private:

  double                                  delta_F;

  std::unique_ptr<GradientFlowBase<dim>>  gf_F;

  typename VT::ArrayType                           initial_time_vectors; //to receive
  typename VT::ArrayType                           final_time_vectors; //to send

  bool                                    local_conv;
};

template<int dim>
ParaReal_Rank_n<dim>::ParaReal_Rank_n(const GFStepType InnerGFMethod)
:     gf_F(this->create_GF(InnerGFMethod))
,     local_conv(false)
{}

template<int dim>
void ParaReal_Rank_n<dim>::set_inner_step_size(const double step_size)
{
      assert(this->n_mpi_processes != 0);

      delta_F = step_size;
      gf_F->set_step_size(delta_F);
      this->n_inner_iter = static_cast<unsigned int>(this->global_T/(this->n_mpi_processes*step_size));
      std::cout << " n_inner_iter: " << this->n_inner_iter << std::endl;
}

template<int dim>
void ParaReal_Rank_n<dim>::run()
{

  //for (unsigned int it=0; it < n_pr_iter; it++)
  while(!this->converged)
  {
      // Receive initial conditions from root
      typename VT::FutureArrayType future = Utilities::MPI::irecv<typename VT::ArrayType>(this->mpi_communicator, root, 0);
      initial_time_vectors = future.get();

      // Set received vectors as initial conditions of inner gf
      gf_F->set_initial_vectors(initial_time_vectors[0], initial_time_vectors[1]);

      gf_F->run(this->n_inner_iter);
      local_conv = gf_F->converged();
      final_time_vectors[0] = gf_F->get_y_vec();
      final_time_vectors[1] = gf_F->get_u_vec();

      typename VT::TupleType tuple_to_send = std::make_tuple(final_time_vectors, local_conv);

      // Send tuple with the final vectors to the root and local convergence results
      Utilities::MPI::isend(tuple_to_send, this->mpi_communicator, root, 0);

      // Receive convergence result from root
      Utilities::MPI::Future<bool> future1 = Utilities::MPI::irecv<bool>(this->mpi_communicator, root, 1);
      this->converged = future1.get();
  }

}
