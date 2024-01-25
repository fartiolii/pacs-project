#include "NumericalAlgorithmBase.h"
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>

using namespace dealii;

struct VectorTypes{
    using ArrayType = std::array<Vector<double>, 2>;
    using VectorArrayType = std::vector<ArrayType>;
    using TupleType = std::tuple<ArrayType, std::tuple<bool, unsigned int>>;
    using FutureArrayType = Utilities::MPI::Future<ArrayType>;
    using FutureTupleType = Utilities::MPI::Future<TupleType>;
    using ConvergenceTuple = std::vector<std::tuple<bool, unsigned int>>;
};


const unsigned int root = 0;


template<int dim>
class ParallelRootBase: public NumericalAlgorithmBase<dim>
{
public:

  using VT = VectorTypes;

  ParallelRootBase(const std::string& linear_system_filename);
  ParallelRootBase(const double gamma_val, const double nu_val, const unsigned int N_grid);
  void run() override;
  

protected:

  bool check_convergence();
  bool check_pr_interval_convergence();

  MPI_Comm                    		mpi_communicator;
  const unsigned int          	        n_mpi_processes;
  const unsigned int                    this_mpi_process;
  
  bool                              	converged;

  std::unique_ptr<DescentStepBase<dim>> gf_G;
  std::unique_ptr<DescentStepBase<dim>> gf_F;
  
  GFStepType 			    	MethodOperatorG;  
  GFStepType 			    	MethodOperatorF;

  //double				global_T; //just parareal
  double 			    	step_size_G;
  double 			    	step_size_F;
  unsigned int 		            	n_iter_G;
  unsigned int 		            	n_iter_F;
  

  VT::VectorArrayType	                F_vectors;
  VT::VectorArrayType               	G_old_vectors;
  VT::VectorArrayType               	G_new_vectors;
  VT::VectorArrayType               	new_yu_vectors;
  VT::VectorArrayType               	old_yu_vectors; 

  double				epsilon=1e-3;
  VT::ConvergenceTuple                 	converged_vec;
  unsigned int                      	converged_rank;
  unsigned int				num_iter_conv;
};

template<int dim>
ParallelRootBase<dim>::ParallelRootBase(const std::string& linear_system_filename)
:     NumericalAlgorithmBase<dim>(linear_system_filename)
    , mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , converged(false)
    , F_vectors(n_mpi_processes)
    , G_old_vectors(n_mpi_processes)
    , G_new_vectors(n_mpi_processes)
    , new_yu_vectors(n_mpi_processes)
    , old_yu_vectors(n_mpi_processes)
    , converged_vec(n_mpi_processes)
{}

template<int dim>
ParallelRootBase<dim>::ParallelRootBase(const double gamma_val, const double nu_val, const unsigned int N_grid)
:     NumericalAlgorithmBase<dim>(gamma_val, nu_val, N_grid)
    , mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , converged(false)
    , F_vectors(n_mpi_processes)
    , G_old_vectors(n_mpi_processes)
    , G_new_vectors(n_mpi_processes)
    , new_yu_vectors(n_mpi_processes)
    , old_yu_vectors(n_mpi_processes)
    , converged_vec(n_mpi_processes)
{}


template<int dim>
bool ParallelRootBase<dim>::check_convergence()
{
    for (unsigned int i=0; i<n_mpi_processes; i++)
      if (std::get<0>(converged_vec[i]) == true)
      {
          converged_rank = i;
          num_iter_conv = std::get<1>(converged_vec[i]);
          return true;
      }
    return false;
}

template<int dim>
bool ParallelRootBase<dim>::check_pr_interval_convergence()
{
    Vector<double> y_diff;
    Vector<double> u_diff;

    for (unsigned int i=0; i<n_mpi_processes; i++)
    {
      y_diff = new_yu_vectors[i][0];
      y_diff.add(-1, old_yu_vectors[i][0]);
      u_diff = new_yu_vectors[i][1];
      u_diff.add(-1, old_yu_vectors[i][1]);

      if (y_diff.linfty_norm() > epsilon || u_diff.linfty_norm() > epsilon)
        return false;
    }
    return true;
}


template<int dim>
void ParallelRootBase<dim>::run()
{
  unsigned int total_n_it=0;
  std::cout << "Initialization" << std::endl;
  for (unsigned int i=0; i<n_mpi_processes; i++)
  {
    gf_G->run_gf(n_iter_G);

    Vector<double> y = gf_G->get_y_vec();
    Vector<double> u = gf_G->get_u_vec();

    G_old_vectors[i][0] = y;
    G_old_vectors[i][1] = u;
    old_yu_vectors[i][0] = y;
    old_yu_vectors[i][1] = u;

    gf_G->output_iteration_results();
  }
  G_new_vectors = G_old_vectors;
  new_yu_vectors = old_yu_vectors;
  gf_G->output_results_vectors();
  total_n_it += n_iter_G*n_mpi_processes;

  // Root computes its own F operator in the first interval [t0, t1]: i.e. computes F at t1
  gf_F->run_gf(n_iter_F);
  F_vectors[0][0] = gf_F->get_y_vec();
  F_vectors[0][1] = gf_F->get_u_vec();
  converged_vec[0] = gf_F->convergence_info();
  total_n_it += n_iter_F;
  
  // Update the value of y and u obtained from root (operator G cancels out)
  new_yu_vectors[0][0] = F_vectors[0][0];
  new_yu_vectors[0][1] = F_vectors[0][1];

  unsigned int it=0;
  while(!converged)
  {
      std::cout << "Iteration n: " << it+1 << std::endl;
      // Send Results to all ranks
      for (unsigned int rank=1; rank<n_mpi_processes; rank++)
      {
        Utilities::MPI::isend(new_yu_vectors[rank-1], mpi_communicator, rank);
      }

      /*THIS BRINGS BACK OUR OLD PR THAT IMPROVES ALSO THE FIRST VALUE
      // Root computes its own F operator in the first interval [t0, t1]: i.e. computes F at t1
      gf_F->run_gf(this->n_iter_F);
      F_vectors[0][0] = gf_F->get_y_vec();
      F_vectors[0][1] = gf_F->get_u_vec();
      converged_vec[0] = gf_F->convergence_info();

      // Update the value of y and u obtained from root (coarse operator cancels out)
      new_yu_vectors[0][0] = F_vectors[0][0];
      new_yu_vectors[0][1] = F_vectors[0][1];
      */

      // Receive F operator and local convergence results from all ranks
      for (unsigned int rank=1; rank<n_mpi_processes; rank++)
      {
        VT::FutureTupleType future = Utilities::MPI::irecv<VT::TupleType>(mpi_communicator, rank);
        VT::TupleType fut_tuple = future.get();
        F_vectors[rank] = std::get<0>(fut_tuple);
        converged_vec[rank] = std::get<1>(fut_tuple);
      }
      total_n_it += n_iter_F;

      // Compute convergence and send result to other ranks
      converged = check_convergence();
      for (unsigned int rank=1; rank<n_mpi_processes; rank++)
      {
        Utilities::MPI::isend(converged, mpi_communicator, rank);
      }

      if(!converged)
      {
        // Compute G_k+1
        G_old_vectors = G_new_vectors;
        for (unsigned int rank=1; rank<n_mpi_processes; rank++)
        {
          // Update G
          gf_G->set_initial_vectors(new_yu_vectors[rank-1][0], new_yu_vectors[rank-1][1]);
          gf_G->run_gf(n_iter_G);
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
        total_n_it += n_iter_G*(n_mpi_processes-1);
        

        // Output results of this iteration process
        for (unsigned int rank=0; rank<n_mpi_processes; rank++)
        {
          gf_G->set_initial_vectors(new_yu_vectors[rank][0], new_yu_vectors[rank][1]);
          gf_G->output_iteration_results();
        }
        gf_G->output_results_vectors();
      }

      it++;

      if (check_pr_interval_convergence())
      {
        std::cout << "Interval shift!" << std::endl;
        std::cout << "Iteration n: " << it+1 << std::endl;
        //U_0 = U_N
        old_yu_vectors = new_yu_vectors;
        gf_G->set_initial_vectors(new_yu_vectors[n_mpi_processes-1][0], new_yu_vectors[n_mpi_processes-1][1]);
        gf_F->set_initial_vectors(new_yu_vectors[n_mpi_processes-1][0], new_yu_vectors[n_mpi_processes-1][1]);

        for (unsigned int i=0; i<n_mpi_processes; i++)
        {
          gf_G->run_gf(n_iter_G);

          Vector<double> y = gf_G->get_y_vec();
          Vector<double> u = gf_G->get_u_vec();

          G_old_vectors[i][0] = y;
          G_old_vectors[i][1] = u;
          new_yu_vectors[i][0] = y;
          new_yu_vectors[i][1] = u;

          gf_G->output_iteration_results();
        }
        total_n_it += n_iter_G*n_mpi_processes;
        
        // Root computes its own F operator in the first interval [t0, t1]: i.e. computes F at t1
        gf_F->run_gf(n_iter_F);
        F_vectors[0][0] = gf_F->get_y_vec();
        F_vectors[0][1] = gf_F->get_u_vec();
        converged_vec[0] = gf_F->convergence_info();
        total_n_it += n_iter_F;

        // Update the value of y and u obtained from root (coarse operator cancels out)
        new_yu_vectors[0][0] = F_vectors[0][0];
        new_yu_vectors[0][1] = F_vectors[0][1];

        G_new_vectors = G_old_vectors;
        gf_G->output_results_vectors();

        it++;
      }

      old_yu_vectors = new_yu_vectors;

  }

  // Output vectors
  std::cout << "Final results " << std::endl;
  std::cout << "Result obtained in: " << (n_iter_G*n_mpi_processes + n_iter_F)*it << " iterations" << std::endl;
  std::cout << "old Result obtained in: " << total_n_it-n_iter_F+num_iter_conv << " iterations" << std::endl;
  gf_G->set_initial_vectors(F_vectors[converged_rank][0], F_vectors[converged_rank][1]);
  gf_G->output_iteration_results();
  gf_G->output_results_vectors();
}


template<int dim>
class ParallelRankNBase: public NumericalAlgorithmBase<dim>
{
public:

  using VT = VectorTypes;

  ParallelRankNBase(const std::string& linear_system_filename);
  ParallelRankNBase(const double gamma_val, const double nu_val, const unsigned int N_grid);
  void run() override;
  

protected:

  MPI_Comm                    		mpi_communicator;
  const unsigned int          	        n_mpi_processes;
  const unsigned int                    this_mpi_process;
  
  bool 					converged;
  std::tuple<bool,unsigned int>         convergence_F;

  std::unique_ptr<DescentStepBase<dim>> gf_F;
   
  GFStepType 			    	MethodOperatorF;

  double 			    	step_size_F;
  unsigned int 		            	n_iter_F;
  
  VT::ArrayType                         initial_time_vectors; //to receive
  VT::ArrayType                         final_time_vectors; //to send
 	
};

template<int dim>
ParallelRankNBase<dim>::ParallelRankNBase(const std::string& linear_system_filename)
:     NumericalAlgorithmBase<dim>(linear_system_filename)
    , mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , converged(false)
{}

template<int dim>
ParallelRankNBase<dim>::ParallelRankNBase(const double gamma_val, const double nu_val, const unsigned int N_grid)
:     NumericalAlgorithmBase<dim>(gamma_val, nu_val, N_grid)
    , mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , converged(false)
{}


template<int dim>
void ParallelRankNBase<dim>::run()
{
  while(!converged)
  {
      // Receive initial conditions from root
      VT::FutureArrayType future = Utilities::MPI::irecv<VT::ArrayType>(mpi_communicator, root);
      initial_time_vectors = future.get();

      // Set received vectors as initial conditions of inner gf
      gf_F->set_initial_vectors(initial_time_vectors[0], initial_time_vectors[1]);

      gf_F->run_gf(n_iter_F);
      convergence_F = gf_F->convergence_info();
      final_time_vectors[0] = gf_F->get_y_vec();
      final_time_vectors[1] = gf_F->get_u_vec();

      VT::TupleType tuple_to_send = std::make_tuple(final_time_vectors, convergence_F);

      // Send tuple with the final vectors to the root and local convergence results
      Utilities::MPI::isend(tuple_to_send, mpi_communicator, root);

      // Receive convergence result from root
      Utilities::MPI::Future<bool> future1 = Utilities::MPI::irecv<bool>(mpi_communicator, root);
      converged = future1.get();
  }
}

