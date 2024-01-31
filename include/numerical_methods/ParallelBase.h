#include "NumericalAlgorithmBase.h"
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>

using namespace dealii;

struct VectorTypes{
    using ArrayType = std::array<Vector<double>, 2>;
    using VectorArrayType = std::vector<ArrayType>;
    using TupleType = std::tuple<ArrayType, bool>;
    using FutureArrayType = Utilities::MPI::Future<ArrayType>;
    using FutureTupleType = Utilities::MPI::Future<TupleType>;
};

constexpr unsigned int root = 0;


template<unsigned int dim>
class ParallelRootBase: public NumericalAlgorithmBase<dim>
{
public:

  using VT = VectorTypes;

  ParallelRootBase(const std::string& linear_system_filename);
  ParallelRootBase(const double gamma_val, const double nu_val, const unsigned int N_grid);
  

protected:

  bool check_convergence();

  MPI_Comm                    		mpi_communicator;
  const unsigned int          	        n_mpi_processes;
  const unsigned int                    this_mpi_process;
  
  bool                              	converged;

  std::unique_ptr<DescentStepBase<dim>> gf_G;
  std::unique_ptr<DescentStepBase<dim>> gf_F;
  
  GFStepType 			    	MethodOperatorG;  
  GFStepType 			    	MethodOperatorF;

  double 			    	step_size_G;
  double 			    	step_size_F;
  unsigned int 		            	n_iter_G;
  unsigned int 		            	n_iter_F;
  

  VT::VectorArrayType	                F_vectors;
  VT::VectorArrayType               	G_old_vectors;
  VT::VectorArrayType               	G_new_vectors;
  VT::VectorArrayType               	new_yu_vectors;

  std::vector<bool>                	converged_vec;
  unsigned int                      	converged_rank;
};

template<unsigned int dim>
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
    , converged_vec(n_mpi_processes)
{}

template<unsigned int dim>
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
    , converged_vec(n_mpi_processes)
{}


template<unsigned int dim>
bool ParallelRootBase<dim>::check_convergence()
{
    for (unsigned int i=0; i<n_mpi_processes; i++)
      if (converged_vec[i] == true)
      {
          converged_rank = i;
          return true;
      }
    return false;
}



template<unsigned int dim>
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
  bool				        convergence_F;

  std::unique_ptr<DescentStepBase<dim>> gf_F;
   
  GFStepType 			    	MethodOperatorF;

  double 			    	step_size_F;
  unsigned int 		            	n_iter_F;
  
  VT::ArrayType                         initial_time_vectors; //to receive
  VT::ArrayType                         final_time_vectors; //to send
 	
};

template<unsigned int dim>
ParallelRankNBase<dim>::ParallelRankNBase(const std::string& linear_system_filename)
:     NumericalAlgorithmBase<dim>(linear_system_filename)
    , mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , converged(false)
{}

template<unsigned int dim>
ParallelRankNBase<dim>::ParallelRankNBase(const double gamma_val, const double nu_val, const unsigned int N_grid)
:     NumericalAlgorithmBase<dim>(gamma_val, nu_val, N_grid)
    , mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , converged(false)
{}


template<unsigned int dim>
void ParallelRankNBase<dim>::run()
{
  while(!converged)
  {
      // Receive initial conditions from root
      VT::FutureArrayType future = Utilities::MPI::irecv<VT::ArrayType>(mpi_communicator, root);
      initial_time_vectors = future.get();

      // Set received vectors as initial conditions of inner gf
      gf_F->set_initial_vectors(initial_time_vectors[0], initial_time_vectors[1]);

      gf_F->run(n_iter_F);
      convergence_F = std::get<0>(gf_F->convergence_info());
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

