#ifndef PARALLEL_ALGORITHMS_H
#define PARALLEL_ALGORITHMS_H

#include "NumericalAlgorithmBase.h"
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>

using namespace dealii;

/**
 * @struct VectorTypes
 * @brief Defines types for vectors used in Parareal and ParaFlow.
 */
struct VectorTypes{
    using ArrayType = std::array<Vector<double>, 2>;
    using VectorArrayType = std::vector<ArrayType>;
    using TupleType = std::tuple<ArrayType, bool>;
    using FutureArrayType = Utilities::MPI::Future<ArrayType>;
    using FutureTupleType = Utilities::MPI::Future<TupleType>;
};

//! Root process rank in MPI communicator
constexpr unsigned int root = 0;

/**
 * @class ParallelRootBase
 * @brief Base class for implementing parallel algorithms on root process.
 * 
 * @tparam dim The space dimension (2 or 3)
 *
 * This class serves as the base for implementing parallel algorithms that execute on the root process.
 */
template<unsigned int dim>
class ParallelRootBase: public NumericalAlgorithmBase<dim>
{
public:

  using VT = VectorTypes;

  /**
   * @brief Constructor for ParallelRootBase.
   *
   * @param linear_system_filename The name of the file containing linear system parameters.
   */
  ParallelRootBase(const std::string& linear_system_filename);
  
  /**
   * @brief Constructor for ParallelRootBase.
   *
   * @param gamma_val The value of the gamma parameter.
   * @param nu_val The value of the nu parameter.
   * @param N_grid The number of grid refinements.
   */
  ParallelRootBase(const double gamma_val, const double nu_val, const unsigned int N_grid);
  

protected:

  /**
   * @brief Checks the convergence status of the algorithm, by checking if any operator F converged.
   *
   * @return True if converged, false otherwise.
   */
  bool check_convergence();

  MPI_Comm                    		mpi_communicator; //! MPI communicator
  const unsigned int          	        n_mpi_processes;  //! Number of processes in the communicator
  const unsigned int                    this_mpi_process; //! Rank of this MPI process
  
  bool                              	converged; //! convergence flag

  std::unique_ptr<DescentStepBase<dim>> gf_G; //! Pointer to the operator G update rule object
  std::unique_ptr<DescentStepBase<dim>> gf_F; //! Pointer to the operator F update rule object
  
  GFStepType 			    	MethodOperatorG;   //! Update rule type of G
  GFStepType 			    	MethodOperatorF;   //! Update rule type of F

  double 			    	step_size_G; //! step size of G
  double 			    	step_size_F; //! step size of F
  unsigned int 		            	n_iter_G; //! number of iterations of G
  unsigned int 		            	n_iter_F; //! number of iterations of F
  

  VT::VectorArrayType	                F_vectors; //! stores vectors y and u computed by each operator F
  VT::VectorArrayType               	G_old_vectors; //! stores vectors y and u computed by operator G at step k
  VT::VectorArrayType               	G_new_vectors; //! stores vectors y and u computed by operator G at step k+1
  VT::VectorArrayType               	new_yu_vectors; //! stores vectors y and u obtained by performing the correction iteration

  std::vector<bool>                	converged_vec; //! Convergence of each operator F
  unsigned int                      	converged_rank; //! Rank of the process whose operator F converged
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


/**
 * @class ParallelRankNBase
 * @brief Base class for implementing parallel algorithms on all processes except root.
 *
 * @tparam dim The space dimension (2 or 3)
 *
 * This class serves as the base for implementing parallel algorithms that execute on all processes except the root.
 */
template<unsigned int dim>
class ParallelRankNBase: public NumericalAlgorithmBase<dim>
{
public:

  using VT = VectorTypes;
  
  /**
   * @brief Constructor for ParallelRankNBase.
   *
   * @param linear_system_filename The name of the file containing linear system parameters.
   */
  ParallelRankNBase(const std::string& linear_system_filename);
  
  /**
   * @brief Constructor for ParallelRankNBase.
   *
   * @param gamma_val The value of the gamma parameter.
   * @param nu_val The value of the nu parameter.
   * @param N_grid The number of grid refinements.
   */
  ParallelRankNBase(const double gamma_val, const double nu_val, const unsigned int N_grid);
  
  /**
   * @brief Runs the parallel algorithm.
   */
  void run() override;
  

protected:

  
  MPI_Comm                    		mpi_communicator; //! MPI communicator
  const unsigned int          	        n_mpi_processes;  //! Number of processes in the communicator
  const unsigned int                    this_mpi_process; //! Rank of this MPI process
  
  bool 					converged; //! convergence flag of the algorithm
  bool				        convergence_F; //! convergence flag of the operator F

  std::unique_ptr<DescentStepBase<dim>> gf_F; //! Pointer to the operator F update rule object
   
  GFStepType 			    	MethodOperatorF; //! Update rule type for F

  double 			    	step_size_F; //! step size of operator F
  unsigned int 		            	n_iter_F; //! number of iterations of operator F
  
  VT::ArrayType                         initial_time_vectors; //! Initial vectors for F received by root
  VT::ArrayType                         final_time_vectors; //! Final vectors obtained by F and sent to root
 	
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
      //! Receive initial conditions from root
      VT::FutureArrayType future = Utilities::MPI::irecv<VT::ArrayType>(mpi_communicator, root);
      initial_time_vectors = future.get();

      //! Set received vectors as initial conditions of operator F
      gf_F->set_initial_vectors(initial_time_vectors[0], initial_time_vectors[1]);

      gf_F->run(n_iter_F);
      convergence_F = std::get<0>(gf_F->convergence_info());
      final_time_vectors[0] = gf_F->get_y_vec();
      final_time_vectors[1] = gf_F->get_u_vec();

      VT::TupleType tuple_to_send = std::make_tuple(final_time_vectors, convergence_F);

      //! Send tuple with the vectors obtained by operator F and its convergence to the root 
      Utilities::MPI::isend(tuple_to_send, mpi_communicator, root);

      //! Receive convergence result from root
      Utilities::MPI::Future<bool> future1 = Utilities::MPI::irecv<bool>(mpi_communicator, root);
      converged = future1.get();
  }
}

#endif // PARALLEL_ALGORITHMS_H
