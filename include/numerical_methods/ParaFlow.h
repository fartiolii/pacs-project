#ifndef PARALLEL_FLOW_H
#define PARALLEL_FLOW_H


#include "ParallelBase.h"

using namespace dealii;

/**
 * @class ParaFlowRoot
 * @brief Implements the ParaFlow algorithm on the root process.
 * 
 * @tparam dim The space dimension (2 or 3)
 *
 * This class implements the ParaFlow algorithm executing on the root process.
 */
template<unsigned int dim>
class ParaFlowRoot: public ParallelRootBase<dim>
{
public:

  using VT = VectorTypes;

  /**
   * @brief Constructor for ParaFlowRoot.
   *
   * @param linear_system_filename The name of the file containing linear system parameters.
   * @param ParaFlow_params_filename The name of the file containing parameters for ParaFlow.
   */
  ParaFlowRoot(const std::string& linear_system_filename, const std::string& ParaFlow_params_filename);
  
  /**
   * @brief Constructor for ParaFlowRoot.
   *
   * @param gamma_val The value of the gamma parameter.
   * @param nu_val The value of the nu parameter.
   * @param N_grid The number of grid refinements.
   * @param ParaFlow_params_filename The name of the file containing parameters for ParaFlow.
   */
  ParaFlowRoot(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& ParaFlow_params_filename);
  
  /**
   * @brief Runs the ParaFlow algorithm on the root process.
   */
  void run() override;
  
private:
  
  /**
   * @brief Retrieves the ParaFlow parameters from a file.
   *
   * @param filename The name of the file containing ParaFlow parameters.
   */
  void get_numerical_method_params(const std::string& filename) override; 	
};

template<unsigned int dim>
ParaFlowRoot<dim>::ParaFlowRoot(const std::string& linear_system_filename, const std::string& ParaFlow_params_filename)
:     ParallelRootBase<dim>(linear_system_filename)
{
  get_numerical_method_params(ParaFlow_params_filename);
  this->gf_G = this->create_GF(this->MethodOperatorG);
  this->gf_F = this->create_GF(this->MethodOperatorF);
  
  this->gf_G->set_step_size(this->step_size_G);
  this->gf_F->set_step_size(this->step_size_F);
}

template<unsigned int dim>
ParaFlowRoot<dim>::ParaFlowRoot(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& ParaFlow_params_filename)
:     ParallelRootBase<dim>(gamma_val, nu_val, N_grid)
{
  get_numerical_method_params(ParaFlow_params_filename);
  this->gf_G = this->create_GF(this->MethodOperatorG);
  this->gf_F = this->create_GF(this->MethodOperatorF);
  
  this->gf_G->set_step_size(this->step_size_G);
  this->gf_F->set_step_size(this->step_size_F);
}



template<unsigned int dim>
void ParaFlowRoot<dim>::get_numerical_method_params(const std::string& filename)
{	
  std::ifstream file(filename);

  json parameters;
  file >> parameters;
  file.close();

  this->step_size_G = parameters["step_size_G"];
  this->step_size_F = parameters["step_size_F"];
  this->n_iter_G = parameters["n_iter_G"];
  this->n_iter_F = parameters["n_iter_F"];
  
  std::string opG = parameters["Solver G"];
  std::string opF = parameters["Solver F"];
  
  if (opG == "Euler")
  	this->MethodOperatorG = GFStepType::EULER;
  else if (opG == "Adam")
  	this->MethodOperatorG = GFStepType::ADAM;
  else
  	std::cerr << "Solver not implemented.\n" << std::endl;
  	
   if (opF == "Euler")
  	this->MethodOperatorF = GFStepType::EULER;
  else if (opF == "Adam")
  	this->MethodOperatorF = GFStepType::ADAM;
  else
  	std::cerr << "Solver not implemented.\n" << std::endl;
  
}


template<unsigned int dim>
void ParaFlowRoot<dim>::run()
{
  unsigned int total_n_it=0; //! Counter of update rule iterations in series
  std::cout << "Initialization" << std::endl;
  for (unsigned int i=0; i<this->n_mpi_processes; i++)
  {
    this->gf_G->run(this->n_iter_G);

    Vector<double> y = this->gf_G->get_y_vec();
    Vector<double> u = this->gf_G->get_u_vec();

    this->G_new_vectors[i][0] = y;
    this->G_new_vectors[i][1] = u;    
    this->new_yu_vectors[i][0] = y;
    this->new_yu_vectors[i][1] = u;

    this->gf_G->output_iteration_results();
  }
  this->G_old_vectors = this->G_new_vectors;
  total_n_it += this->n_iter_G*this->n_mpi_processes;

  unsigned int it=0;
  while(!this->converged)
  {
      std::cout << "Iteration n: " << it+1 << std::endl;
      //! Root sends initial conditions to all the other ranks
      for (unsigned int rank=1; rank<this->n_mpi_processes; rank++)
      {
        Utilities::MPI::isend(this->new_yu_vectors[rank-1], this->mpi_communicator, rank);
      }
      
      //! Root propagates its F operator 
      this->gf_F->run(this->n_iter_F);
      this->F_vectors[0][0] = this->gf_F->get_y_vec();
      this->F_vectors[0][1] = this->gf_F->get_u_vec();
      this->converged_vec[0] = std::get<0>(this->gf_F->convergence_info());
	
      //! Root receives F operator results and convergence results from all the other ranks
      for (unsigned int rank=1; rank<this->n_mpi_processes; rank++)
      {
        VT::FutureTupleType future = Utilities::MPI::irecv<VT::TupleType>(this->mpi_communicator, rank);
        VT::TupleType fut_tuple = future.get();
        this->F_vectors[rank] = std::get<0>(fut_tuple);
        this->converged_vec[rank] = std::get<1>(fut_tuple);
      }
      total_n_it += this->n_iter_F;

      //! Root computes convergence and sends the result to the other ranks
      this->converged = this->check_convergence();
      for (unsigned int rank=1; rank<this->n_mpi_processes; rank++)
      {
        Utilities::MPI::isend(this->converged, this->mpi_communicator, rank);
      }

      if(!this->converged)
      {
  	//! Perform correction iteration
	//! Update the value of y and u obtained from root 
	this->new_yu_vectors[0][0] = this->F_vectors[0][0];
	this->new_yu_vectors[0][1] = this->F_vectors[0][1];
	
        this->G_old_vectors = this->G_new_vectors;
        for (unsigned int rank=1; rank<this->n_mpi_processes; rank++)
        {
          //! Update G
          this->gf_G->set_initial_vectors(this->new_yu_vectors[rank-1][0], this->new_yu_vectors[rank-1][1]);
          this->gf_G->run(this->n_iter_G);
          this->G_new_vectors[rank][0] = this->gf_G->get_y_vec();
          this->G_new_vectors[rank][1] = this->gf_G->get_u_vec();

          //! Update values of y and u with correction iteration
          Vector<double> y_temp(this->G_new_vectors[rank][0]);

          y_temp -= this->G_old_vectors[rank][0];
          y_temp += this->F_vectors[rank][0];

          Vector<double> u_temp(this->G_new_vectors[rank][1]);
          u_temp -= this->G_old_vectors[rank][1];
          u_temp += this->F_vectors[rank][1];

          this->new_yu_vectors[rank][0] = y_temp;
          this->new_yu_vectors[rank][1] = u_temp;
        }
        total_n_it += this->n_iter_G*(this->n_mpi_processes-1);
        

        //! Output results of this iteration process
        for (unsigned int rank=0; rank<this->n_mpi_processes; rank++)
        {
          this->gf_G->set_initial_vectors(this->new_yu_vectors[rank][0], this->new_yu_vectors[rank][1]);
          this->gf_G->output_iteration_results();
        }
      }

      if(!this->converged)
      {
      	//! Interval shift by imposing U_0 = U_N
      	this->new_yu_vectors[0] = this->new_yu_vectors[this->n_mpi_processes-1];
      	this->gf_G->set_initial_vectors(this->new_yu_vectors[0][0], this->new_yu_vectors[0][1]);
      	this->gf_F->set_initial_vectors(this->new_yu_vectors[0][0], this->new_yu_vectors[0][1]);

	//! Operator G computes the initial conditions on the new interval
      	for (unsigned int i=0; i<this->n_mpi_processes; i++)
      	{
  	  this->gf_G->run(this->n_iter_G);

  	  Vector<double> y = this->gf_G->get_y_vec();
  	  Vector<double> u = this->gf_G->get_u_vec();

  	  this->G_new_vectors[i][0] = y;
  	  this->G_new_vectors[i][1] = u;
  	  this->new_yu_vectors[i][0] = y;
  	  this->new_yu_vectors[i][1] = u;
        }
        total_n_it += this->n_iter_G*this->n_mpi_processes;
      }
      it++;

  }

  //! Output of final results
  std::cout << "Final results " << std::endl;
  std::cout << "ParaFlow converged in: " << total_n_it << " iterations" << std::endl;
  this->gf_G->set_initial_vectors(this->F_vectors[this->converged_rank][0], this->F_vectors[this->converged_rank][1]);
  this->gf_G->output_iteration_results(); //! outputs the cost functional J and the norm of g at the solution vectors y_vec and u_vec
  this->gf_G->output_results_vectors(); //! outputs the obtained y_vec and u_vec in .vtk files for visualization
}

/**
 * @class ParaFlowRankN
 * @brief Implements the ParaFlow algorithm on all processes except root.
 * 
 * @tparam dim The space dimension (2 or 3)
 *
 * This class implements the ParaFlow algorithm executing on all processes except the root.
 */
template<unsigned int dim>
class ParaFlowRankN: public ParallelRankNBase<dim>
{
public:

  using VT = VectorTypes;
  
  /**
   * @brief Constructor for ParaFlowRankN.
   *
   * @param linear_system_filename The name of the file containing linear system parameters.
   * @param ParaFlow_params_filename The name of the file containing parameters for ParaFlow.
   */
  ParaFlowRankN(const std::string& linear_system_filename, const std::string& ParaFlow_params_filename);
  
  /**
   * @brief Constructor for ParaFlowRankN.
   *
   * @param gamma_val The value of the gamma parameter.
   * @param nu_val The value of the nu parameter.
   * @param N_grid The number of grid refinements.
   * @param ParaFlow_params_filename The name of the file containing parameters for ParaFlow.
   */
  ParaFlowRankN(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& ParaFlow_params_filename);  

private:

  /**
   * @brief Retrieves the ParaFlow parameters from a file.
   *
   * @param filename The name of the file containing ParaFlow parameters.
   */
  void get_numerical_method_params(const std::string& filename) override;
  
};

template<unsigned int dim>
ParaFlowRankN<dim>::ParaFlowRankN(const std::string& linear_system_filename, const std::string& ParaFlow_params_filename)
:     ParallelRankNBase<dim>(linear_system_filename)
{
  get_numerical_method_params(ParaFlow_params_filename);
  this->gf_F = this->create_GF(this->MethodOperatorF);
  this->gf_F->set_step_size(this->step_size_F);
}

template<unsigned int dim>
ParaFlowRankN<dim>::ParaFlowRankN(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& ParaFlow_params_filename)
:     ParallelRankNBase<dim>(gamma_val, nu_val, N_grid)
{
  get_numerical_method_params(ParaFlow_params_filename);
  this->gf_F = this->create_GF(this->MethodOperatorF);
  this->gf_F->set_step_size(this->step_size_F);
}

template<unsigned int dim>
void ParaFlowRankN<dim>::get_numerical_method_params(const std::string& filename)
{	
  std::ifstream file(filename);

  json parameters;
  file >> parameters;
  file.close();

  this->step_size_F = parameters["step_size_F"];
  this->n_iter_F = parameters["n_iter_F"];
  std::string opF = parameters["Solver F"];
  	
  if (opF == "Euler")
  	this->MethodOperatorF = GFStepType::EULER;
  else if (opF == "Adam")
  	this->MethodOperatorF = GFStepType::ADAM;
  else
  	std::cerr << "Solver not implemented.\n" << std::endl;
}

#endif // PARALLEL_FLOW_H
