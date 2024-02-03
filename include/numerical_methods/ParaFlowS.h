#ifndef PARA_FLOW_S_H
#define PARA_FLOW_S_H

#include "NumericalAlgorithmBase.h"


using namespace dealii;

/**
 * @struct VectorTypes
 * @brief Contains type aliases for vectors used in ParaFlowS.
 */
struct VectorTypes{
    using ArrayType = std::array<Vector<double>, 2>;
    using VectorArrayType = std::vector<ArrayType>;
};


/**
 * @class ParaFlowS
 * @brief Implements the ParaFlowS algorithm.
 *
 * This class implements the ParaFlowS algorithm.
 */
template<unsigned int dim>
class ParaFlowS: public NumericalAlgorithmBase<dim>
{
public:
  using VT = VectorTypes;

  /**
   * @brief Constructor for ParaFlowS.
   *
   * @param linear_system_filename The name of the file containing linear system parameters.
   * @param ParaFlowS_params_filename The name of the file containing parameters for ParaFlowS.
   */
  ParaFlowS(const std::string& linear_system_filename, const std::string& ParaFlowS_params_filename);
  
  /**
   * @brief Constructor for ParaFlowS.
   *
   * @param gamma_val The value of the gamma parameter.
   * @param nu_val The value of the nu parameter.
   * @param N_grid The number of grid refinements.
   * @param ParaFlowS_params_filename The name of the file containing parameters for ParaFlowS.
   */
  ParaFlowS(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& ParaFlowS_params_filename);
  
  /**
   * @brief Runs the ParaFlowS algorithm.
   */
  void run() override;

private:

  /**
   * @brief Retrieves the ParaFlowS parameters from a file.
   *
   * @param filename The name of the file containing ParaFlowS parameters.
   */
  void get_numerical_method_params(const std::string& filename) override;


  bool                              converged;  //! convergence flag

  std::unique_ptr<DescentStepBase<dim>> gf_G;  //! Pointer to the operator G update rule object
  std::unique_ptr<DescentStepBase<dim>> gf_F;  //! Pointer to the operator F update rule object
  
  GFStepType 			    MethodOperatorG;  //! Update rule type of G
  GFStepType 			    MethodOperatorF;  //! Update rule type of F
  
  unsigned int 			    N; //! Parameter of ParaFlowS
  double 			    step_size_G; //! step size of G
  double 			    step_size_F; //! step size of F
  unsigned int 		            n_iter_G; //! number of iterations of G
  unsigned int 		            n_iter_F; //! number of iterations of F
  

  VT::ArrayType	                    F_vectors; //! stores vectors y and u computed by operator F
  VT::VectorArrayType               G_old_vectors; //! stores vectors y and u computed by operator G at step k
  VT::VectorArrayType               G_new_vectors; //! stores vectors y and u computed by operator G at step k+1
  VT::VectorArrayType               new_yu_vectors; //! stores vectors y and u obtained by performing the correction iteration

  std::vector<double>               J_eval; //! vector containing the value of J for each y and u in new_yu_vectors

};


template<unsigned int dim>
ParaFlowS<dim>::ParaFlowS(const std::string& linear_system_filename, const std::string& ParaFlowS_params_filename)
:        NumericalAlgorithmBase<dim>(linear_system_filename), converged(false)
{	
  get_numerical_method_params(ParaFlowS_params_filename);
  gf_G = this->create_GF(MethodOperatorG);
  gf_F = this->create_GF(MethodOperatorF);
  
  gf_G->set_step_size(step_size_G);
  gf_F->set_step_size(step_size_F);
  
  G_old_vectors.resize(N);
  G_new_vectors.resize(N);
  new_yu_vectors.resize(N);
  J_eval.resize(N);
}

template<unsigned int dim>
ParaFlowS<dim>::ParaFlowS(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& ParaFlowS_params_filename)
:         NumericalAlgorithmBase<dim>(gamma_val, nu_val, N_grid), converged(false)
{
  get_numerical_method_params(ParaFlowS_params_filename);
  gf_G = this->create_GF(MethodOperatorG);
  gf_F = this->create_GF(MethodOperatorF);
  
  gf_G->set_step_size(step_size_G);
  gf_F->set_step_size(step_size_F);
  
  G_old_vectors.resize(N);
  G_new_vectors.resize(N);
  new_yu_vectors.resize(N);
  J_eval.resize(N);
}

template<unsigned int dim>
void ParaFlowS<dim>::get_numerical_method_params(const std::string& filename)
{	
  std::ifstream file(filename);

  json parameters;
  file >> parameters;
  file.close();

  N = parameters["N"];
  step_size_G = parameters["step_size_G"];
  step_size_F = parameters["step_size_F"];
  n_iter_G = parameters["n_iter_G"];
  n_iter_F = parameters["n_iter_F"];
  
  std::string opG = parameters["Solver G"];
  std::string opF = parameters["Solver F"];
  
  if (opG == "Euler")
  	MethodOperatorG = GFStepType::EULER;
  else if (opG == "Adam")
  	MethodOperatorG = GFStepType::ADAM;
  else
  	std::cerr << "Solver not implemented.\n" << std::endl;
  	
   if (opF == "Euler")
  	MethodOperatorF = GFStepType::EULER;
  else if (opF == "Adam")
  	MethodOperatorF = GFStepType::ADAM;
  else
  	std::cerr << "Solver not implemented.\n" << std::endl;
  
}



template<unsigned int dim>
void ParaFlowS<dim>::run()
{
  unsigned int total_n_it=0; //! Total number of update rule iterations required
  std::cout << "Initialization" << std::endl;
  for (unsigned int i=0; i<N; i++)
  {
    gf_G->run(n_iter_G);
    J_eval[i] = gf_G->evaluate_J();

    Vector<double> y = gf_G->get_y_vec();
    Vector<double> u = gf_G->get_u_vec();

    G_new_vectors[i][0] = y;
    G_new_vectors[i][1] = u;
    new_yu_vectors[i][0] = y;
    new_yu_vectors[i][1] = u;

    gf_G->output_iteration_results();
  }
  G_old_vectors = G_new_vectors;
  gf_G->output_results_vectors();
  total_n_it += n_iter_G*N;

  unsigned int idx_minJ; //! index of the minimum J in J_eval
  unsigned int local_convergence_iter(n_iter_F); 

  unsigned int it=0;
  while(!converged)
  {
      std::cout << "Iteration n: " << it+1 << std::endl;
      
      //! We find the index of the values that minimize the cost functional and set the values as initial conditions for F
      idx_minJ = std::min_element(J_eval.begin(), J_eval.end()) - J_eval.begin();
      gf_F->set_initial_vectors(new_yu_vectors[idx_minJ][0], new_yu_vectors[idx_minJ][1]);
      gf_F->run(n_iter_F);
      gf_F->output_iteration_results();
      total_n_it += n_iter_F;
      
      J_eval[0] = gf_F->evaluate_J();
      F_vectors[0] = gf_F->get_y_vec();
      F_vectors[1] = gf_F->get_u_vec();
      converged = std::get<0>(gf_F->convergence_info());
     

      if(converged)
      {
      	local_convergence_iter = std::get<1>(gf_F->convergence_info()); //! Number of iterations of F to converge
      	total_n_it = total_n_it - n_iter_F + local_convergence_iter;
      }
      else
      {
        //! Perform correction iteration
        //! Update the values of y and u for i=0
        new_yu_vectors[0][0] = F_vectors[0];
        new_yu_vectors[0][1] = F_vectors[1];

        G_old_vectors = G_new_vectors;
        for (unsigned int i=1; i<N; i++)
        {
          //! Update G
          gf_G->set_initial_vectors(new_yu_vectors[i-1][0], new_yu_vectors[i-1][1]);
          gf_G->run(n_iter_G);
          G_new_vectors[i][0] = gf_G->get_y_vec();
          G_new_vectors[i][1] = gf_G->get_u_vec();

          //! Update values of y and u with the correction iteration
          Vector<double> y_temp(G_new_vectors[i][0]);

          y_temp -= G_old_vectors[i][0];
          y_temp += F_vectors[0];

          Vector<double> u_temp(G_new_vectors[i][1]);
          u_temp -= G_old_vectors[i][1];
          u_temp += F_vectors[1];

          new_yu_vectors[i][0] = y_temp;
          new_yu_vectors[i][1] = u_temp;

          
          gf_G->set_initial_vectors(new_yu_vectors[i][0], new_yu_vectors[i][1]);
          gf_G->output_iteration_results();
          J_eval[i] = gf_G->evaluate_J();

        }
        
        total_n_it += n_iter_G*(N-1);

      }

      it++;
  }

  //! Output of final results
  std::cout << "Final result " << std::endl;
  std::cout << "ParaFlowS converged in: " << total_n_it << " iterations" << std::endl;
  gf_G->set_initial_vectors(F_vectors[0], F_vectors[1]);
  gf_G->output_iteration_results(); //! outputs the cost functional J and the norm of g at the solution vectors y_vec and u_vec
  gf_G->output_results_vectors();  //! outputs the obtained y_vec and u_vec in .vtk files for visualization
}


#endif // PARA_FLOW_S_H
