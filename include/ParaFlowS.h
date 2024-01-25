#include "NumericalAlgorithmBase.h"


using namespace dealii;

struct VectorTypes{
    using ArrayType = std::array<Vector<double>, 2>;
    using VectorArrayType = std::vector<ArrayType>;
};


template<int dim>
class ParaFlowS: public NumericalAlgorithmBase<dim>
{
public:
  using VT = VectorTypes;

  ParaFlowS(const std::string& linear_system_filename, const std::string& ParaFlowS_params_filename);
  ParaFlowS(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& ParaFlowS_params_filename);
  void run() override;

private:

  void get_numerical_method_params(const std::string& filename) override;


  bool                              converged;

  std::unique_ptr<DescentStepBase<dim>> gf_G;
  std::unique_ptr<DescentStepBase<dim>> gf_F;
  
  GFStepType 			    MethodOperatorG;  
  GFStepType 			    MethodOperatorF;
  
  unsigned int 			    N;
  double 			    step_size_G;
  double 			    step_size_F;
  unsigned int 		            n_iter_G;
  unsigned int 		            n_iter_F;
  

  VT::ArrayType	                    F_vectors;
  VT::VectorArrayType               G_old_vectors;
  VT::VectorArrayType               G_new_vectors;
  VT::VectorArrayType               new_yu_vectors;

  std::vector<double>               J_eval;
  std::vector<double>               g_eval;

};


template<int dim>
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
  g_eval.resize(N);
}

template<int dim>
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
  g_eval.resize(N);
}

template<int dim>
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



template<int dim>
void ParaFlowS<dim>::run()
{
  unsigned int total_n_it=0;
  std::cout << "Initialization" << std::endl;
  for (unsigned int i=0; i<N; i++)
  {
    gf_G->run_gf(n_iter_G);
    J_eval[i] = gf_G->evaluate_J();

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
  total_n_it += n_iter_G*N;

  unsigned int idx_minJ; 
  unsigned int local_convergence_iter(n_iter_F);

  unsigned int it=0;
  while(!converged)
  {
      std::cout << "Iteration n: " << it+1 << std::endl;
      
      idx_minJ = std::min_element(J_eval.begin(), J_eval.end()) - J_eval.begin();
      gf_F->set_initial_vectors(new_yu_vectors[idx_minJ][0], new_yu_vectors[idx_minJ][1]);
      gf_F->run_gf(n_iter_F);
      total_n_it += n_iter_F;
      
      gf_F->output_iteration_results();
      J_eval[0] = gf_F->evaluate_J();
      g_eval[0] = gf_F->evaluate_g();
      F_vectors[0] = gf_F->get_y_vec();
      F_vectors[1] = gf_F->get_u_vec();
      converged = std::get<0>(gf_F->convergence_info());
      
      // Update the value of y and u of i=0
      new_yu_vectors[0][0] = F_vectors[0];
      new_yu_vectors[0][1] = F_vectors[1];
 

      if(converged)
      {
      	local_convergence_iter = std::get<1>(gf_F->convergence_info());
      	total_n_it = total_n_it - n_iter_F + local_convergence_iter;
      }
      else
      {
        // Compute G_k+1
        G_old_vectors = G_new_vectors;
        for (unsigned int i=1; i<N; i++)
        {
          /*
          if (i==0)
          {
          	gf_G->set_initial_vectors(new_yu_vectors[idx_minJ][0], new_yu_vectors[idx_minJ][1]);
          	G_old_vectors[i][0] = new_yu_vectors[idx_minJ][0];
                G_old_vectors[i][1] = new_yu_vectors[idx_minJ][1];
          }
          else
          {
          	gf_G->set_initial_vectors(new_yu_vectors[i-1][0], new_yu_vectors[i-1][1]);
          }*/
          // Update G
          gf_G->set_initial_vectors(new_yu_vectors[i-1][0], new_yu_vectors[i-1][1]);
          gf_G->run_gf(n_iter_G);
          G_new_vectors[i][0] = gf_G->get_y_vec();
          G_new_vectors[i][1] = gf_G->get_u_vec();
          //gf_G->output_iteration_results();

          // Update values of y and u
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
          g_eval[i] = gf_G->evaluate_g();

        }
        
        total_n_it += n_iter_G*(N-1);

        // Search for the index i that achieves the minimum and use that for the next iteration of F
        //idx_minJ = std::min_element(J_eval.begin(), J_eval.end()) - J_eval.begin();
        // If the constraint is far from being satisfied we choose the closest solution to satisfying it

        //if (g_eval[idx_minJ] > 1.5)
        //idx_minJ = std::min_element(g_eval.begin(), g_eval.end()) - g_eval.begin();

        //gf_F->set_initial_vectors(new_yu_vectors[idx_minJ][0], new_yu_vectors[idx_minJ][1]);

        // write to file intermediate results for visit
        gf_G->output_results_vectors();
      }

      it++;
  }

  // Output vectors
  std::cout << "Final result " << std::endl;
  std::cout << "Result obtained in: " << total_n_it << " iterations" << std::endl;
  std::cout << "old Result obtained in: " << n_iter_G*(N+(N-1)*it) + n_iter_F*(it-1)+local_convergence_iter << " iterations" << std::endl;
  gf_G->set_initial_vectors(F_vectors[0], F_vectors[1]);
  gf_G->output_iteration_results();
  gf_G->output_results_vectors();
  std::cout << "old Result obtained in: " << (n_iter_G*N + n_iter_F)*it << " iterations" << std::endl;
}
