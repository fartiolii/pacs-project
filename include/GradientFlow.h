#include "NumericalAlgorithmBase.h"

using namespace dealii;



template<int dim>
class GradientFlow: public NumericalAlgorithmBase<dim>
{
public:
  GradientFlow(const std::string& linear_system_filename, const std::string& ParaFlowS_params_filename);
  GradientFlow(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& GF_params_filename);
  void run() override;

private:

  void get_numerical_method_params(const std::string& filename) override;


  std::unique_ptr<DescentStepBase<dim>> gf;
  
  GFStepType 			    MethodOperatorGF;  
  
  double 			    step_size;

};


template<int dim>
GradientFlow<dim>::GradientFlow(const std::string& linear_system_filename, const std::string& GF_params_filename)
:        NumericalAlgorithmBase<dim>(linear_system_filename)
{	
  get_numerical_method_params(GF_params_filename);
  gf = this->create_GF(MethodOperatorGF);
  gf->set_step_size(step_size);
}

template<int dim>
GradientFlow<dim>::GradientFlow(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& GF_params_filename)
:         NumericalAlgorithmBase<dim>(gamma_val, nu_val, N_grid)
{
  get_numerical_method_params(GF_params_filename);
  gf = this->create_GF(MethodOperatorGF);
  gf->set_step_size(step_size);
}

template<int dim>
void GradientFlow<dim>::get_numerical_method_params(const std::string& filename)
{	
  std::ifstream file(filename);

  json parameters;
  file >> parameters;
  file.close();

  step_size = parameters["step_size"];
  
  std::string solver = parameters["Solver"];
  
  if (solver == "Euler")
  	MethodOperatorGF = GFStepType::EULER;
  else if (solver == "Adam")
  	MethodOperatorGF = GFStepType::ADAM;
  else
  	std::cerr << "Solver not implemented.\n" << std::endl;  
}



template<int dim>
void GradientFlow<dim>::run()
{
  gf->run();
}
