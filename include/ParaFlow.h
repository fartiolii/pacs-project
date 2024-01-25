#include "ParallelBase.h"

using namespace dealii;


template<int dim>
class ParaFlowRoot: public ParallelRootBase<dim>
{
public:

  using VT = VectorTypes;

  ParaFlowRoot(const std::string& linear_system_filename, const std::string& ParaReal_params_filename);
  ParaFlowRoot(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& ParaReal_params_filename);
  
private:

  void get_numerical_method_params(const std::string& filename) override; 	
};

template<int dim>
ParaFlowRoot<dim>::ParaFlowRoot(const std::string& linear_system_filename, const std::string& ParaReal_params_filename)
:     ParallelRootBase<dim>(linear_system_filename)
{
  get_numerical_method_params(ParaReal_params_filename);
  this->gf_G = this->create_GF(this->MethodOperatorG);
  this->gf_F = this->create_GF(this->MethodOperatorF);
  
  this->gf_G->set_step_size(this->step_size_G);
  this->gf_F->set_step_size(this->step_size_F);
}

template<int dim>
ParaFlowRoot<dim>::ParaFlowRoot(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& ParaReal_params_filename)
:     ParallelRootBase<dim>(gamma_val, nu_val, N_grid)
{
  get_numerical_method_params(ParaReal_params_filename);
  this->gf_G = this->create_GF(this->MethodOperatorG);
  this->gf_F = this->create_GF(this->MethodOperatorF);
  
  this->gf_G->set_step_size(this->step_size_G);
  this->gf_F->set_step_size(this->step_size_F);
}



template<int dim>
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



template<int dim>
class ParaFlowRankN: public ParallelRankNBase<dim>
{
public:

  using VT = VectorTypes;

  ParaFlowRankN(const std::string& linear_system_filename, const std::string& ParaFlow_params_filename);
  ParaFlowRankN(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& ParaFlow_params_filename);  

private:

  void get_numerical_method_params(const std::string& filename) override;
  
};

template<int dim>
ParaFlowRankN<dim>::ParaFlowRankN(const std::string& linear_system_filename, const std::string& ParaFlow_params_filename)
:     ParallelRankNBase<dim>(linear_system_filename)
{
  get_numerical_method_params(ParaFlow_params_filename);
  this->gf_F = this->create_GF(this->MethodOperatorF);
  this->gf_F->set_step_size(this->step_size_F);
}

template<int dim>
ParaFlowRankN<dim>::ParaFlowRankN(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& ParaFlow_params_filename)
:     ParallelRankNBase<dim>(gamma_val, nu_val, N_grid)
{
  get_numerical_method_params(ParaFlow_params_filename);
  this->gf_F = this->create_GF(this->MethodOperatorF);
  this->gf_F->set_step_size(this->step_size_F);
}

template<int dim>
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


