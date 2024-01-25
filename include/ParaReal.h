#include "ParallelBase.h"

using namespace dealii;

template<int dim>
class ParaRealRoot: public ParallelRootBase<dim>
{
public:

  using VT = VectorTypes;

  ParaRealRoot(const std::string& linear_system_filename, const std::string& ParaReal_params_filename);
  ParaRealRoot(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& ParaReal_params_filename);
  

private:

  void get_numerical_method_params(const std::string& filename) override;
  void set_number_iter();

  double	global_T; 
};

template<int dim>
ParaRealRoot<dim>::ParaRealRoot(const std::string& linear_system_filename, const std::string& ParaReal_params_filename)
:     ParallelRootBase<dim>(linear_system_filename)
{
  get_numerical_method_params(ParaReal_params_filename);
  this->gf_G = this->create_GF(this->MethodOperatorG);
  this->gf_F = this->create_GF(this->MethodOperatorF);
  
  this->gf_G->set_step_size(this->step_size_G);
  this->gf_F->set_step_size(this->step_size_F);
  
  set_number_iter();
}

template<int dim>
ParaRealRoot<dim>::ParaRealRoot(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& ParaReal_params_filename)
:     ParallelRootBase<dim>(gamma_val, nu_val, N_grid)
{
  get_numerical_method_params(ParaReal_params_filename);
  this->gf_G = this->create_GF(this->MethodOperatorG);
  this->gf_F = this->create_GF(this->MethodOperatorF);
  
  this->gf_G->set_step_size(this->step_size_G);
  this->gf_F->set_step_size(this->step_size_F);
  
  set_number_iter();
}



template<int dim>
void ParaRealRoot<dim>::get_numerical_method_params(const std::string& filename)
{	
  std::ifstream file(filename);

  json parameters;
  file >> parameters;
  file.close();

  global_T = parameters["T"];
  this->step_size_G = parameters["step_size_G"];
  this->step_size_F = parameters["step_size_F"];
  
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
void ParaRealRoot<dim>::set_number_iter()
{
    assert(this->n_mpi_processes != 0);

    this->n_iter_G = static_cast<unsigned int>(global_T/(this->n_mpi_processes*this->step_size_G));
    this->n_iter_F = static_cast<unsigned int>(global_T/(this->n_mpi_processes*this->step_size_F));
    std::cout << " n_iter_G: " << this->n_iter_G <<  " n_iter_F: " << this->n_iter_F << std::endl;
}




template<int dim>
class ParaRealRankN: public ParallelRankNBase<dim>
{
public:

  using VT = VectorTypes;

  ParaRealRankN(const std::string& linear_system_filename, const std::string& ParaReal_params_filename);
  ParaRealRankN(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& ParaReal_params_filename);
  

private:

  void get_numerical_method_params(const std::string& filename) override;
  void set_number_iter();

  double	global_T; 
 	
};

template<int dim>
ParaRealRankN<dim>::ParaRealRankN(const std::string& linear_system_filename, const std::string& ParaReal_params_filename)
:     ParallelRankNBase<dim>(linear_system_filename)
{
  get_numerical_method_params(ParaReal_params_filename);
  this->gf_F = this->create_GF(this->MethodOperatorF);
  this->gf_F->set_step_size(this->step_size_F);
  set_number_iter();
}

template<int dim>
ParaRealRankN<dim>::ParaRealRankN(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& ParaReal_params_filename)
:     ParallelRankNBase<dim>(gamma_val, nu_val, N_grid)
{
  get_numerical_method_params(ParaReal_params_filename);
  this->gf_F = this->create_GF(this->MethodOperatorF);
  this->gf_F->set_step_size(this->step_size_F);
  set_number_iter();
}

template<int dim>
void ParaRealRankN<dim>::get_numerical_method_params(const std::string& filename)
{	
  std::ifstream file(filename);

  json parameters;
  file >> parameters;
  file.close();

  global_T = parameters["T"];
  this->step_size_F = parameters["step_size_F"];
  
  std::string opF = parameters["Solver F"];
  	
  if (opF == "Euler")
  	this->MethodOperatorF = GFStepType::EULER;
  else if (opF == "Adam")
  	this->MethodOperatorF = GFStepType::ADAM;
  else
  	std::cerr << "Solver not implemented.\n" << std::endl;
  
}

template<int dim>
void ParaRealRankN<dim>::set_number_iter()
{
    assert(this->n_mpi_processes != 0);

    this->n_iter_F = static_cast<unsigned int>(global_T/(this->n_mpi_processes*this->step_size_F));
    std::cout <<  " n_iter_F: " << this->n_iter_F << std::endl;
}



