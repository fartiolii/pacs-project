#include "DescentStep.h"
#include <fstream>
#include <nlohmann/json.hpp>


using namespace dealii;
using json = nlohmann::json;



template<int dim>
class NumericalAlgorithmBase
{
public:

  NumericalAlgorithmBase(){};
  NumericalAlgorithmBase(const std::string& filename);
  NumericalAlgorithmBase(const double gamma_val, const double nu_val, const unsigned int N);
  virtual void run() = 0;
 

protected:
	
  virtual void get_numerical_method_params(const std::string& filename) = 0;
  std::unique_ptr<DescentStepBase<dim>> create_GF(const GFStepType GFMethod);
  void get_linear_system_params(const std::string& filename);
  void update_parameters(const double gamma_val, const double nu_val, const unsigned int N);
  
  double 	gamma;
  double 	nu;
  unsigned int  grid_refinement;

};

template<int dim>
NumericalAlgorithmBase<dim>::NumericalAlgorithmBase(const std::string& filename)
{
	get_linear_system_params(filename);
}

template<int dim>
NumericalAlgorithmBase<dim>::NumericalAlgorithmBase(const double gamma_val, const double nu_val, const unsigned int N)
: gamma(gamma_val), nu(nu_val), grid_refinement(N)
{}


template<int dim>
void NumericalAlgorithmBase<dim>::get_linear_system_params(const std::string& filename)
{
  std::ifstream file(filename);

  json parameters;
  file >> parameters;
  file.close();

  gamma = parameters["gamma"];
  nu = parameters["nu"];
  grid_refinement = parameters["Num grid refinements"];
}

template<int dim>
void NumericalAlgorithmBase<dim>::update_parameters(const double gamma_val, const double nu_val, const unsigned int N)
{
  gamma = gamma_val;
  nu = nu_val;
  grid_refinement = N;
}


template<int dim>
std::unique_ptr<DescentStepBase<dim>> NumericalAlgorithmBase<dim>::create_GF(const GFStepType GFMethod)
{
  switch(GFMethod)
  {
        case GFStepType::EULER:
            return std::make_unique<DescentStepEuler<dim>>(gamma, nu, grid_refinement);

        case GFStepType::ADAM:
            return std::make_unique<DescentStepAdam<dim>>(gamma, nu, grid_refinement);

        default:
            std::cerr << "Solver not implemented.\n" << std::endl;
            return nullptr;
  }
}


