#ifndef NUMERICAL_ALGORITHM_BASE_H
#define NUMERICAL_ALGORITHM_BASE_H


#include "DescentStep.h"
#include <fstream>
#include <nlohmann/json.hpp>


using namespace dealii;
using json = nlohmann::json;


/**
 * @class NumericalAlgorithmBase
 * @brief Base class for all numerical methods implemented.
 * 
 * @tparam dim The space dimension (2 or 3)
 *
 * This class serves as a base for the numerical methods and provides common functionalities
 * such as reading parameters of the linear system from file and creating pointers to DescentStepBase objects.
 */
template<unsigned int dim>
class NumericalAlgorithmBase
{
public:

  /**
     * @brief Default constructor for NumericalAlgorithmBase.
     */
  NumericalAlgorithmBase(){};
  
  /**
     * @brief Constructor for NumericalAlgorithmBase.
     *
     * @param filename The name of the file containing the linear system parameters.
     */
  NumericalAlgorithmBase(const std::string& filename);
  
  /**
     * @brief Constructor for NumericalAlgorithmBase which accepts in input the linear system parameters.
     *
     * @param gamma_val The value of the gamma parameter.
     * @param nu_val The value of the nu parameter.
     * @param N The number of grid refinements.
     */
  NumericalAlgorithmBase(const double gamma_val, const double nu_val, const unsigned int N);
  
  /**
     * @brief Runs the numerical method.
     */
  virtual void run() = 0;
 

protected:
	
  /**
     * @brief Reads the parameters that characterize the numerical method from a file.
     *
     * @param filename The name of the file containing parameters.
     */
  virtual void get_numerical_method_params(const std::string& filename) = 0;
  
  /**
     * @brief Creates an object derived from DescentStepBase, depending on the update rule chosen.
     *
     * @param GFMethod The type of update rule.
     * @return A unique pointer to the created derived object from DescentStepBase.
     */
  std::unique_ptr<DescentStepBase<dim>> create_GF(const GFStepType GFMethod);
  
  /**
     * @brief Reads linear system parameters from a file.
     *
     * @param filename The name of the file containing parameters.
     */
  void get_linear_system_params(const std::string& filename);
  
  
  //! Linear system parameters
  double 	gamma;
  double 	nu;
  unsigned int  grid_refinement;

};

template<unsigned int dim>
NumericalAlgorithmBase<dim>::NumericalAlgorithmBase(const std::string& filename)
{
	get_linear_system_params(filename);
}

template<unsigned int dim>
NumericalAlgorithmBase<dim>::NumericalAlgorithmBase(const double gamma_val, const double nu_val, const unsigned int N)
: gamma(gamma_val), nu(nu_val), grid_refinement(N)
{}


template<unsigned int dim>
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



template<unsigned int dim>
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

#endif // NUMERICAL_ALGORITHM_BASE_H

