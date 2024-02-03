#ifndef GRADIENT_FLOW_H
#define GRADIENT_FLOW_H

#include "NumericalAlgorithmBase.h"

using namespace dealii;


/**
 * @class GradientFlow
 * @brief Class for implementing the GradientFlow algorithm.
 *
 * @tparam dim The space dimension (2 or 3)
 *
 * This class implements the GradientFlow algorithm based on the NumericalAlgorithmBase class.
 */
template<unsigned int dim>
class GradientFlow: public NumericalAlgorithmBase<dim>
{
public:

  /**
     * @brief Constructor for GradientFlow.
     *
     * @param linear_system_filename The name of the file containing linear system parameters.
     * @param ParaFlowS_params_filename The name of the file containing GradientFlow parameters.
     */
  GradientFlow(const std::string& linear_system_filename, const std::string& ParaFlowS_params_filename);
  
  /**
     * @brief Constructor for GradientFlow.
     *
     * @param gamma_val The value of the gamma parameter.
     * @param nu_val The value of the nu parameter.
     * @param N_grid The number of grid refinements.
     * @param GF_params_filename The name of the file containing GradientFlow parameters.
     */
  GradientFlow(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& GF_params_filename);
  
  /**
     * @brief Runs the GradientFlow algorithm.
     */
  void run() override;

private:
 
  /**
     * @brief Reads GradientFlow method parameters from a file.
     *
     * @param filename The name of the file containing parameters.
     */
  void get_numerical_method_params(const std::string& filename) override;

  
  std::unique_ptr<DescentStepBase<dim>> gf; //! Pointer to object derived from DescentStepBase 
  
  GFStepType 			    MethodOperatorGF;  //! Update rule type
  
  double 			    step_size; //! Update rule step size

};


template<unsigned int dim>
GradientFlow<dim>::GradientFlow(const std::string& linear_system_filename, const std::string& GF_params_filename)
:        NumericalAlgorithmBase<dim>(linear_system_filename)
{	
  get_numerical_method_params(GF_params_filename);
  gf = this->create_GF(MethodOperatorGF);
  gf->set_step_size(step_size);
}

template<unsigned int dim>
GradientFlow<dim>::GradientFlow(const double gamma_val, const double nu_val, const unsigned int N_grid, const std::string& GF_params_filename)
:         NumericalAlgorithmBase<dim>(gamma_val, nu_val, N_grid)
{
  get_numerical_method_params(GF_params_filename);
  gf = this->create_GF(MethodOperatorGF);
  gf->set_step_size(step_size);
}

template<unsigned int dim>
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



template<unsigned int dim>
void GradientFlow<dim>::run()
{
  gf->run();
}

#endif // GRADIENT_FLOW_H
