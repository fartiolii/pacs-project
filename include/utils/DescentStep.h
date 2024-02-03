#ifndef DESCENT_STEP_METHODS_H
#define DESCENT_STEP_METHODS_H


#include "DescentStepBase.h"


using namespace dealii;

/**
 * @class DescentStepEuler
 * @brief Class representing a descent step using the Explicit Euler update rule.
 *
 * @tparam dim The space dimension (2 or 3)
 *
 * This class provides functionalities for performing a descent step using the Explicit Euler method.
 */
 
template<unsigned int dim>
class DescentStepEuler: public DescentStepBase<dim>
{
public:
  /**
     * @brief Default constructor for DescentStepEuler.
     */
  DescentStepEuler(){};
  
  /**
     * @brief Constructor for DescentStepEuler. Takes in input the input parameters for the LinearSystem class
     *
     * @param gamma_val The value of the gamma parameter.
     * @param nu_val The value of the nu parameter.
     * @param N The number of grid refinements.
     */
  DescentStepEuler(const double gamma_val, const double nu_val, const unsigned int N);

private:
  /**
     * @brief Performs a descent step for vectors y_vec and u_vec using the Explicit Euler method.
     */
  void vectors_iteration_step() override;

};

template<unsigned int dim>
DescentStepEuler<dim>::DescentStepEuler(const double gamma_val, const double nu_val, const unsigned int N)
: DescentStepBase<dim>(gamma_val, nu_val, N)
{}


template<unsigned int dim>
void DescentStepEuler<dim>::vectors_iteration_step()
{
  this->y_vec.add(this->step_size, this->phi1);
  this->u_vec.add(this->step_size, this->phi2);
}

/**
 * @class DescentStepAdam
 * @brief Class representing a descent step using the Adam update rule.
 *
 * @tparam dim The space dimension (2 or 3)
 *
 * This class provides functionalities for performing a descent step using the Adam method.
 */
template <unsigned int dim>
class DescentStepAdam: public DescentStepBase<dim>
{
public:
  /**
     * @brief Default constructor for DescentStepAdam.
     */
  DescentStepAdam();
  
  /**
     * @brief Constructor for DescentStepAdam. Takes in input the input parameters for the LinearSystem class
     *
     * @param gamma_val The value of the gamma parameter.
     * @param nu_val The value of the nu parameter.
     * @param N The number of grid refinements.
     */
  DescentStepAdam(const double gamma_val, const double nu_val, const unsigned int N);

private:

  /**
     * @brief Initializes the dimensions of the vectors representing the first and second moment of phi1 and phi2.
     */
  void initialize_dimension_aux_vectors();
  
  /**
     * @brief Performs a descent step for vectors y_vec and u_vec using the Adam method.
     */
  void vectors_iteration_step() override;
  
  /**
     * @brief Sets the initial vectors from which successive approximations are computed and initializes the moments vectors.
     *
     * @param y0 Initial vector for y_vec.
     * @param u0 Initial vector for u_vec.
     */
  void set_initial_vectors(const Vector<double>& y0, const Vector<double>& u0) override;

  //! Parameters beta1, beta2 and epsilon of the Adam update rule
  double                beta1=0.95;
  double                beta2=0.999;
  double                eps=1e-8;

  //! Vectors representing the first and second moment of phi1 and phi2.
  Vector<double>        m_y;
  Vector<double>        v_y;
  Vector<double>        m_u;
  Vector<double>        v_u;

};

template<unsigned int dim>
DescentStepAdam<dim>::DescentStepAdam()
{
  initialize_dimension_aux_vectors();
}

template<unsigned int dim>
DescentStepAdam<dim>::DescentStepAdam(const double gamma_val, const double nu_val, const unsigned int N)
: DescentStepBase<dim>(gamma_val, nu_val, N)
{
  initialize_dimension_aux_vectors();
}


template<unsigned int dim>
void DescentStepAdam<dim>::initialize_dimension_aux_vectors()
{
  m_y.reinit(this->dim_vec);
  v_y.reinit(this->dim_vec);
  m_u.reinit(this->dim_vec);
  v_u.reinit(this->dim_vec);
}

template <unsigned int dim>
void DescentStepAdam<dim>::set_initial_vectors(const Vector<double>& y0, const Vector<double>& u0)
{
  assert(y0.size() == this->dim_vec);
  assert(u0.size() == this->dim_vec);
  this->y_vec = y0;
  this->u_vec = u0;

  this->linear_system.update_vectors(this->y_vec,this->u_vec);
  initialize_dimension_aux_vectors();
}

template<unsigned int dim>
void DescentStepAdam<dim>::vectors_iteration_step()
{
  //! The first and second order momentum terms are updated for y with the new descent direction phi1 
  m_y *= beta1;
  m_y.add(1-beta1, this->phi1);

  Vector<double> phi1_squared(this->phi1);
  phi1_squared.scale(this->phi1);
  v_y *= beta2;
  v_y.add(1-beta2, phi1_squared);

  Vector<double> m_hat(m_y);
  Vector<double> v_hat(v_y);

  m_hat /= (1-beta1);
  v_hat /= (1-beta2);

  //! The descent direction is computed but it still requires the projection on Vg
  Vector<double> temp_y(this->dim_vec);
  for (unsigned int i=0; i<this->dim_vec; i++)
    temp_y(i) = m_hat(i)/(std::sqrt(v_hat(i))+eps);
  
  //! The first and second order momentum terms are updated for u with the new descent direction phi2
  m_u *= beta1;
  m_u.add(1-beta1, this->phi2);

  Vector<double> phi2_squared(this->phi2);
  phi2_squared.scale(this->phi2);
  v_u *= beta2;
  v_u.add(1-beta2, phi2_squared);

  Vector<double> m_hat_u(m_u);
  Vector<double> v_hat_u(v_u);

  m_hat_u /= (1-beta1);
  v_hat_u /= (1-beta2);
 
 
  //! The descent direction is computed but it still requires the projection on Vg
  Vector<double> temp_u(this->dim_vec);
  for (unsigned int i=0; i<this->dim_vec; i++)
    temp_u(i) = m_hat_u(i)/(std::sqrt(v_hat_u(i))+eps);
	
  //! The computed descent directions are projected on Vg 
  this->linear_system.projected_moments(temp_y, temp_u);

  Vector<double> proj_y(this->linear_system.get_projection_vec1());
  Vector<double> proj_u(this->linear_system.get_projection_vec2());

  //y_vec and u_vec are updated with the descent directions
  this->y_vec.add(this->step_size, proj_y);
  this->u_vec.add(this->step_size, proj_u);

}

#endif // DESCENT_STEP_METHODS_H
