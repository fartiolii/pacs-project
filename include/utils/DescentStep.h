#include "DescentStepBase.h"


using namespace dealii;

template<unsigned int dim>
class DescentStepEuler: public DescentStepBase<dim>
{
public:

  DescentStepEuler(){};
  DescentStepEuler(const double gamma_val, const double nu_val, const unsigned int N);

private:

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

template <unsigned int dim>
class DescentStepAdam: public DescentStepBase<dim>
{
public:

  DescentStepAdam();
  DescentStepAdam(const double gamma_val, const double nu_val, const unsigned int N);

private:

  void initialize_dimension_aux_vectors();
  void vectors_iteration_step() override;
  void set_initial_vectors(const Vector<double>& y0, const Vector<double>& u0) override;

  double                beta1=0.95;
  double                beta2=0.999;
  double                eps=1e-8;

  //Auxiliary vectors required for the Adam vector update
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


  Vector<double> temp_y(this->dim_vec);
  for (unsigned int i=0; i<this->dim_vec; i++)
    temp_y(i) = m_hat(i)/(std::sqrt(v_hat(i))+eps);

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

  Vector<double> temp_u(this->dim_vec);
  for (unsigned int i=0; i<this->dim_vec; i++)
    temp_u(i) = m_hat_u(i)/(std::sqrt(v_hat_u(i))+eps);
	
  this->linear_system.projected_moments(temp_y, temp_u);

  Vector<double> proj_y(this->linear_system.get_projection_vec1());
  Vector<double> proj_u(this->linear_system.get_projection_vec2());

  this->y_vec.add(this->step_size, proj_y);
  this->u_vec.add(this->step_size, proj_u);

}
