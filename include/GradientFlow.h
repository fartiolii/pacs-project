#include "GradientFlowBase.h"

///@note: split non-template object into header and source

using namespace dealii;

template<int dim>
class GradientFlowEuler: public GradientFlowBase<dim>
{
public:

  GradientFlowEuler(){};
  void set_step_size(const double step_size) override;

private:

  void vectors_iteration_step() override;

  double                step_size=1;

};

template<int dim>
void GradientFlowEuler<dim>::set_step_size(const double sz)
{
  step_size = sz;
}

template<int dim>
void GradientFlowEuler<dim>::vectors_iteration_step()
{
  this->y_vec.add(step_size, this->phi1);
  this->u_vec.add(step_size, this->phi2);
}

template <int dim>
class GradientFlowAdam: public GradientFlowBase<dim>
{
public:

  GradientFlowAdam();
  void set_step_size(const double step_size) override;

private:

  void initialize_dimension_aux_vectors();
  void vectors_iteration_step() override;

  double                beta1=0.98;
  double                beta2=0.999;
  double                eps=1e-8;
  double                alpha=0.001;

  //Auxiliary vectors required for the Adam vector update
  Vector<double>        m_y;
  Vector<double>        v_y;

  Vector<double>        m_u;
  Vector<double>        v_u;

};

template<int dim>
GradientFlowAdam<dim>::GradientFlowAdam()
{
  initialize_dimension_aux_vectors();
}

template<int dim>
void GradientFlowAdam<dim>::set_step_size(const double step_size)
{
  alpha = step_size;
}


template<int dim>
void GradientFlowAdam<dim>::initialize_dimension_aux_vectors()
{
  m_y.reinit(this->dim_vec);
  v_y.reinit(this->dim_vec);
  m_u.reinit(this->dim_vec);
  v_u.reinit(this->dim_vec);
}

template<int dim>
void GradientFlowAdam<dim>::vectors_iteration_step()
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


  Vector<double> temp(this->dim_vec);
  for (unsigned int i=0; i<this->dim_vec; i++)
    ///@note: math functions should always have explicit namespace
    temp(i) = m_hat(i)/(std::sqrt(v_hat(i))+eps);


  this->y_vec.add(alpha, temp);

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

  this->u_vec.add(alpha, temp_u);

}

template<int dim>
class GradientFlowRMSProp: public GradientFlowBase<dim>
{
public:

  GradientFlowRMSProp();
  void set_step_size(const double step_size) override;

private:

  void initialize_dimension_aux_vectors();
  void vectors_iteration_step() override;

  double                beta=0.9;
  double                eps=1e-8;
  double                alpha=1e-3;

  //Auxiliary vectors required for the RMSProp vector update
  Vector<double>        v_y;
  Vector<double>        v_u;

};

template<int dim>
GradientFlowRMSProp<dim>::GradientFlowRMSProp()
{
  initialize_dimension_aux_vectors();
}

template<int dim>
void GradientFlowRMSProp<dim>::set_step_size(const double step_size)
{
  alpha = step_size;
}


template<int dim>
void GradientFlowRMSProp<dim>::initialize_dimension_aux_vectors()
{
  v_y.reinit(this->dim_vec);
  v_u.reinit(this->dim_vec);
}

template<int dim>
void GradientFlowRMSProp<dim>::vectors_iteration_step()
{
  // Update y
  Vector<double> phi1_squared(this->phi1);
  phi1_squared.scale(this->phi1);
  v_y *= beta;
  v_y.add(1-beta, phi1_squared);

  Vector<double> temp_y(this->dim_vec);
  for (unsigned int i=0; i<this->dim_vec; i++)
    temp_y(i) = this->phi1(i)/(std::sqrt(v_y(i))+eps);

  this->y_vec.add(alpha, temp_y);

  // Update u
  Vector<double> phi2_squared(this->phi2);
  phi2_squared.scale(this->phi2);
  v_u *= beta;
  v_u.add(1-beta, phi2_squared);

  Vector<double> temp_u(this->dim_vec);
  for (unsigned int i=0; i<this->dim_vec; i++)
    temp_u(i) = this->phi2(i)/(std::sqrt(v_u(i))+eps);

  this->u_vec.add(alpha, temp_u);

}
