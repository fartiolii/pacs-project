#include "GradientFlowBase.h"

using namespace dealii;

class GradientFlowEuler: public GradientFlowBase
{
public:

  GradientFlowEuler();
  void set_step_size(const double step_size) override;

private:

  void vectors_iteration_step() override;

  double                step_size=1;

};

GradientFlowEuler::GradientFlowEuler()
{}

void GradientFlowEuler::set_step_size(const double sz)
{
  step_size = sz;
}

void GradientFlowEuler::vectors_iteration_step()
{
  y_vec.add(step_size, phi1);
  u_vec.add(step_size, phi2);
}

class GradientFlowAdam: public GradientFlowBase
{
public:

  GradientFlowAdam();
  void set_step_size(const double step_size) override;

private:

  void initialize_dimension_aux_vectors();
  void vectors_iteration_step() override;

  double                beta1=0.9;
  double                beta2=0.999;
  double                eps=1e-8;
  double                alpha=0.001;

  //Auxiliary vectors required for the Adam vector update
  Vector<double>        m_y;
  Vector<double>        v_y;

  Vector<double>        m_u;
  Vector<double>        v_u;

  Vector<double>        y_last;
  Vector<double>        u_last;

};

GradientFlowAdam::GradientFlowAdam()
{
  initialize_dimension_aux_vectors();
}

void GradientFlowAdam::set_step_size(const double step_size)
{
  alpha = step_size;
}


void GradientFlowAdam::initialize_dimension_aux_vectors()
{
  m_y.reinit(dim_vec);
  v_y.reinit(dim_vec);
  m_u.reinit(dim_vec);
  v_u.reinit(dim_vec);
  y_last.reinit(dim_vec);
  u_last.reinit(dim_vec);
}

void GradientFlowAdam::vectors_iteration_step()
{

  m_y *= beta1;
  m_y.add(1-beta1, phi1);

  Vector<double> phi1_squared(phi1);
  phi1_squared.scale(phi1);
  v_y *= beta2;
  v_y.add(1-beta2, phi1_squared);

  Vector<double> m_hat(m_y);
  Vector<double> v_hat(v_y);

  m_hat /= (1-beta1);
  v_hat /= (1-beta2);

  Vector<double> temp(dim_vec);
  for (unsigned int i=0; i<dim_vec; i++)
    temp(i) = m_hat(i)/(sqrt(v_hat(i))+eps);

  y_vec.add(alpha, temp);

  m_u *= beta1;
  m_u.add(1-beta1, phi2);

  Vector<double> phi2_squared(phi2);
  phi2_squared.scale(phi2);
  v_u *= beta2;
  v_u.add(1-beta2, phi2_squared);

  Vector<double> m_hat_u(m_u);
  Vector<double> v_hat_u(v_u);

  m_hat_u /= (1-beta1);
  v_hat_u /= (1-beta2);

  Vector<double> temp_u(dim_vec);
  for (unsigned int i=0; i<dim_vec; i++)
    temp_u(i) = m_hat_u(i)/(sqrt(v_hat_u(i))+eps);

  u_vec.add(alpha, temp_u);

}
