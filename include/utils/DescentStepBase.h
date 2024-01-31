#include "LinearSystem.h"


using namespace dealii;


constexpr double phi1_threshold = 5*1e-5;
constexpr double phi2_threshold = 1*1e-4;


enum class GFStepType {
    EULER,
    ADAM
};

template <unsigned int dim>
class DescentStepBase
{
public:

  DescentStepBase();
  DescentStepBase(const double gamma_val, const double nu_val, const unsigned int N);

  virtual void set_initial_vectors(const Vector<double>& y0, const Vector<double>& u0);
  void set_step_size(const double step_sz);
  const Vector<double>& get_y_vec() const { return y_vec; };
  const Vector<double>& get_u_vec() const { return u_vec; };
  
  void run();
  void run(const unsigned int n_iter);
  
  std::tuple<bool,unsigned int> convergence_info() const;
  unsigned int get_vector_size() const;
  double evaluate_J() const;
  double evaluate_g() const;
  
  void output_iteration_results() const;
  void output_results_vectors() const;

protected:

  bool converged() const;
  void initialize_dimensions();
  void descent_step();
  virtual void vectors_iteration_step() = 0;


  LinearSystem<dim>          linear_system;

  unsigned int               dim_vec;

  Vector<double>             y_vec;
  Vector<double>             u_vec;

  Vector<double>             phi1;
  Vector<double>             phi2;
  
  double		     step_size;
  
  unsigned int               convergence_iter;

};

template <unsigned int dim>
DescentStepBase<dim>::DescentStepBase()
:
  linear_system()
  ,dim_vec(linear_system.get_vector_size())
  ,convergence_iter(0)
{
  initialize_dimensions();
}

template <unsigned int dim>
DescentStepBase<dim>::DescentStepBase(const double gamma_val, const double nu_val, const unsigned int N)
:
  linear_system(gamma_val, nu_val, N)
  ,dim_vec(linear_system.get_vector_size())
  ,convergence_iter(0)
{
  initialize_dimensions();
}

template <unsigned int dim>
void DescentStepBase<dim>::initialize_dimensions()
{
  y_vec.reinit(dim_vec);
  u_vec.reinit(dim_vec);
  phi1.reinit(dim_vec);
  phi2.reinit(dim_vec);
}

template <unsigned int dim>
void DescentStepBase<dim>::set_step_size(const double step_sz)
{
  step_size = step_sz;
}

template <unsigned int dim>
bool DescentStepBase<dim>::converged() const
{
  return (phi1.l2_norm() < phi1_threshold) && (phi2.l2_norm() < phi2_threshold);
}

template <unsigned int dim>
std::tuple<bool, unsigned int> DescentStepBase<dim>::convergence_info() const
{
  return std::make_tuple(converged(), convergence_iter);
}

template <unsigned int dim>
void DescentStepBase<dim>::set_initial_vectors(const Vector<double>& y0, const Vector<double>& u0)
{
  assert(y0.size() == dim_vec);
  assert(u0.size() == dim_vec);
  y_vec = y0;
  u_vec = u0;

  linear_system.update_vectors(y_vec,u_vec);
}


template <unsigned int dim>
void DescentStepBase<dim>::descent_step()
{
  linear_system.solve_system();

  phi1 = linear_system.get_phi1();
  phi2 = linear_system.get_phi2();

  vectors_iteration_step(); 

  linear_system.update_vectors(y_vec,u_vec);
}

template <unsigned int dim>
void DescentStepBase<dim>::run()
{
  descent_step(); 
  unsigned int convergence_iter = 1; 

  while (!converged())
  {
    descent_step();

    if (convergence_iter % 200 == 0){
      std::cout << "Iteration " << convergence_iter << std::endl;
      output_iteration_results();
    }
    convergence_iter++;
  }
  std::cout << "Gradient Flow converged in: " << convergence_iter << " iterations" << std::endl;
  output_iteration_results();
  output_results_vectors();
}


template <unsigned int dim>
void DescentStepBase<dim>::run(const unsigned int n_iter)
{
  assert(n_iter > 0);

  descent_step(); 
  convergence_iter = 1; 

  while (convergence_iter < n_iter && !converged())
  {
    descent_step();
    convergence_iter++;
  }

}


template <unsigned int dim>
double DescentStepBase<dim>::evaluate_J() const
{
  return linear_system.evaluate_J();
}

template <unsigned int dim>
double DescentStepBase<dim>::evaluate_g() const
{
  return linear_system.evaluate_g();
}

template <unsigned int dim>
unsigned int DescentStepBase<dim>::get_vector_size() const
{
  return linear_system.get_vector_size();
}

template <unsigned int dim>
void DescentStepBase<dim>::output_iteration_results() const
{
  std::cout << "J: " << std::setprecision(7) << linear_system.evaluate_J() << " Norm of g: " 
  << linear_system.evaluate_g() << std::endl;
}


template <unsigned int dim>
void DescentStepBase<dim>::output_results_vectors() const
{
  linear_system.output_result_vectors();
}

