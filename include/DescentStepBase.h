#include "LinearSystem.h"


using namespace dealii;


constexpr double phi1_threshold = 5*1e-5;
constexpr double phi2_threshold = 1*1e-4;


enum class GFStepType {
    EULER,
    ADAM
};

template <int dim>
class DescentStepBase
{
public:

  DescentStepBase();
  DescentStepBase(const double gamma_val, const double nu_val, const unsigned int N);

  const Vector<double>& get_y_vec() const { return y_vec; };
  const Vector<double>& get_u_vec() const { return u_vec; };
  virtual void set_initial_vectors(const Vector<double>& y0, const Vector<double>& u0);
  virtual void set_step_size(const double step_size) = 0;
  void run_gf(const unsigned int n_iter=1);
  void run();
  std::tuple<bool,unsigned int> convergence_info() const;
  double evaluate_J() const;
  double evaluate_g() const;
  void output_results_vectors() const;
  void output_iteration_results() const;

protected:

  bool converged() const;
  void initialize_dimensions();
  void descent_step();
  virtual void vectors_iteration_step() = 0;

  //void output_convergence_plots(const std::vector<double>& J_vec, const std::vector<double>& phi1_norm, const std::vector<double>& phi2_norm) const;

  LinearSystem<dim>          linear_system;

  unsigned int               dim_vec;

  Vector<double>             y_vec;
  Vector<double>             u_vec;

  Vector<double>             phi1;
  Vector<double>             phi2;
  
  unsigned int               convergence_iter;

};

template <int dim>
DescentStepBase<dim>::DescentStepBase()
:
  linear_system()
  ,dim_vec(linear_system.get_vector_size())
  ,convergence_iter(0)
{
  initialize_dimensions();
}

template <int dim>
DescentStepBase<dim>::DescentStepBase(const double gamma_val, const double nu_val, const unsigned int N)
:
  linear_system(gamma_val, nu_val, N)
  ,dim_vec(linear_system.get_vector_size())
  ,convergence_iter(0)
{
  initialize_dimensions();
}

template <int dim>
void DescentStepBase<dim>::initialize_dimensions()
{
  y_vec.reinit(dim_vec);
  u_vec.reinit(dim_vec);
  phi1.reinit(dim_vec);
  phi2.reinit(dim_vec);
}

template <int dim>
bool DescentStepBase<dim>::converged() const
{
  return (phi1.l2_norm() < phi1_threshold) && (phi2.l2_norm() < phi2_threshold);
          //&&(linear_system.evaluate_g() < g_threshold));
}

template <int dim>
std::tuple<bool, unsigned int> DescentStepBase<dim>::convergence_info() const
{
  return std::make_tuple(converged(), convergence_iter);
          //&&(linear_system.evaluate_g() < g_threshold));
}

template <int dim>
void DescentStepBase<dim>::set_initial_vectors(const Vector<double>& y0, const Vector<double>& u0)
{
  assert(y0.size() == dim_vec);
  assert(u0.size() == dim_vec);
  y_vec = y0;
  u_vec = u0;

  linear_system.update_vectors(y_vec,u_vec);
}


template <int dim>
void DescentStepBase<dim>::descent_step()
{
  linear_system.solve_system();

  phi1 = linear_system.get_phi1();
  phi2 = linear_system.get_phi2();

  vectors_iteration_step(); //updates y_vec and u_vec

  linear_system.update_vectors(y_vec,u_vec);
}


template <int dim>
void DescentStepBase<dim>::run_gf(const unsigned int n_iter)
{
  assert(n_iter > 0);

  descent_step(); // first iter
  convergence_iter = 1; // iter counter

  while (convergence_iter < n_iter && !converged())
  {
    descent_step();
    convergence_iter++;
  }
  //std::cout << "phi1: " << phi1.l2_norm() << " phi2: " << phi2.l2_norm() << std::endl;

}

template <int dim>
void DescentStepBase<dim>::run()
{
  descent_step(); // first iter
  unsigned int k = 1; // iter counter

  while (!converged())
  {
    descent_step();

    if (k % 200 == 0){
      output_iteration_results();
      //std::cout << "phi1: " << phi1.l2_norm() << " phi2: " << phi2.l2_norm() << std::endl;
    }
    k++;
  }
  std::cout << "Gradient Flow converged in k: " << k << " iterations" << std::endl;
  output_iteration_results();
  output_results_vectors();
  //std::cout << "phi1: " << phi1.l2_norm() << " phi2: " << phi2.l2_norm() << std::endl;

}

template <int dim>
double DescentStepBase<dim>::evaluate_J() const
{
  return linear_system.evaluate_J();
}

template <int dim>
double DescentStepBase<dim>::evaluate_g() const
{
  return linear_system.evaluate_g();
}

template <int dim>
void DescentStepBase<dim>::output_iteration_results() const
{
  std::cout << std::setprecision(11) << linear_system.evaluate_J() << " g: " << linear_system.evaluate_g() << std::endl;
}


template <int dim>
void DescentStepBase<dim>::output_results_vectors() const
{
  linear_system.output_result_vectors();
}

/*
void DescentStepBase::output_convergence_plots(const std::vector<double>& J_vec, const std::vector<double>& phi1_norm, const std::vector<double>& phi2_norm) const
{
  std::ofstream file;
  file.open("J_evaluation.dat");
  for (unsigned int iter=0; iter<J_vec.size(); iter++)
  {
      file << iter << " " << J_vec[iter] << std::endl;
  }
  file.close();

  file.open("phi1_norm.dat");
  for (unsigned int iter=0; iter<phi1_norm.size(); iter++)
  {
      file << iter << " " << phi1_norm[iter] << std::endl;
  }
  file.close();

  file.open("phi2_norm.dat");
  for (unsigned int iter=0; iter<phi2_norm.size(); iter++)
  {
      file << iter << " " << phi2_norm[iter] << std::endl;
  }
  file.close();

  Gnuplot gp;
  gp << "set terminal png " << std::endl;
  gp << "set grid" << std::endl;
  gp << "set output 'J_eval.png' " << std::endl;
  gp << "set ylabel 'n_iter'" << std::endl;
  gp << "set xlabel 'J'" << std::endl;
  gp << "set multiplot layout 1,2" << std::endl;
  gp << "plot 'J_evaluation.dat' with lp lw 0.5 title 'Evaluation Loss Function J'" << std::endl;
  gp << "unset multiplot" << std::endl;

}
*/
