#include "LinearSystem.h"
//#include "gnuplot-iostream.h"

///@note: split non-template object into header and source


using namespace dealii;


 ///@note: constexpr?
constexpr double phi1_threshold = 1e-6;
constexpr double phi2_threshold = 1e-6;
//constexpr double g_threshold = 1e-3;


enum class GFStepType {
    EULER,
    ADAM,
    RMSPROP
};

template <int dim>
class GradientFlowBase
{
public:

  GradientFlowBase();

  const Vector<double>& get_y_vec() const { return y_vec; };
  const Vector<double>& get_u_vec() const { return u_vec; };
  void set_initial_vectors(const Vector<double>& y0, const Vector<double>& u0);
  virtual void set_step_size(const double step_size) = 0;
  void run(const unsigned int n_iter=1);
  bool converged() const;
  void output_results_vectors() const;
  void output_iteration_results() const;

protected:

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

};

template <int dim>
GradientFlowBase<dim>::GradientFlowBase()
:
  linear_system()
  ,dim_vec(linear_system.get_vector_size())
{
  initialize_dimensions();
}

template <int dim>
void GradientFlowBase<dim>::initialize_dimensions()
{
  y_vec.reinit(dim_vec);
  u_vec.reinit(dim_vec);
  phi1.reinit(dim_vec);
  phi2.reinit(dim_vec);
}

template <int dim>
bool GradientFlowBase<dim>::converged() const
{
  return (phi1.l2_norm() < phi1_threshold) && (phi2.l2_norm() < phi2_threshold);
          //&&(linear_system.evaluate_g() < g_threshold));
}

template <int dim>
void GradientFlowBase<dim>::set_initial_vectors(const Vector<double>& y0, const Vector<double>& u0)
{
  assert(y0.size() == dim_vec);
  assert(u0.size() == dim_vec);
  y_vec = y0;
  u_vec = u0;

  linear_system.update_vectors(y_vec,u_vec);
}


template <int dim>
void GradientFlowBase<dim>::descent_step()
{
  linear_system.solve_system();

  phi1 = linear_system.get_phi1();
  phi2 = linear_system.get_phi2();

  vectors_iteration_step(); //updates y_vec and u_vec

  linear_system.update_vectors(y_vec,u_vec);
}


template <int dim>
void GradientFlowBase<dim>::run(const unsigned int n_iter)
{
  assert(n_iter > 0);

  descent_step(); // first iter
  unsigned int k = 1; // iter counter

  while (k < n_iter && !converged())
  {
    descent_step();
    k++;
  }
  //std::cout << "phi1: " << phi1.l2_norm() << " phi2: " << phi2.l2_norm() << std::endl;

}

template <int dim>
void GradientFlowBase<dim>::output_iteration_results() const
{
  std::cout << std::setprecision(11) << linear_system.evaluate_J() << " g: " << linear_system.evaluate_g() << std::endl;
}


template <int dim>
void GradientFlowBase<dim>::output_results_vectors() const
{
  linear_system.output_result_vectors();
}

/*
void GradientFlowBase::output_convergence_plots(const std::vector<double>& J_vec, const std::vector<double>& phi1_norm, const std::vector<double>& phi2_norm) const
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
