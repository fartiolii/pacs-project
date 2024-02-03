#ifndef DESCENT_STEP_BASE_H
#define DESCENT_STEP_BASE_H


#include "LinearSystem.h"


using namespace dealii;


constexpr double phi1_threshold = 5*1e-5; //! Tolerance on the L2-norm of phi1
constexpr double phi2_threshold = 1*1e-4; //! Tolerance on the L2-norm of phi2

/**
 * @enum GFStepType
 * @brief Enumeration for update rule step types.
 */
enum class GFStepType {
    EULER,	//! Explicit Euler update rule
    ADAM	//! Adam update rule
};


/**
 * @class DescentStepBase
 * @brief Abstract base class representing different update rule types
 * 
 * @tparam dim The space dimension (2 or 3)
 *
 * This class provides the base functionalities for computing successive approximations of vectors y_vec and u_vec
 * according to different possible update rules.
 */
template <unsigned int dim>
class DescentStepBase
{
public:
  
  /**
     * @brief Default constructor for DescentStepBase.
     */
  DescentStepBase();
  
  /**
     * @brief Constructor for DescentStepBase. Takes in input the input parameters of the LinearSystem class
     *
     * @param gamma_val The value of the gamma parameter.
     * @param nu_val The value of the nu parameter.
     * @param N The number of grid refinements.
     */
  DescentStepBase(const double gamma_val, const double nu_val, const unsigned int N);

  /**
     * @brief Sets the initial vectors from which successive approximations are computed.
     *
     * @param y0 Initial vector for y_vec.
     * @param u0 Initial vector for u_vec.
     */
  virtual void set_initial_vectors(const Vector<double>& y0, const Vector<double>& u0);
  
  /**
     * @brief Sets the step size for the descent step.
     *
     * @param step_sz The step size.
     */
  void set_step_size(const double step_sz);
  
  /**
     * @brief Gets the y vector.
     *
     * @return The y vector.
     */
  const Vector<double>& get_y_vec() const { return y_vec; };
  
  /**
     * @brief Gets the u vector.
     *
     * @return The u vector.
     */
  const Vector<double>& get_u_vec() const { return u_vec; };
  
  /**
     * @brief Runs the descent step until convergence.
     */
  void run();
  
  /**
     * @brief Runs the descent step for a specified number of iterations.
     *
     * @param n_iter The number of iterations.
     */
  void run(const unsigned int n_iter);
  
  /**
     * @brief Retrieves convergence information.
     *
     * @return A tuple indicating whether the descent process has converged and the number of iterations.
     */
  std::tuple<bool,unsigned int> convergence_info() const;
  
  /**
     * @brief Gets the size of the vectors y_vec and u_vec.
     *
     * @return The size of the vectors.
     */
  unsigned int get_vector_size() const;
  
  /**
     * @brief Evaluates the cost functional J.
     *
     * @return The value of the cost functional J.
     */
  double evaluate_J() const;
  
  /**
     * @brief Outputs the cost functional J and the norm of g to the console.
     */
  void output_iteration_results() const;
  
  /**
     * @brief Outputs y_vec and u_vec to VTK files.
     */
  void output_results_vectors() const;

protected:

  /**
     * @brief Checks the convergence condition on the L2-norms of phi1 and phi2.
     *
     * @return True if converged, false otherwise.
     */
  bool converged() const;
  
  /**
     * @brief Initializes vector dimensions.
     */
  void initialize_dimensions();
  
  /**
     * @brief Performs a descent step.
     * A descent step is computed by solving the linear system with current y_vec and u_vec, retrieving
     * phi1 and phi2 and computing the new y_vec and u_vec according to the respective update rules
     */
  void descent_step();
  
  /**
     * @brief Computes the new y_vec and u_vec, given phi1 and phi2, according to the update rule.
     */
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

#endif // DESCENT_STEP_BASE_H
