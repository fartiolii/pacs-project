#ifndef LINEAR_SYSTEM_H
#define LINEAR_SYSTEM_H

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>


#include <fstream>
#include <iostream>
#include <cassert>
#include <cmath>
#include <memory>


using namespace dealii;

/**
 * @class LinearSystem
 * @brief Class representing the linear system to solve to compute the descent directions for the Explicit Euler and Adam update rules
 *
 * @tparam dim The space dimension (2 or 3)
 *
 * This class encapsulates the functionalities to assemble and solve the linear system,
 * given the node coefficients y_vec and u_vec of the vectors y and u.
 *
 * The linear system is defined as:
 *    A_matrix * x = rhs_vec
 * where A_matrix is a sparse matrix, x=[phi1_vec, phi2_vec, lambda]^T is the solution vector, 
 * and rhs=[-grad_Jy, -grad_Ju, -g_vec]^T is the right-hand side vector.
 * 
 */

template<unsigned int dim>
class LinearSystem
{
public:

  /**
     * @brief Constructor for the LinearSystem class.
     *
     * @param gamma_val The value of gamma parameter for the discrete optimal control problem
     * @param nu_val The value of nu parameter for the discrete optimal control problem
     * @param N The number of grid refinements of the mesh
     */
  LinearSystem(const double gamma_val, const double nu_val, const unsigned int N);

  /**
     * @brief Solves the linear system, given y_vec and u_vec.
     */
  void solve_system();
  
  /**
     * @brief Computes the projections on the manifold V_g of the descent directions of y and u 
     *
     * @param moment_y Vector representing the descent direction of y.
     * @param moment_u Vector representing the descent direction of u.
     * 
     * Method used by the Adam update rule to project on V_g the descent directions computed with the first and 
     * second moments. It computes the projections by solving the linear system with the same A_matrix and by replacing 
     * in rhs_vec the gradients of J with the descent directions.
     *
     */
  void projected_moments(const Vector<double>& moment_y, const Vector<double>& moment_u);
  
  /**
     * @brief Updates the node coefficient vectors y_vec and u_vec
     *
     * @param y Vector representing y_vec.
     * @param u Vector representing u_vec.
     */
  void update_vectors(const Vector<double> &y, const Vector<double> &u);
  
  /**
     * @brief Gets the size of the node coefficients vector (size of y_vec and u_vec).
     *
     * @return The size of the node coefficients vector.
     */
  unsigned int get_vector_size() const { return vector_size; };
  
  /**
     * @brief Gets the phi1 vector which is the descent direction for y obtained solving the linear system.
     *
     * @return The phi1 vector.
     */
  const Vector<double>& get_phi1() const { return phi1_vec; };
  
  /**
     * @brief Gets the phi2 vector which is the descent direction for u obtained solving the linear system.
     *
     * @return The phi2 vector.
     */
  const Vector<double>& get_phi2() const { return phi2_vec; };
  
  /**
     * @brief Gets the proj_vec1 vector which is the projected descent direction for y obtained with the projected_moments method.
     *
     * @return The proj_vec1 vector.
     */
  const Vector<double>& get_projection_vec1() const { return proj_vec1; };
  
  /**
     * @brief Gets the proj_vec2 vector which is the projected descent direction for u obtained with the projected_moments method.
     *
     * @return The proj_vec2 vector.
     */
  const Vector<double>& get_projection_vec2() const { return proj_vec2; };
  
  /**
     * @brief Evaluates the objective function J given y_vec and u_vec.
     *
     * @return The value of the objective function J.
     */
  double evaluate_J() const;
  
  /**
     * @brief Evaluates the L2-norm of the constraint g, given y_vec and u_vec.
     *
     * @return The L2-norm of g.
     */
  double evaluate_g() const;
  
  /**
     * @brief Outputs the y_vec and u_vec vectors to VTK files.
     */
  void output_result_vectors() const;


private:

  /**
     * @brief Creates the mesh and refines it N times
     */
  void make_grid();
  
  /**
     * @brief Initializes all the vectors with the dimension given by the number of nodes in the mesh
     */
  void initialize_dimensions();
  
  /**
     * @brief Distributes the degrees of freedom of the mesh and creates the sparsity pattern of the mass and stiffness matrices
     */
  void setup_system();
  
  /**
     * @brief Assembles the mass and stiffness matrices
     */
  void assemble_matrices();
  
  /**
     * @brief Computes the sparsity pattern for matrix A_matrix, based on the sparsity pattern of the stiffness and mass matrices
     */
  void set_global_sparsity_pattern();
  
  /**
     * @brief Makes the grid, assembles mass and stiffness matrices and computes the global sparsity pattern for A_matrix
     */
  void setup_linear_system();
  

  /**
     * @brief Computes the values of phi_vec and Jacobian_phi given y_vec 
     */
  void set_phi();
  
  /**
     * @brief Assembles KTilde given its sparsity pattern and the values in phi_vec, Jacobian_phi, stiffness_matrix and mass_matrix
     */
  void assemble_KTilde();
  
  /**
     * @brief Assembles A_matrix given its sparsity pattern and the values in the matrices K_Tilde and mass_matrix
     */
  void assemble_A();
  
  /**
     * @brief Assembles the rhs_vec vector by computing the gradients of J and evaluating g, with the current y_vec and u_vec
     */
  void assemble_rhs();

  /**
     * @brief Solves the linear system, once A_matrix and rhs_vec have been assembled.
     */
  void solve();
  
  //! Attributes describing the triangulation of the mesh and the global numbering of the degrees of freedom
  Triangulation<dim> 	   triangulation;  
  const FE_Q<dim>          fe;
  const FESystem<dim>      fe_system;
  DoFHandler<dim>          dof_handler;
  DoFHandler<dim>          dof_handler_system;
  
  //! Quadrature formula used for the integrals required to assemble mass and stiffness matrices
  const QGaussSimplex<dim> quadrature_formula;

  //! Number of node coefficients of y_vec and u_vec
  unsigned int          vector_size;

  //! Number of times to refine the grid
  unsigned int          grid_refinement;
  
  //! Parameters of the discrete optimal control problem
  double                nu;
  double                gamma;
  double                yd=1;
  
  SparsityPattern       sparsity_pattern;
  SparseMatrix<double>  stiffness_matrix;
  SparseMatrix<double>  mass_matrix;
  SparseMatrix<double>  K_tilde;

  SparsityPattern       global_sparsity_pattern;
  SparseMatrix<double>  A_matrix;

  Vector<double> y_vec;
  Vector<double> u_vec;

  Vector<double> yd_vec;
  
  //! Vectors containing the values of phi(y) and its Jacobian matrix (which is diagonal)
  Vector<double> phi_vec;
  Vector<double> Jacobian_phi;
  
  Vector<double> rhs_vec;

  //! Vector in which the solution of the linear system is stored
  Vector<double> solution_vec;  
  Vector<double> phi1_vec;
  Vector<double> phi2_vec;
  
  //! Vectors containing the projections of the descent directions for y and u (Adam update rule)
  Vector<double> proj_vec1;
  Vector<double> proj_vec2;

};




template<unsigned int dim>
LinearSystem<dim>::LinearSystem(const double gamma_val, const double nu_val, const unsigned int N)
: 
    fe(1)
  , fe_system(fe, 3)
  , dof_handler(triangulation)
  , dof_handler_system(triangulation)
  , quadrature_formula(fe.degree+1)
  , vector_size(0)
  , grid_refinement(N)
  , nu(nu_val)
  , gamma(gamma_val)
{
  setup_linear_system();
  DoFRenumbering::component_wise(dof_handler_system, std::vector<unsigned int>{0,1,2});
}





template<unsigned int dim>
void LinearSystem<dim>::make_grid()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(grid_refinement);  
}


template<unsigned int dim>
void LinearSystem<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  dof_handler_system.distribute_dofs(fe_system);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  stiffness_matrix.reinit(sparsity_pattern);
  mass_matrix.reinit(sparsity_pattern);

  vector_size = dof_handler.n_dofs();

  initialize_dimensions();

  yd_vec.add(yd);
}

template<unsigned int dim>
void LinearSystem<dim>::initialize_dimensions()
{
  assert(vector_size != 0);
  y_vec.reinit(vector_size);
  u_vec.reinit(vector_size);
  phi_vec.reinit(vector_size);
  Jacobian_phi.reinit(vector_size);
  yd_vec.reinit(vector_size);
  proj_vec1.reinit(vector_size);
  proj_vec2.reinit(vector_size);

  rhs_vec.reinit(dof_handler_system.n_dofs());        
  solution_vec.reinit(dof_handler_system.n_dofs());

  phi1_vec.reinit(vector_size);
  phi2_vec.reinit(vector_size);
  
}

template<unsigned int dim>
void LinearSystem<dim>::assemble_matrices()
{

  QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix_stiffness(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_matrix_mass(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      cell_matrix_stiffness = 0;
      cell_matrix_mass = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {

          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
            {
              cell_matrix_stiffness(i, j) +=
                (fe_values.shape_grad(i, q_index) * //! grad v_i(x_q)
                 fe_values.shape_grad(j, q_index) * //! grad v_j(x_q)
                 fe_values.JxW(q_index));           //! dx

              cell_matrix_mass(i, j) +=
                (fe_values.shape_value(i, q_index) * //! v_i(x_q)
                 fe_values.shape_value(j, q_index) * //! v_j(x_q)
                 fe_values.JxW(q_index));           //! dx
            }

        }
      cell->get_dof_indices(local_dof_indices);

      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
        {
          stiffness_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix_stiffness(i, j));
          mass_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix_mass(i, j));
        }

    }

  //! Setting boundary values on mass and stiffness matrices 
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           boundary_values);

  Vector<double> temp_rhs(vector_size);
  Vector<double> temp_sol(vector_size);
  temp_rhs.add(1);
  MatrixTools::apply_boundary_values(boundary_values,
                                     mass_matrix,
                                     temp_sol,
                                     temp_rhs);
  MatrixTools::apply_boundary_values(boundary_values,
                                     stiffness_matrix,
                                     temp_sol,
                                     temp_rhs);

}



template<unsigned int dim>
void LinearSystem<dim>::setup_linear_system()
{
  make_grid();
  setup_system();
  assemble_matrices();
  set_global_sparsity_pattern();
}


template<unsigned int dim>
void LinearSystem<dim>::set_phi()
{
  auto phi = [] (double y) { return std::exp(y);};
  auto grad_phi = [] (double y) { return std::exp(y);};

  for (unsigned int i=0; i != vector_size; i++)
  {
      phi_vec(i) = phi(y_vec(i));
      Jacobian_phi(i) = grad_phi(y_vec(i));
  }

  //! Set boundary values on phi and its Jacobian matrix
  IndexSet boun_val = DoFTools::extract_boundary_dofs(dof_handler);

  for (const auto &bv: boun_val)
  {
    phi_vec(bv) = 0;
    Jacobian_phi(bv) = 0;
  }


}

template<unsigned int dim>
void LinearSystem<dim>::assemble_KTilde()
{
      set_phi();

      K_tilde.reinit(sparsity_pattern);
      
      for (unsigned int row=0; row!=mass_matrix.m(); row++)
      {
          auto row_it=mass_matrix.begin(row);
          while(row_it!=mass_matrix.end(row))
          {
              K_tilde.set(row, row_it->column(), (row_it->value())*Jacobian_phi(row_it->column()));
              row_it++;
          }
      }

      K_tilde *= gamma;

      for (unsigned int row=0; row!=stiffness_matrix.m(); row++)
      {
          auto row_it=stiffness_matrix.begin(row);
          while(row_it!=stiffness_matrix.end(row))
          {
              K_tilde.add(row, row_it->column(), row_it->value());
              row_it++;
          }
      }
}

template<unsigned int dim>
void LinearSystem<dim>::set_global_sparsity_pattern()
{
    DynamicSparsityPattern global_sp(dof_handler_system.n_dofs());

    //! adding indexes for diagonal values in the upper identity matrix
    unsigned int n_diag_vals = 2*vector_size;
    for (unsigned int i=0; i<n_diag_vals; i++)
        global_sp.add(i,i);

    //! adding indexes for lower block Kstar and -M (sparse matrices)
    unsigned int disp_row = 2*vector_size; //! row index displacement (for both matrices)
    unsigned int disp_col = vector_size; //! column index displacement (for -M)
    for (SparsityPattern::const_iterator it=sparsity_pattern.begin(); it!=sparsity_pattern.end(); it++)
    {
        global_sp.add(it->row()+disp_row,it->column()); //! Ktilde block
        global_sp.add(it->row()+disp_row,it->column()+disp_col); //! -M block
    }

    //! leverage symmetry for the upper triangular blocks indexes
    global_sp.symmetrize();

    global_sparsity_pattern.copy_from(global_sp);

}

template<unsigned int dim>
void LinearSystem<dim>::assemble_A()
{

  A_matrix.reinit(global_sparsity_pattern);

  unsigned int disp_row(2*vector_size);
  unsigned int disp_col(vector_size);

  for (unsigned int row=0; row!=mass_matrix.m(); row++)
  {
      auto col_it=mass_matrix.begin(row);
      while(col_it != mass_matrix.end(row))
      {
          A_matrix.set(disp_row+col_it->row(), disp_col+col_it->column(), -(col_it->value()));
          col_it++;
      }
  }

  //! Adding K_tilde matrix in lower left corner
  for (unsigned int row=0; row!=K_tilde.m(); row++)
  {
      auto col_it=K_tilde.begin(row);
      while(col_it != K_tilde.end(row))
      {
          A_matrix.set(disp_row+col_it->row(), col_it->column(), col_it->value());
          col_it++;
      }
  }

  A_matrix *= 2;
  A_matrix.symmetrize();

  //! Adding ones on the upper left diagonal
  for (unsigned int i=0; i < 2*vector_size; i++)
  {
      A_matrix.set(i,i,1);
  }
}

template<unsigned int dim>
void LinearSystem<dim>::assemble_rhs()
{
  Vector<double> grad_Jy(vector_size);
  Vector<double> grad_Ju(vector_size);
  Vector<double> g(vector_size);

  Vector<double> diff_yvec(y_vec);
  diff_yvec -= yd_vec;

  mass_matrix.vmult(grad_Jy, diff_yvec);

  mass_matrix.vmult(grad_Ju, u_vec);
  grad_Ju *= nu;

  grad_Jy *= -1;
  grad_Ju *= -1;

  std::vector<unsigned int> indices(vector_size);
  std::iota(indices.begin(), indices.end(), 0);

  grad_Jy.extract_subvector_to(indices.begin(), indices.end(), rhs_vec.begin());


  auto it = rhs_vec.begin();
  it += vector_size;
  grad_Ju.extract_subvector_to(indices.begin(), indices.end(), it);


  Vector<double> temp(vector_size);

  mass_matrix.vmult(temp, phi_vec);
  g.add(gamma, temp);
  stiffness_matrix.vmult(temp, y_vec);
  g += temp;
  mass_matrix.vmult(temp, u_vec);
  g -= temp;

  g *= -1;

  it += vector_size;
  g.extract_subvector_to(indices.begin(), indices.end(), it);
}


template<unsigned int dim>
void LinearSystem<dim>::solve()
{
  //! Applying boundary values to phi1 and phi2
  FEValuesExtractors::Vector phi(0);
  ComponentMask phi_mask = fe_system.component_mask(phi);

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler_system,
                                           0,
                                           Functions::ZeroFunction<dim>(3),
                                           boundary_values,
                                           phi_mask);

  MatrixTools::apply_boundary_values(boundary_values,
                                     A_matrix,
                                     solution_vec,
                                     rhs_vec);


  SolverControl                       solver_control(200000, 1e-6 * rhs_vec.l2_norm());
  SolverMinRes<Vector<double>>        solver(solver_control);
  solver.solve(A_matrix, solution_vec, rhs_vec, PreconditionIdentity());


  std::vector<unsigned int> indices(vector_size);
  std::iota(indices.begin(), indices.end(), 0);
  solution_vec.extract_subvector_to(indices.begin(), indices.end(), phi1_vec.begin());
  std::iota(indices.begin(), indices.end(), vector_size);
  solution_vec.extract_subvector_to(indices.begin(), indices.end(), phi2_vec.begin());

}

template<unsigned int dim>
void LinearSystem<dim>::output_result_vectors() const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(y_vec, "y_vec");
  data_out.build_patches();

  std::ofstream output("y_vec.vtk");
  data_out.write_vtk(output);

  DataOut<dim> data_out_u;

  data_out_u.attach_dof_handler(dof_handler);
  data_out_u.add_data_vector(u_vec, "u_vec");
  data_out_u.build_patches();

  std::ofstream output_u("u_vec.vtk");
  data_out_u.write_vtk(output_u);
}

template<unsigned int dim>
void LinearSystem<dim>::update_vectors(const Vector<double> &y, const Vector<double> &u)
{
  assert(y.size() == vector_size);
  assert(u.size() == vector_size);
  y_vec=y;
  u_vec=u;
}

template<unsigned int dim>
void LinearSystem<dim>::solve_system()
{
  assemble_KTilde();
  assemble_A();
  assemble_rhs();
  solve();
}

template<unsigned int dim>
double LinearSystem<dim>::evaluate_J() const
{
  double J;

  Vector<double> diff_yvec(y_vec);
  diff_yvec -= yd_vec;
  Vector<double> temp(vector_size);
  mass_matrix.vmult(temp, diff_yvec);

  J = 0.5*(temp*diff_yvec);

  mass_matrix.vmult(temp, u_vec);

  J = J + 0.5*nu*(temp*u_vec);

  return J;
}

template<unsigned int dim>
double LinearSystem<dim>::evaluate_g() const
{
  Vector<double> g(vector_size);
  Vector<double> temp(vector_size);

  mass_matrix.vmult(temp, phi_vec);
  g.add(gamma, temp);
  stiffness_matrix.vmult(temp, y_vec);
  g += temp;
  mass_matrix.vmult(temp, u_vec);
  g -= temp;

  return g.l2_norm();
}

template<unsigned int dim>
void LinearSystem<dim>::projected_moments(const Vector<double>& moment_y, const Vector<double>& moment_u)
{
  //! Creating a temporary rhs vector which substitutes the gradients of J with the descent directions in input
  Vector<double> temp_rhs(rhs_vec);

  std::vector<unsigned int> indices(vector_size);
  std::iota(indices.begin(), indices.end(), 0);

  auto it = temp_rhs.begin();
  moment_y.extract_subvector_to(indices.begin(), indices.end(), it);

  it += vector_size;
  moment_u.extract_subvector_to(indices.begin(), indices.end(), it);


  Vector<double> temp_solution(dof_handler_system.n_dofs());

  //! Applying boundary values to the system
  FEValuesExtractors::Vector phi(0);
  ComponentMask phi_mask = fe_system.component_mask(phi);

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler_system,
                                           0,
                                           Functions::ZeroFunction<dim>(3),
                                           boundary_values,
                                           phi_mask);

  MatrixTools::apply_boundary_values(boundary_values,
                                     A_matrix,
                                     temp_solution,
                                     temp_rhs);

  //! Solve the system and extract the two projections
  SolverControl                       solver_control(200000, 1e-6 * temp_rhs.l2_norm());
  SolverMinRes<Vector<double>>        solver(solver_control);
  solver.solve(A_matrix, temp_solution, temp_rhs, PreconditionIdentity());

  std::vector<unsigned int> indexes(vector_size);
  std::iota(indexes.begin(), indexes.end(), 0);
  temp_solution.extract_subvector_to(indexes.begin(), indexes.end(), proj_vec1.begin());
  std::iota(indexes.begin(), indexes.end(), vector_size);
  temp_solution.extract_subvector_to(indexes.begin(), indexes.end(), proj_vec2.begin());
}

#endif // LINEAR_SYSTEM_H
