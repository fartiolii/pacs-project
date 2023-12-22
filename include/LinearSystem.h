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

///@note: split non-template object into header and source



using namespace dealii;

///@note: is it worth it to make it template?
template<int dim>
class LinearSystem
{
public:
  LinearSystem();

  void update_vectors(const Vector<double> &y, const Vector<double> &u);
  void solve_system();
  void output_result_vectors() const;
  const Vector<double>& get_phi1() const { return phi1_vec; };
  const Vector<double>& get_phi2() const { return phi2_vec; };
  double evaluate_J() const;
  double evaluate_g() const;
  void test_Kstar();

  unsigned int get_vector_size() const { return prob_size; };




private:
  void make_grid();
  void initialize_dimensions();
  void setup_system();
  void assemble_system();
  void initialize_vectors_y_u();

  void set_phi();
  void assemble_Kstar();
  void set_global_sparsity_pattern();
  void assemble_A();
  void assemble_rhs();

  void setup_linear_system();

  void solve();
  void output_results() const;



  Triangulation<dim> triangulation;

  const MappingFE<2>     mapping;
  const FE_SimplexP<2>   fe;

  const FESystem<2>      fe_system;

  const QGaussSimplex<2> quadrature_formula;

  DoFHandler<2>          dof_handler;
  DoFHandler<2>          dof_handler_system;

  unsigned int          prob_size;
  double                alpha=0.1;
  double                gamma=0.5;
  double                yd=1;
  SparsityPattern       sparsity_pattern;
  SparseMatrix<double>  stiffness_matrix;
  SparseMatrix<double>  mass_matrix;
  SparseMatrix<double>  K_star;

  SparsityPattern       global_sparsity_pattern;
  SparseMatrix<double>  A_matrix;

  Vector<double> y_vec;
  Vector<double> u_vec;

  Vector<double> yd_vec;
  Vector<double> phi_vec;
  Vector<double> Jacobian_phi;
  Vector<double> rhs_vec;
  Vector<double> phi1_vec;
  Vector<double> phi2_vec;

  Vector<double> solution_vec;

};


LinearSystem::LinearSystem()
  :
    mapping(FE_SimplexP<2>(1))
  , fe(1)
  , fe_system(fe, 3)
  , quadrature_formula(fe.degree+1)
  , dof_handler(triangulation)
  , dof_handler_system(triangulation)
  , prob_size(0)
{
  setup_linear_system();
  DoFRenumbering::component_wise(dof_handler_system, std::vector<unsigned int>{0,1,2});
}



void LinearSystem::make_grid()
{
  triangulation.clear();
  GridIn<2>(triangulation).read("tri.msh");
  triangulation.refine_global(5);


  std::ofstream out("grid-LinSys.svg");
  GridOut       grid_out;
  grid_out.write_svg(triangulation, out);
}



void LinearSystem::setup_system()
{
  dof_handler.distribute_dofs(fe);
  dof_handler_system.distribute_dofs(fe_system);
  ///@note: do not leave commented or unused code (in the final version, for now is fine)
  /*
  std::cout << "Number of degrees of freedom vector: " << dof_handler.n_dofs()
            << std::endl;
  std::cout << "Number of degrees of freedom system: " << dof_handler_system.n_dofs()
            << std::endl;
  */

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);


  stiffness_matrix.reinit(sparsity_pattern);
  mass_matrix.reinit(sparsity_pattern);

  prob_size = dof_handler.n_dofs();

  initialize_dimensions();

  yd_vec.add(yd);
}

void LinearSystem::initialize_dimensions()
{
  assert(prob_size != 0);
  y_vec.reinit(prob_size);
  u_vec.reinit(prob_size);
  phi_vec.reinit(prob_size);
  Jacobian_phi.reinit(prob_size);
  yd_vec.reinit(prob_size);


  rhs_vec.reinit(dof_handler_system.n_dofs());        //3*prob_size
  solution_vec.reinit(dof_handler_system.n_dofs());

  phi1_vec.reinit(prob_size);
  phi2_vec.reinit(prob_size);
}

void LinearSystem::assemble_system()
{

  QGauss<2> quadrature_formula(fe.degree + 1);

  FEValues<2> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix_stiffness(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_matrix_mass(dofs_per_cell, dofs_per_cell);
  //Vector<double>     cell_y_vec(dofs_per_cell);
  //Vector<double>     cell_u_vec(dofs_per_cell);

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
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

              cell_matrix_mass(i, j) +=
                (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                 fe_values.shape_value(j, q_index) * // phi_j(x_q)
                 fe_values.JxW(q_index));           // dx
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

  // Setting boundary values on mass and stiffness matrices
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<2>(),
                                           boundary_values);

  Vector<double> temp_rhs(prob_size);
  Vector<double> temp_sol(prob_size);
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


void LinearSystem::initialize_vectors_y_u()
{
  y_vec.add(0.01);
  u_vec.add(0.01);

  // Setting boundary values on solution vectors
  IndexSet boundary_values = DoFTools::extract_boundary_dofs(dof_handler);

  for (const auto &bv: boundary_values)
  {
      y_vec(bv) = 0.0;
      u_vec(bv) = 0.0;
  }

}


void LinearSystem::setup_linear_system()
{
  make_grid();
  setup_system();
  assemble_system();
  initialize_vectors_y_u();
  set_global_sparsity_pattern();
  std::cout << "Linear system successfully set up with " <<  dof_handler.n_dofs() << " dofs per vector" << std::endl;
}


void LinearSystem::set_phi()
{
  //auto phi = [] (double y) { return exp(y);};
  //auto grad_phi = [] (double y) { return exp(y);};
  auto phi = [] (double y) { return exp(y)*y;};
  auto grad_phi = [] (double y) { return (1+y)*exp(y);};
  //auto phi = [] (double y) { return std::pow(y,3);};
  //auto grad_phi = [] (double y) { return 3*(y*y);};

  for (unsigned int i=0; i != prob_size; i++)
  {
      phi_vec(i) = phi(y_vec(i));
      Jacobian_phi(i) = grad_phi(y_vec(i));
  }


  // Set boundary values on phi and its Jacobian
  IndexSet boun_val = DoFTools::extract_boundary_dofs(dof_handler);

  for (const auto &bv: boun_val)
  {
    phi_vec(bv) = 0;
    Jacobian_phi(bv) = 0;
  }


}

void LinearSystem::assemble_Kstar()
{
      set_phi();

      // Matrix multiplication: gamma*M*Jacobian_phi (product between a SparseMatrix and a Vector)
      K_star.reinit(sparsity_pattern);

      for (unsigned int row=0; row!=mass_matrix.m(); row++)
      {
          auto row_it=mass_matrix.begin(row);
          while(row_it!=mass_matrix.end(row))
          {
              K_star.set(row, row_it->column(), (row_it->value())*Jacobian_phi(row_it->column()));
              row_it++;
          }
      }

      K_star *= gamma;

      for (unsigned int row=0; row!=stiffness_matrix.m(); row++)
      {
          auto row_it=stiffness_matrix.begin(row);
          while(row_it!=stiffness_matrix.end(row))
          {
              K_star.add(row, row_it->column(), row_it->value());
              row_it++;
          }
      }
}

void LinearSystem::set_global_sparsity_pattern()
{
    DynamicSparsityPattern global_sp(dof_handler_system.n_dofs());

    // adding indexes for diagonal values in the upper identity matrix
    unsigned int n_diag_vals = 2*prob_size;
    for (unsigned int i=0; i<n_diag_vals; i++)
        global_sp.add(i,i);

    // adding indexes for lower block Kstar and -M (sparse matrices)
    unsigned int disp_row = 2*prob_size; // row index displacement (for both matrices)
    unsigned int disp_col = prob_size; // column index displacement (for -M)
    for (SparsityPattern::const_iterator it=sparsity_pattern.begin(); it!=sparsity_pattern.end(); it++)
    {
        global_sp.add(it->row()+disp_row,it->column()); // Kstar block
        global_sp.add(it->row()+disp_row,it->column()+disp_col); // -M block
    }


    // leverage symmetry for the upper triangular blocks indexes
    global_sp.symmetrize();

    global_sparsity_pattern.copy_from(global_sp);

}

void LinearSystem::assemble_A()
{

  A_matrix.reinit(global_sparsity_pattern);


  unsigned int disp_row(2*prob_size);
  unsigned int disp_col(prob_size);

  for (unsigned int row=0; row!=mass_matrix.m(); row++)
  {
      auto col_it=mass_matrix.begin(row);
      while(col_it != mass_matrix.end(row))
      {
          A_matrix.set(disp_row+col_it->row(), disp_col+col_it->column(), -(col_it->value()));
          col_it++;
      }
  }

  // Adding K_star matrix in lower left corner
  for (unsigned int row=0; row!=K_star.m(); row++)
  {
      auto col_it=K_star.begin(row);
      while(col_it != K_star.end(row))
      {
          A_matrix.set(disp_row+col_it->row(), col_it->column(), col_it->value());
          col_it++;
      }
  }

  A_matrix *= 2;
  A_matrix.symmetrize();

  // Adding ones on the upper left diagonal
  for (unsigned int i=0; i < 2*prob_size; i++)
  {
      A_matrix.set(i,i,1);
  }
}

void LinearSystem::assemble_rhs()
{
  Vector<double> grad_Jy(prob_size);
  Vector<double> grad_Ju(prob_size);
  Vector<double> g(prob_size);

  Vector<double> diff_yvec(y_vec);
  diff_yvec -= yd_vec;

  mass_matrix.vmult(grad_Jy, diff_yvec);

  mass_matrix.vmult(grad_Ju, u_vec);

  grad_Ju *= alpha;

  grad_Jy *= -1;
  grad_Ju *= -1;


  // Indices of the vectors to be copied into rhs
  std::vector<unsigned int> indices(prob_size);
  std::iota(indices.begin(), indices.end(), 0);

  grad_Jy.extract_subvector_to(indices.begin(), indices.end(), rhs_vec.begin());


  auto it = rhs_vec.begin();
  it += prob_size;
  grad_Ju.extract_subvector_to(indices.begin(), indices.end(), it);


  Vector<double> temp(prob_size);

  mass_matrix.vmult(temp, phi_vec);
  g.add(gamma, temp);
  stiffness_matrix.vmult(temp, y_vec);
  g += temp;
  mass_matrix.vmult(temp, u_vec);
  g -= temp;

  g *= -1;

  it += prob_size;
  g.extract_subvector_to(indices.begin(), indices.end(), it);
}

void LinearSystem::test_Kstar()
{
  assemble_Kstar();
  Vector<double> sol_Kstar(prob_size);
  Vector<double> test_rhs(prob_size);
  Vector<double> temp(prob_size);

  for (unsigned int i=0; i<prob_size; i++)
  {
      temp(i) = 1;
  }
  mass_matrix.vmult(test_rhs, temp);

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<2>(),
                                           boundary_values);

  MatrixTools::apply_boundary_values(boundary_values,
                                     K_star,
                                     sol_Kstar,
                                     test_rhs);

  SolverControl            solver_control(10000, 1e-6*test_rhs.l2_norm());
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(K_star, sol_Kstar, test_rhs, PreconditionIdentity());

  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(sol_Kstar, "solution_test_Kstar");
  data_out.build_patches();

  std::ofstream output("solution_test_Kstar.vtk");
  data_out.write_vtk(output);


}

void LinearSystem::solve()
{
  // Applying boundary values to phi1 and phi2
  FEValuesExtractors::Vector phi(0);
  //FEValuesExtractors::Scalar phi(0);
  ComponentMask phi_mask = fe_system.component_mask(phi);

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler_system,
                                           0,
                                           Functions::ZeroFunction<2>(3),
                                           boundary_values,
                                           phi_mask);

  MatrixTools::apply_boundary_values(boundary_values,
                                     A_matrix,
                                     solution_vec,
                                     rhs_vec);


  SolverControl                       solver_control(200000, 1e-6 * rhs_vec.l2_norm());
  SolverMinRes<Vector<double>>        solver(solver_control);
  solver.solve(A_matrix, solution_vec, rhs_vec, PreconditionIdentity());


  std::vector<unsigned int> indices(prob_size);
  std::iota(indices.begin(), indices.end(), 0);
  solution_vec.extract_subvector_to(indices.begin(), indices.end(), phi1_vec.begin());
  std::iota(indices.begin(), indices.end(), prob_size);
  solution_vec.extract_subvector_to(indices.begin(), indices.end(), phi2_vec.begin());

}

void LinearSystem::output_result_vectors() const
{
  DataOut<2> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(y_vec, "y_vec");
  data_out.build_patches();

  std::ofstream output("y_vec.vtk");
  data_out.write_vtk(output);

  DataOut<2> data_out_u;

  data_out_u.attach_dof_handler(dof_handler);
  data_out_u.add_data_vector(u_vec, "u_vec");
  data_out_u.build_patches();

  std::ofstream output_u("u_vec.vtk");
  data_out_u.write_vtk(output_u);
}

void LinearSystem::update_vectors(const Vector<double> &y, const Vector<double> &u)
{
  assert(y.size() == prob_size);
  assert(u.size() == prob_size);
  y_vec=y;
  u_vec=u;
}

void LinearSystem::solve_system()
{
  assemble_Kstar();
  assemble_A();
  assemble_rhs();
  solve();
}

double LinearSystem::evaluate_J() const
{
  double J;

  Vector<double> diff_yvec(y_vec);
  diff_yvec -= yd_vec;
  Vector<double> temp(prob_size);
  mass_matrix.vmult(temp, diff_yvec);

  J = 0.5*(temp*diff_yvec);

  mass_matrix.vmult(temp, u_vec);

  J = J + 0.5*alpha*(temp*u_vec);

  return J;
}

double LinearSystem::evaluate_g() const
{
  Vector<double> g(prob_size);
  Vector<double> temp(prob_size);

  mass_matrix.vmult(temp, phi_vec);
  g.add(gamma, temp);
  stiffness_matrix.vmult(temp, y_vec);
  g += temp;
  mass_matrix.vmult(temp, u_vec);
  g -= temp;

  return g.l2_norm();
}
