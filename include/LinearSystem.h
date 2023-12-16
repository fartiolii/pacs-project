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


// ...and this is to import the deal.II namespace into the global scope:
using namespace dealii;

class LinearSystem
{
public:
  LinearSystem();

  void run();
  void update_vectors(const Vector<double> &y, const Vector<double> &u);
  void solve_system();
  void output_result_vectors() const;
  const Vector<double>& get_phi1() const { return phi1_vec; };
  const Vector<double>& get_phi2() const { return phi2_vec; };
  const Vector<double>& get_g() const { return g_vec; };
  double evaluate_J() const;
  double evaluate_g() const;
  void test_Kstar();

  unsigned int get_vector_size() const { return prob_size; };

  //friend void GradientFlow::output_results_vectors(const LinearSystem& linearSystem, const GradientFlow& gradFlow);



private:
  void make_grid();
  void initialize_dimensions();
  void setup_system();
  void assemble_system();
  void initialize_vectors_y_u();

  // Then there are the member functions that mostly do what their names
  // suggest and whose have been discussed in the introduction already. Since
  // they do not need to be called from outside, they are made private to this
  // class.up_system();

  void set_phi();
  void assemble_Kstar();
  void set_global_sparsity_pattern();
  void assemble_A();
  void assemble_rhs();

  void setup_linear_system();

  void solve();
  void output_results() const;



  Triangulation<2> triangulation;

  //FE_Q<2>          fe;
  const MappingFE<2>     mapping;
  const FE_SimplexP<2>   fe;

  const FESystem<2>      fe_system;

  const QGaussSimplex<2> quadrature_formula;

  DoFHandler<2>          dof_handler;
  DoFHandler<2>          dof_handler_system;

  unsigned int          prob_size;
  double                alpha=0.1;
  double                beta=0;
  double                gamma=0.5;
  double                yd=1;
  SparsityPattern       sparsity_pattern;
  SparseMatrix<double>  stiffness_matrix;
  SparseMatrix<double>  mass_matrix;
  SparseMatrix<double>  K_star;

  SparsityPattern       global_sparsity_pattern;
  SparseMatrix<double>  A_matrix;

  // ...and variables which will hold the right hand side and solution
  // vectors.
  Vector<double> y_vec;
  Vector<double> u_vec;

  Vector<double> yd_vec;
  Vector<double> phi_vec;
  Vector<double> Jacobian_phi;
  Vector<double> rhs_vec;
  Vector<double> phi1_vec;
  Vector<double> phi2_vec;
  Vector<double> g_vec;

  Vector<double> solution_vec;

};

// @sect4{Step3::Step3}

// In the constructor, we set the polynomial degree of the finite element and
// the number of quadrature points. Furthermore, we initialize the MappingFE
// object with a (linear) FE_SimplexP object so that it can work on simplex
// meshes.
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



// @sect4{Step3::make_grid}

void LinearSystem::make_grid()
{
  triangulation.clear();
  GridIn<2>(triangulation).read("tri.msh");
  triangulation.refine_global(3);


  std::ofstream out("grid-LinSys.svg");
  GridOut       grid_out;
  grid_out.write_svg(triangulation, out);
  //std::cout << "Grid written to grid-LinSys.svg" << std::endl;


  //std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
}


// @sect4{Step3::setup_system}

// Next we enumerate all the degrees of freedom and set up matrix and vector
// objects to hold the system data. Enumerating is done by using
// DoFHandler::distribute_dofs(), as we have seen in the step-2 example. Since
// we use the FE_Q class and have set the polynomial degree to 1 in the
// constructor, i.e. bilinear elements, this associates one degree of freedom
// with each vertex. While we're at generating output, let us also take a look
// at how many degrees of freedom are generated:
void LinearSystem::setup_system()
{
  dof_handler.distribute_dofs(fe);
  dof_handler_system.distribute_dofs(fe_system);
  /*
  std::cout << "Number of degrees of freedom vector: " << dof_handler.n_dofs()
            << std::endl;
  std::cout << "Number of degrees of freedom system: " << dof_handler_system.n_dofs()
            << std::endl;
  */
  // There should be one DoF for each vertex. Since we have a 32 times 32
  // grid, the number of DoFs should be 33 times 33, or 1089.

  // As we have seen in the previous example, we set up a sparsity pattern by
  // first creating a temporary structure, tagging those entries that might be
  // nonzero, and then copying the data over to the SparsityPattern object
  // that can then be used by the system matrix.
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  // Note that the SparsityPattern object does not hold the values of the
  // matrix, it only stores the places where entries are. The entries
  // themselves are stored in objects of type SparseMatrix, of which our
  // variable system_matrix is one.
  //
  // The distinction between sparsity pattern and matrix was made to allow
  // several matrices to use the same sparsity pattern. This may not seem
  // relevant here, but when you consider the size which matrices can have,
  // and that it may take some time to build the sparsity pattern, this
  // becomes important in large-scale problems if you have to store several
  // matrices in your program.
  stiffness_matrix.reinit(sparsity_pattern);
  mass_matrix.reinit(sparsity_pattern);

  // The last thing to do in this function is to set the sizes of the right
  // hand side vector and the solution vector to the right values:
  prob_size = dof_handler.n_dofs();

  initialize_dimensions();

  // to remove
  for (unsigned int i=0; i<prob_size; i++)
  {
      yd_vec(i) = yd;
  }

}

// @sect4{Step3::assemble_system}
void LinearSystem::initialize_dimensions()
{
  assert(prob_size != 0);
  y_vec.reinit(prob_size);
  u_vec.reinit(prob_size);
  phi_vec.reinit(prob_size);
  Jacobian_phi.reinit(prob_size);
  yd_vec.reinit(prob_size);
  g_vec.reinit(prob_size);


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
      // We are now sitting on one cell, and we would like the values and
      // gradients of the shape functions be computed, as well as the
      // determinants of the Jacobian matrices of the mapping between
      // reference cell and true cell, at the quadrature points. Since all
      // these values depend on the geometry of the cell, we have to have the
      // FEValues object re-compute them on each cell:
      fe_values.reinit(cell);

      // Next, reset the local cell's contributions to global matrix and
      // global right hand side to zero, before we fill them:
      cell_matrix_stiffness = 0;
      cell_matrix_mass = 0;

      // Now it is time to start integration over the cell, which we
      // do by looping over all quadrature points, which we will
      // number by q_index.
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          // First assemble the matrix: For the Laplace problem, the
          // matrix on each cell is the integral over the gradients of
          // shape function i and j. Since we do not integrate, but
          // rather use quadrature, this is the sum over all
          // quadrature points of the integrands times the determinant
          // of the Jacobian matrix at the quadrature point times the
          // weight of this quadrature point. You can get the gradient
          // of shape function $i$ at quadrature point with number q_index by
          // using <code>fe_values.shape_grad(i,q_index)</code>; this
          // gradient is a 2-dimensional vector (in fact it is of type
          // Tensor@<1,dim@>, with here dim=2) and the product of two
          // such vectors is the scalar product, i.e. the product of
          // the two shape_grad function calls is the dot
          // product. This is in turn multiplied by the Jacobian
          // determinant and the quadrature point weight (that one
          // gets together by the call to FEValues::JxW() ). Finally,
          // this is repeated for all shape functions $i$ and $j$:
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
          // We then do the same thing for the right hand side. Here,
          // the integral is over the shape function i times the right
          // hand side function, which we choose to be the function
          // with constant value one (more interesting examples will
          // be considered in the following programs).
          /*
          for (const unsigned int i : fe_values.dof_indices())
            cell_y_vec(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            1. *                                // f(x_q)
                            fe_values.JxW(q_index));            // dx
          */
        }
      // Now that we have the contribution of this cell, we have to transfer
      // it to the global matrix and right hand side. To this end, we first
      // have to find out which global numbers the degrees of freedom on this
      // cell have. Let's simply ask the cell for that information:
      cell->get_dof_indices(local_dof_indices);

      // Then again loop over all shape functions i and j and transfer the
      // local elements to the global matrix. The global numbers can be
      // obtained using local_dof_indices[i]:
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

      // And again, we do the same thing for the right hand side vector.
      //for (const unsigned int i : fe_values.dof_indices())
        //system_rhs(local_dof_indices[i]) += cell_rhs(i);

    }

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<2>(),
                                           boundary_values);
  // Now that we got the list of boundary DoFs and their respective boundary
  // values, let's use them to modify the system of equations
  // accordingly. This is done by the following function call:
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

  /*
 std::cout << "Dim Stiffness matrix: " << stiffness_matrix.m()<< "x" <<  stiffness_matrix.n() << std::endl;
 std::cout << "Number of elements in the sparsity pattern: " << stiffness_matrix.n_nonzero_elements() << std::endl;
 std::cout << "Number of actually non-zero elements stiffness matrix: " << stiffness_matrix.n_actually_nonzero_elements() << std::endl;
 std::cout << "Dim Mass matrix: " << mass_matrix.m()<< "x" <<  mass_matrix.n() << std::endl;
 std::cout << "Number of elements in the sparsity pattern: " << mass_matrix.n_nonzero_elements() << std::endl;
 std::cout << "Number of actually non-zero elements mass matrix: " << mass_matrix.n_actually_nonzero_elements() << std::endl;
 */
}


void LinearSystem::initialize_vectors_y_u()
{
  /*
  std::ifstream in_y("y_solution");
  std::ifstream in_u("u_solution");

  y_vec.block_read(in_y);
  u_vec.block_read(in_u);

  for (unsigned int i=0; i<prob_size; i++)
  {
      y_vec(i) = 0.1;
      u_vec(i) = 0.01;
  }
  */
  y_vec.add(0.01);
  u_vec.add(0.01);

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


  // Set phi on boundary values to 0
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
    //unsigned int global_nrows = 3*prob_size;
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
  //set_global_sparsity_pattern();
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
  g.extract_subvector_to(indices.begin(), indices.end(), g_vec.begin());
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
  // Now that we got the list of boundary DoFs and their respective boundary
  // values, let's use them to modify the system of equations
  // accordingly. This is done by the following function call:
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
  // Now that we got the list of boundary DoFs and their respective boundary
  // values, let's use them to modify the system of equations
  // accordingly. This is done by the following function call:

  /*
  std:: cout << phi_mask << std::endl;
  std::cout << "BV print "<< std::endl;
  for (const auto &bv: boundary_values)
  {
      std::cout << bv.second << std::endl;
  }

  Vector<double> grad_Jy(prob_size);

  Vector<double> diff_yvec(y_vec);
  diff_yvec -= yd_vec;

  mass_matrix.vmult(grad_Jy, diff_yvec);

  for (auto i=0; i<prob_size; i++)
  {
      std::cout << "rhs " << rhs_vec(i) << " grad_Jy " << grad_Jy(i) << std::endl;
  }


  */
  MatrixTools::apply_boundary_values(boundary_values,
                                     A_matrix,
                                     solution_vec,
                                     rhs_vec);

  /*
  IndexSet boun_val = DoFTools::extract_boundary_dofs(dof_handler_system);

  std::cout << "Boundary Values rhs: " << std::endl;
  for (const auto &bv: boun_val)
  {
      std::cout << rhs_vec(bv) << std::endl;
  }
  */
  SolverControl                       solver_control(200000, 1e-6 * rhs_vec.l2_norm());
  SolverMinRes<Vector<double>>        solver(solver_control);
  solver.solve(A_matrix, solution_vec, rhs_vec, PreconditionIdentity());


  std::vector<unsigned int> indices(prob_size);
  std::iota(indices.begin(), indices.end(), 0);
  solution_vec.extract_subvector_to(indices.begin(), indices.end(), phi1_vec.begin());
  std::iota(indices.begin(), indices.end(), prob_size);
  solution_vec.extract_subvector_to(indices.begin(), indices.end(), phi2_vec.begin());
  //std::iota(indices.begin(), indices.end(), 2*prob_size);
  //rhs_vec.extract_subvector_to(indices.begin(), indices.end(), g_vec.begin());

  //std::cout << "g vec " << g_vec.l2_norm() << std::endl;
  //std::cout << "gamma vec " << std::endl;
  /*
  Vector<double> sol_3(prob_size);
  std::iota(indices.begin(), indices.end(), 2*prob_size);
  solution_vec.extract_subvector_to(indices.begin(), indices.end(), sol_3.begin());

  Vector<double> grad_Jy(prob_size);

  for (unsigned int i=0; i<prob_size; i++)
  {
      grad_Jy(i) = rhs_vec(i);
  }

  SparseMatrix<double> K_temp(sparsity_pattern);
  unsigned int disp_col = 2*prob_size;
  for (auto it=sparsity_pattern.begin(); it!=sparsity_pattern.end(); it++)
    K_temp.set(it->row(), it->column(), A_matrix(it->row(), disp_col+it->column()));

  Vector<double> temp(prob_size);
  K_star.vmult(temp, sol_3);
  grad_Jy -= temp;

  Vector<double> temp1(phi1_vec);
  temp1 -= grad_Jy;

  std::cout << "residual phi1: " << temp1.l2_norm() << std::endl;

  //std::vector<double> local_phi_1 (n_q_points);

  const FEValuesExtractors::Scalar phi1(0);
  const FEValuesExtractors::Scalar phi2(1);

  FEValues<2> fe_values(fe_system,
                        quadrature_formula,
                        update_values);

  std::vector<double> phi_1(prob_size);
  std::vector<double> phi_2(prob_size);

  for (DoFHandler<2>::active_cell_iterator cell=dof_handler_system.begin_active(); cell!=dof_handler_system.end(); ++cell)
    {
      fe_values.reinit(cell);

      fe_values[phi1].get_function_values (solution_vec,
                                           phi_1);
      fe_values[phi2].get_function_values (solution_vec,
                                           phi_2);
    }

  for (unsigned int i=0; i<prob_size; i++)
  {
      phi1_vec(i) = phi_1[i];
      phi2_vec(i) = phi_2[i];
      std::cout << phi_1[i] << std::endl;
  }
  */

  /*
  Vector<double> temp(dof_handler_system.n_dofs());


  A_matrix.vmult(temp, solution_vec);
  temp -= rhs_vec;
  std::cout << "Residual: " << temp.linfty_norm() << std::endl;



  IndexSet boundary_val = DoFTools::extract_boundary_dofs(dof_handler_system);

  std::cout << "Boundary Values 1: " << std::endl;
  for (const auto &bv: boundary_val)
  {
      std::cout << solution_vec(bv) << std::endl;
      std::cout << rhs_vec(bv) << std::endl;
  }

  std::cout << "Boundary Values 2: " << std::endl;
  for (const auto &bv: boundary_val)
  {
      std::cout << phi2_vec(bv) << std::endl;
  }
  */
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


void LinearSystem::run()
{
  make_grid();
  setup_system();
  assemble_system();
  assemble_Kstar();
  assemble_A();
  assemble_rhs();
  //test_Kstar();
  solve();

  /*
  Vector<double> new_y(prob_size);
  new_y.add(2);
  Vector<double> new_u(prob_size);
  new_u.add(0.5);
  update_vectors(new_y, new_u);
  std::cout << y_vec(0) << std::endl;
  std::cout << u_vec(0) << std::endl;

  //output_results();
  */
  std::ofstream out("global_sp.svg");
  global_sparsity_pattern.print_svg(out);

}

/*
int main()
{
  deallog.depth_console(2);

  //LinearSystem linear_system;
  //linear_system.run();

  LinearSystem upd_linear;
  upd_linear.solve_system();


  unsigned int size = upd_linear.get_vector_size();
  Vector<double> new_y(size);
  new_y.add(2);
  Vector<double> new_u(size);
  new_u.add(0.5);
  upd_linear.update_vectors(new_y, new_u);

  upd_linear.solve_system();


  std::cout << "J: " << upd_linear.evaluate_J()<< std::endl;
  const Vector<double> phi1(upd_linear.get_phi1());
  const Vector<double> phi2(upd_linear.get_phi2());


  return 0;
}
*/
