#include "GradientFlow.h"
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>

//#include "gnuplot-iostream.h"

using namespace dealii;

const unsigned int root = 0;

class ParaRealBase
{
public:

  ParaRealBase();
  virtual void run(const double T, const unsigned int n_pr_iter) = 0;
  void set_final_time(const double T);

protected:

  MPI_Comm                    mpi_communicator;
  //ConditionalOStream          pcout;

  double                      global_T;
  unsigned int                n_inner_iter;

  const unsigned int          n_mpi_processes;
  const unsigned int          this_mpi_process;

};

ParaRealBase::ParaRealBase()
:     mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    //, pcout(std::cout, (this_mpi_process == 0))
{}

void ParaRealBase::set_final_time(const double T)
{
  global_T = T;
}


class ParaReal_Root: public ParaRealBase
{
public:

  ParaReal_Root();
  void set_inner_step_size(const double step_size=0.1);
  void set_outer_step_size(const double outer_it=1.0);
  virtual void run(const double T, const unsigned int n_pr_iter) override;

private:

  void outer_loop();

  double                      delta_G;
  double                      delta_F;

  GradientFlow                gf_F;
  GradientFlow                gf_G;

  unsigned int                n_outer_iter;

  std::vector<double>         outer_time_vector;

  std::vector<std::tuple<Vector<double>, Vector<double>>> F_vectors; //to receive
  std::vector<std::tuple<Vector<double>, Vector<double>>> G_old_vectors;
  std::vector<std::tuple<Vector<double>, Vector<double>>> G_new_vectors;
  std::vector<std::tuple<Vector<double>, Vector<double>>> new_yu_vectors; //to send

};

ParaReal_Root::ParaReal_Root()
:     gf_F()
    , gf_G()
    , F_vectors(n_mpi_processes)
    , G_old_vectors(n_mpi_processes)
    , G_new_vectors(n_mpi_processes)
    , new_yu_vectors(n_mpi_processes)
{}


void ParaReal_Root::set_inner_step_size(const double step_size)
{
    assert(n_mpi_processes != 0);

    delta_F = step_size;
    gf_F.set_step_size(delta_F);
    n_inner_iter = static_cast<unsigned int>(global_T/(n_mpi_processes*step_size));
    std::cout << " n_inner_iter: " << n_inner_iter << std::endl;
    //n_inner_iter = inner_it;
    //delta_F = std::min(global_T/(n_mpi_processes*n_inner_iter), 5.0);
    //gf_F.set_step_size(delta_F);
}

void ParaReal_Root::set_outer_step_size(const double outer_it)
{
  assert(n_mpi_processes != 0);

  //delta_G = std::min(global_T/(n_mpi_processes*outer_it), 20.0);
  //gf_G.set_step_size(delta_G);
  delta_G = outer_it;
  gf_G.set_step_size(delta_G);
  n_outer_iter = static_cast<unsigned int>(global_T/(n_mpi_processes*outer_it));
  std::cout << " n_outer_iter: " << n_outer_iter << std::endl;
  /*
  for (unsigned int i=0; i<n_mpi_processes; i++)
    outer_time_vector.push_back(i*delta_G*outer_it);
  outer_time_vector.push_back(global_T);
  */
}

void ParaReal_Root::run(const double T, const unsigned int n_pr_iter)
{
  set_final_time(T);

  //set_inner_step_size();
  //set_outer_step_size();

  std::cout << "Outer step: " << delta_G << std::endl;
  std::cout << "Inner step: " << delta_F << std::endl;

  std::cout << "Initialization" << std::endl;
  // Initialization
  for (unsigned int i=0; i<n_mpi_processes; i++)
  {
    // NB: We are assuming initial conditions equal all along the n initial points (no inner gf)
    gf_G.run(n_outer_iter);

    Vector<double> y = gf_G.get_y_vec();
    Vector<double> u = gf_G.get_u_vec();

    G_old_vectors[i] = std::make_tuple(y, u);
    new_yu_vectors[i] = std::make_tuple(y, u);

    gf_G.output_iteration_results();
  }
  G_new_vectors = G_old_vectors;
  gf_G.output_results_vectors();

  for (unsigned int it=0; it < n_pr_iter; it++)
  {
      std::cout << "Iteration n: " << it << std::endl;
      // Send Results to all ranks
      for (unsigned int rank=1; rank<n_mpi_processes; rank++)
      {
        Utilities::MPI::isend(new_yu_vectors[rank-1], mpi_communicator, rank);
      }

      if (it == 0)
      {
        // Root computes its own F operator in the first interval [t0, t1]: i.e. computes F at t1
        gf_F.run(n_inner_iter);
        F_vectors[0] = std::make_tuple(gf_F.get_y_vec(), gf_F.get_u_vec());
      }

      // Receive F operator from all ranks
      for (unsigned int rank=1; rank<n_mpi_processes; rank++)
      {
        Utilities::MPI::Future<std::tuple<Vector<double>, Vector<double>>> future = Utilities::MPI::irecv<std::tuple<Vector<double>, Vector<double>>>(mpi_communicator, rank);
        F_vectors[rank] = future.get();
      }

      // Compute G_k+1
      G_old_vectors = G_new_vectors;
      for (unsigned int rank=1; rank<n_mpi_processes; rank++)
      {
        gf_G.set_initial_vectors(std::get<0>(F_vectors[rank-1]), std::get<1>(F_vectors[rank-1]));
        gf_G.run(n_outer_iter);
        G_new_vectors[rank] = std::make_tuple(gf_G.get_y_vec(), gf_G.get_u_vec());
        //gf_G.output_iteration_results();
      }

      // Compute new y and u (used as initial conditions of inner processes)
      for (unsigned int rank=0; rank<n_mpi_processes; rank++)
      {
        Vector<double> y_temp(std::get<0>(G_new_vectors[rank]));

        y_temp -= std::get<0>(G_old_vectors[rank]);

        y_temp += std::get<0>(F_vectors[rank]);

        Vector<double> u_temp(std::get<1>(G_new_vectors[rank]));
        u_temp -= std::get<1>(G_old_vectors[rank]);
        u_temp += std::get<1>(F_vectors[rank]);

        new_yu_vectors[rank] = std::make_tuple(y_temp, u_temp);

      }


      // Output results of this iteration process
      for (unsigned int rank=0; rank<n_mpi_processes; rank++)
      {
        gf_G.set_initial_vectors(std::get<0>(new_yu_vectors[rank]), std::get<1>(new_yu_vectors[rank]));
        gf_G.output_iteration_results();
      }

  }

  // Output vectors
  std::cout << "Final results " << std::endl;
  gf_G.set_initial_vectors(std::get<0>(new_yu_vectors[n_mpi_processes-1]), std::get<1>(new_yu_vectors[n_mpi_processes-1]));
  gf_G.output_iteration_results();
  gf_G.output_results_vectors();
}

class ParaReal_Rank_n: public ParaRealBase
{
public:

  ParaReal_Rank_n();
  void set_inner_step_size(const double step_size=0.1);
  virtual void run(const double T, const unsigned int n_pr_iter) override;

private:

  //virtual void set_step_size(const unsigned int inner_it=20) override
  double                      delta_F;

  GradientFlow                gf_F;

  std::tuple<Vector<double>, Vector<double>> initial_time_vectors; //to receive
  std::tuple<Vector<double>, Vector<double>> final_time_vectors; //to send
};

ParaReal_Rank_n::ParaReal_Rank_n()
:     gf_F()
{}

/*
void ParaReal_Rank_n::set_step_size(const unsigned int inner_it)
{
  n_inner_iter = inner_it;
  delta_F = global_T/(n_mpi_processes*n_inner_iter);
  gf_F.set_step_size(delta_F);
}*/
void ParaReal_Rank_n::set_inner_step_size(const double step_size)
{
    assert(n_mpi_processes != 0);

    delta_F = step_size;
    gf_F.set_step_size(delta_F);
    n_inner_iter = static_cast<unsigned int>(global_T/(n_mpi_processes*step_size));
    std::cout << "delta_F: " << delta_F << std::endl;
    //n_inner_iter = inner_it;
    //delta_F = std::min(global_T/(n_mpi_processes*n_inner_iter), 5.0);
    //gf_F.set_step_size(delta_F);
}

void ParaReal_Rank_n::run(const double T, const unsigned int n_pr_iter)
{
  set_final_time(T);
  //set_inner_step_size();

  for (unsigned int it=0; it < n_pr_iter; it++)
  {
      // Receive initial conditions from root
      Utilities::MPI::Future<std::tuple<Vector<double>, Vector<double>>> future = Utilities::MPI::irecv<std::tuple<Vector<double>, Vector<double>>>(mpi_communicator, root);
      initial_time_vectors = future.get();

      // Set received vectors as initial conditions of inner gf
      gf_F.set_initial_vectors(std::get<0>(initial_time_vectors), std::get<1>(initial_time_vectors));

      gf_F.run(n_inner_iter);

      final_time_vectors = std::make_tuple(gf_F.get_y_vec(), gf_F.get_u_vec());

      // Send the final vectors to the root
      Utilities::MPI::isend(final_time_vectors, mpi_communicator, root);
  }

}
