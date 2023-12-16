#include "include/ParaReal.h"

using namespace dealii;


int main(int argc, char **argv)
{
    Utilities::MPI::MPI_InitFinalize  mpi_initialization(argc, argv);
    MPI_Comm mpi_communicator(MPI_COMM_WORLD);
    unsigned int this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator));

    double T(20000); //T(120000);
    unsigned int n_pr_it = 3;

    if (this_mpi_process == 0)
    {
      ParaReal_Root pr;
      pr.set_final_time(T);
      pr.set_outer_step_size(2.0);
      pr.set_inner_step_size(0.5);
      pr.run(T, n_pr_it);
    }
    else
    {
      ParaReal_Rank_n pr;
      pr.set_final_time(T);
      pr.set_inner_step_size(0.5);
      pr.run(T, n_pr_it);
    }


    return 0;
}
