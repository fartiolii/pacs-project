///@note: include path should be absolute
#include "include/ParaRealAlternative.h"

using namespace dealii;


int main(int argc, char **argv)
{
    Utilities::MPI::MPI_InitFinalize  mpi_initialization(argc, argv);
    MPI_Comm mpi_communicator(MPI_COMM_WORLD);
    unsigned int this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator));

    unsigned int n_inner_it = 200;
    //double inner_step_size = 1.5;
    //double outer_step_size = 8.0;
    GFStepType outer_method(GFStepType::ADAM);
    GFStepType inner_method(GFStepType::ADAM);

    if (this_mpi_process == 0)
    {
      ParaReal_Root pr(outer_method, inner_method);
      pr.set_num_inner_it(n_inner_it);
      //pr.set_outer_step_size(outer_step_size);
      //pr.set_inner_step_size(inner_step_size);
      pr.run();
    }
    else
    {
      ParaReal_Rank_n pr(inner_method);
      pr.set_num_inner_it(n_inner_it);
      //pr.set_inner_step_size(inner_step_size);
      pr.run();
    }


    return 0;
}
