#include <ParaReal.h>

using namespace dealii;


///@note: read parameters from a file for more flexibility

int main(int argc, char **argv)
{
    Utilities::MPI::MPI_InitFinalize  mpi_initialization(argc, argv);
    MPI_Comm mpi_communicator(MPI_COMM_WORLD);
    unsigned int this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator));

    constexpr int dim=2;
    double T(300);
    double inner_step_size = 1.0;
    double outer_step_size = 1.5;

    GFStepType outer_method(GFStepType::EULER);
    GFStepType inner_method(GFStepType::EULER);

    if (this_mpi_process == 0)
    {
      ParaReal_Root<dim> pr(outer_method, inner_method);
      pr.set_final_time(T);
      pr.set_outer_step_size(outer_step_size);
      pr.set_inner_step_size(inner_step_size);
      pr.run();
    }
    else
    {
      ParaReal_Rank_n<dim> pr(inner_method);
      pr.set_final_time(T);
      pr.set_inner_step_size(inner_step_size);
      pr.run();
    }


    return 0;
}
