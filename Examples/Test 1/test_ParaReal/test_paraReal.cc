#include <ParaReal.h>

using namespace dealii;

constexpr int dim=2;


int main(int argc, char **argv)
{
    Utilities::MPI::MPI_InitFinalize  mpi_initialization(argc, argv);
    MPI_Comm mpi_communicator(MPI_COMM_WORLD);
    unsigned int this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator));
    
    //! Read parameters from file
    std::string linearSystem_file("../config_params.json");
    std::string ParaReal_file("ParaReal_params.json");


    if (this_mpi_process == 0)
    {
      ParaRealRoot<dim> pr(linearSystem_file, ParaReal_file);
      pr.run();
    }
    else
    {
      ParaRealRankN<dim> pr(linearSystem_file, ParaReal_file);
      pr.run();
    }

    return 0;
}
