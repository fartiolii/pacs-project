#include <ParaFlow.h>


using namespace dealii;
constexpr int dim=2;

    
int main(int argc, char **argv)
{
    Utilities::MPI::MPI_InitFinalize  mpi_initialization(argc, argv);
    MPI_Comm mpi_communicator(MPI_COMM_WORLD);
    unsigned int this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator));
    
    //! Read parameters from file
    std::string linearSystem_file("../config_params.json");
    std::string ParaFlow_file("ParaFlow_params.json");


    if (this_mpi_process == 0)
    {
      ParaFlowRoot<dim> pf(linearSystem_file, ParaFlow_file);
      pf.run();
    }
    else
    {
      ParaFlowRankN<dim> pf(linearSystem_file, ParaFlow_file);
      pf.run();
    }

    return 0;
}

