#include <ParaFlow.h>


using namespace dealii;
constexpr int dim=2;

    
int main(int argc, char **argv)
{
    Utilities::MPI::MPI_InitFinalize  mpi_initialization(argc, argv);
    MPI_Comm mpi_communicator(MPI_COMM_WORLD);
    unsigned int this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator));
    
    std::ifstream file("../config_params.json");
    json parameters;
    file >> parameters;
    file.close();

    double gamma = parameters["gamma"];
    double nu = parameters["nu"];
    
    //! Read parameters of ParaFlow from file
    std::string ParaFlow_file("ParaFlow_params.json");
    
    std::vector<unsigned int> grid_refinements{2, 3, 4, 5};
    
    
    for (unsigned int n=0; n<grid_refinements.size(); n++)
    {
    	if (this_mpi_process == 0)
	{
    	   std::cout << "\n\nGrid refined " << grid_refinements[n] << " times" << std::endl;
	   ParaFlowRoot<dim> pf(gamma, nu, grid_refinements[n], ParaFlow_file);
	   pf.run();
	}
	else
	{
	   ParaFlowRankN<dim> pf(gamma, nu, grid_refinements[n], ParaFlow_file);
	   pf.run();
	}
    }



    return 0;
}

