#include <ParaFlowS.h>

using namespace dealii;

constexpr int dim=2;

int main()
{
    std::ifstream file("../config_params.json");
    json parameters;
    file >> parameters;
    file.close();

    double gamma = parameters["gamma"];
    double nu = parameters["nu"];
    
    //! Read ParaFlowS parameters from file
    std::string ParaFlowS_file("ParaFlowS_params.json");
    
    std::vector<unsigned int> grid_refinements{2, 3, 4, 5, 6};
    
    
    for (unsigned int n=0; n<grid_refinements.size(); n++)
    {
    	std::cout << "\n\nGrid refined " << grid_refinements[n] << " times" << std::endl;
    	ParaFlowS<dim> PFs(gamma, nu, grid_refinements[n], ParaFlowS_file);
    	PFs.run();
    };

    return 0;
}
