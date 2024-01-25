#include <GradientFlow.h>

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
  
    std::string GradientFlow_file("GradientFlow_params.json");
    
    std::vector<unsigned int> grid_refinements{2, 3, 4, 5, 6};
    
    
    for (unsigned int n=0; n<grid_refinements.size(); n++)
    {
    	std::cout << "Grid refined " << grid_refinements[n] << " times" << std::endl;
    	GradientFlow<dim> gf(gamma, nu, grid_refinements[n], GradientFlow_file);
    	gf.run();
    };

    return 0;
}
