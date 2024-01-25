#include <GradientFlow.h>

using namespace dealii;
constexpr int dim=2;


int main()
{
	
    std::string linearSystem_file("../config_params.json");
    std::string GradientFlow_file("GradientFlow_params.json");
    
    GradientFlow<dim> gf(linearSystem_file, GradientFlow_file);
    gf.run();
    

    return 0;
}
