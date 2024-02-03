#include <ParaFlowS.h>

using namespace dealii;

constexpr int dim=2;

int main()
{

    std::string linearSystem_file("../config_params.json");
    std::string ParaFlowS_file("ParaFlowS_params.json");

    ParaFlowS<dim> PFs(linearSystem_file, ParaFlowS_file);
    PFs.run();

    return 0;
}
