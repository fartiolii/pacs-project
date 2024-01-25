///@note: include path should be absolute
#include <ParaFlowS.h>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace dealii;
using json = nlohmann::json;

constexpr int dim=2;

int main()
{
    // Read JSON object from file
    std::ifstream file("params.json");

    json parameters;
    file >> parameters;
    file.close();

    unsigned int N = parameters["N"];
    unsigned int n_inner_it = parameters["n_inner_it"];
    unsigned int n_outer_it = parameters["n_outer_it"];
    double inner_step_size = parameters["inner_step_size"];
    double outer_step_size = parameters["outer_step_size"];
    GFStepType outer_method(GFStepType::EULER);
    GFStepType inner_method(GFStepType::EULER);

    ParaFlowS<dim> pr(outer_method, inner_method);
    pr.run(N, outer_step_size, inner_step_size, n_outer_it, n_inner_it);


    return 0;
}
