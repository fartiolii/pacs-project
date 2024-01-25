#include <GradientFlow.h>

using namespace dealii;


///@note: read parameters from a file for more flexibility

int main()
{
    constexpr int dim=2;

    GradientFlowEuler<dim> gf_Euler;
    GradientFlowAdam<dim> gf_Adam;

    //gf_Euler.set_step_size(1.5);
    //gf_Euler.run();

    gf_Adam.set_step_size(1.0);
    gf_Adam.run();

    return 0;
}
