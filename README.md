This repository contains the code for the APSC project "Parareal-accelerated gradient flow iterations for optimal control problems".

# Subfolder structure
/include contains all the header files of the template classes 
/Examples contains all the files to compile and run the tests 
/Report contains the report of the project 


# Requirements

Before installing and using this library, ensure that the following dependencies are met:

- Linux machine with CMake ≥ 3.25.1
- GNU bash ≥ 5.2.15
- deal.II library version ≥ 9.5.0, which can be found in the mk module, 2024.0 version
- VisIt version ≥ 3.3.3


# Installation
This repository can be cloned as follows:
```
git clone https://github.com/fartiolii/pacs-project.git
cd pacs-project
```
Since it is only a header library, this command provides its installation.


# Test execution
To compile and run the examples, navigate to the desired example folder (e.g., `Examples/Test 2/Test_GradientFlow`) and follow these steps:
```
cmake -DDEAL_II_DIR=/path/to/dealii .
make release
make run
```
To compile and run in parallel the examples related to the Parareal and ParaFlow algorithms navigate to the desired example folder (e.g., `Examples/Test 2/Test_ParaFlow`) and follow these steps:
```
cmake -DDEAL_II_DIR=/path/to/dealii .
make release
make 
mpirun -np 4 ./test_paraFlow
```
In the example above, the number of processors has been chosen equal to 4 but it can be changed as desired.

To clean the directory from execution files run:
```
make clean
```
To remove all output files run:
```
make distclean
```

# Results visualization
Each test produces two output files ("y_vec.vtk" and "u_vec.vtk") which contain the obtained optimal vectors y and u, that
can be visualized using VisIt.

