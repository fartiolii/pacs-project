
This repository contains the code for the APSC project.

# Requirements
@note: dealii in the mk module is not ok? if not put the minimal required version.
This repository requires to install the Finite Element library deal.II available at https://www.dealii.org/.
It is recommended to install it with candi as follows (advanced configuration available at https://github.com/dealii/candi):

```
sudo apt-get update
sudo apt-get upgrade
git clone https://github.com/dealii/candi.git
cd candi
./candi.sh
```

# Cloning the repository
This repository can be cloned as follows:
```
git clone https://github.com/fartiolii/pacs-project.git
cd pacs-project
```

# Compile and run the test
@note: all the test in the report should be easily reproducible
@note: meshes in a dedicated folder
@note: scalability tests (weak and strong)

```
cmake -DDEAL_II_DIR=/path/to/dealii .
make release
make
mpirun -np 4 ./test
```

# Results visualization

The obtained solution vectors y and u (contained in the files "y_vec.vtk" and "u_vec.vtk")
can be visualized using VisIt available at https://visit-dav.github.io/visit-website/index.html.

The expected solution vector files are provided for testing in the folder TestSolutionVectors.

@note: evaluation table: https://webeep.polimi.it/pluginfile.php/542661/mod_resource/content/1/Evaluation.pdf
@note: see examples of reports https://formaggia.faculty.polimi.it/?page_id=129
@note: doxygen comments are not mandatory but are very appreiated (you can use copilot or chatgpt to generate them, but double check the result)
@note: leaving the code as templated on the dimension is complicated from the point of view of the mathematics?