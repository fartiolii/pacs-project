This repository contains the code for the APSC project.

Requirements
This repository requires to install the Finite Element library deal.II available at https://www.dealii.org/.
It is recommended to install it with candi as follows (advanced configuration available at https://github.com/dealii/candi):

sudo apt-get update
sudo apt-get upgrade
git clone https://github.com/dealii/candi.git
cd candi
./candi.sh

Compile and run the test 

cmake -DDEAL_II_DIR=/path/to/dealii .
make release
make
mpirun -np 4 ./test 


