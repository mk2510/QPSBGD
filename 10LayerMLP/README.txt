
In order to run the code for the experiments two things need to be prepared:
1) Conda enviroment
2) Aquire a Gurobi license
3) Aquire the data.

Conda enviroment:
Prepare and install the conda enviroment with: 'conda create --name <env> --file requirements.txt'

Gurobi license:
On the gurobi website as an accademic person, it is possible to aquire a free license-

Data aquisition:
To reproduce the experiments of the paper it is required to download the datasets:


wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t

place those files in the data folder

Experiments:
The experiments are in the file 10layers.ipynb. The result is figure 3 right plot




