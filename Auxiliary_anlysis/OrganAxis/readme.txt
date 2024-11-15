In order to support these analysis pipelines you need to setup an environment (we call it imagespot) as follows 
Instructions for imagespot env 

1) Create a new conda environment:
$ conda create -n imagespot python=3.9
say 'y' when asked

2) activate new environment 
$ conda activate imagespot 

3) install scanpy
$ conda install -c conda-forge scanpy python-igraph leidenalg

4) install jupyter-lab 
$ conda install -c conda-forge jupyterlab

6) install open-cv
$ pip install opencv-python

6.5) optional - install cellpose 
		$ pip install cellpose 
		$ pip uninstall torch
		$ conda install pytorch cudatoolkit=11.3 -c pytorch


8) add the new enviroment to jupyter lab path  
$ ipython kernel install --name imagespot --user
