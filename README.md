# 3d_Finite_Differences


This repo contains code to perform diffusion modeling in a plastic rod using a cuda implementation of finite differences. This is desigend to be run with a GPU on Google Colab. The ipynb files contains all the commands to compile, run, and do some simple plots of the diffusion modeling. First, though, you must upload the cuda files to your Google Drive, mount that on Colab, and adjust the paths in the compilation commands. 

## Diffusion

The two cuda files do very similar things, but save slightly different information. The `solverSaveFull.cu` file saves all of the oxygen concentrations periodically. This allows an examination of all the data at fixed points in time. It is necesary to not save on most iterations to preserve memory. As is, the execution of the cod is limited by writing to and reading from disk as it generates a 4 GB file. The `solverSaveSlice.cu` only saves the center slice in the vertical direction, but does so after every itration. This is possible because this involves saving a lot less data, only 400 MB. This executes in roughly 1/3 the time of the first file, despite doing exactly the same simulation. This code can be executed from `RunDiffusion.ipynb`. 

The rod being modeled currently has x and y dimension of 100 and z dimension of 1000.

## Radicals

The third cuda file, `solverRadicals.cu`, simulates radical formation and destruction. It has parameters to control the oxygen difusion, radical difusion, radical creation, radical-oxygen anihilation, and radical-radical cross linking. The file currently saves a 500 by 100 slice of the 500x100x100 crystal every cycle, as well as the activity, defeind as the number of radical-oxygen anihilations which took place. This code can be executed from `RunRadDamage.ipynb`.
