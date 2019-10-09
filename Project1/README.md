# Project 1
This folder contains the programs used in project 1. The report written in this project can be found in FYS_STK3155_Project1.pdf, and additional figures can be found in the folders: Frankefigs, Terrainfigs, biasvar.

## Program description
project1.py contains a class Project1 and some utility functions related to general data analysis of 2d->1d functions. The class is used both when analyzing the Franke function, and when analyzing the terrain data.

terrain.py contains the class Terrain which inherits the methods from Project1. It also loads the data, instead of generating it based on a function such as the Franke function. The program also contains utility funcitons for further analysis of the terrain data.

Ridge.py contains the ridge function, both our own and the one in scikit-learn. It is stored as a class with initialized lambda parameters. Lasso.py is similar, but for the Lasso function

test_class.py contains unit tests of the OLS, ridge and LASSO methods, can be run using pytest or nosetests

terrain.py uses a terrain dataset "SRTM_data_Norway_1.tif". This can be downloaded from https://github.com/CompPhysics/MachineLearning/blob/master/doc/Projects/2019/Project1/DataFiles/SRTM_data_Norway_1.tif.
