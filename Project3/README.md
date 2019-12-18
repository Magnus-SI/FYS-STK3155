# Project 3
## Program descriptions
 - **models.py**  
  Contains most of the classes and functions used to produce the results that can be seen in the report. Including implementations of xgboost, Logistic Regression, Neural networks, a combination of a neural network and xgboost (not used in the final report), as well as the class analyze used to analyze and compare multiple models. It also contains a function for hyperparameter tuning, utilized by outer functions in the program such as xgbtreeopter

  - **reportfigs.py**  
  Imports from models.py and runs the necessary functions to produce the plots and tables in the report.

  - **pulsar.py**  
  Loads and performs preprocessing on the pulsar data. Note that this requires the file pulsar_stars.csv in the Project3 subfolder to run. This file can be found on kaggle https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star.

The Folder AdditionalFigures contains additional figures that did not make it to the report.

**CNN.py** and **CNNtoXGB.py** are partially finished programs we ended up not using.
