# Project 2
Note that the codes and report are not yet finalized. Final version will be added before the extended deadline on wednesday 13.11 along with an update of this readme, describing the functionalities of our code.
Report saved as FYS_STK3155Project2.pdf

## Program descriptions
 - **Analyze.py**  
 Contains a class 'ModelAnalysis' to preform k-fold cross validation, return ROC curves and other functions to analyze a model.

  - **cancer.py**  
  Preformes logistic regression, and trains a neural network on the cancer dataset, given by scikit learn. Used as a test case.

  - **creditcard.py**  
  Preformes logistic regression, and trains three neural networks on the credit card data (with the sigmoid, ReLU and ELU activation functions), the main dataset used for classification in our report. Finds number of true/false negatives and positives, and plots ROC curves.

  - **Functions.py**  
  Contains various functions used as cost functions, activation functions by the neural networks. Also contains function to evaluate the preformance of classifiers.

  - **LogisticRegression.py**  
  Contains a class for logistic regression.

  - **NeuralNet.py**  
  Contains a class for neural networks.

  - **NNregressor.py**  
  Inherits the Project1 class in ../Project1/project1.py, with tools for analyzing the Franke Function. This is used for the regression case of our FFNN, and some other methods are added specific to this case, including an optimal hyperparameter search, and bias variance plotting for the different hyperparamters. Various functions in if __name__ == "__main__" calls the class functionalities and correspond to the results we use in the report.

  - **test_functions.py**  
  Contains Python test functions.

We also have the folders 'Classificationfigs' and 'Frankefigs' with the plots created by cancer.py, creditcard.py and NNregressor.py
