# easyML
This is a python package developed for the course DS5010


## Summary :
The overall purpose of the package is to give a headstart for user to model a business problem. For eg: If a user has a dataset on which prediction has to be made, the user can use easyML to do predictions. The same applies for a classification problem too. In case of prediction problem, easyML would model the data using Multiple Linear Regression and Lasso Regression. On the other hand for a classification problem, easyML models using Logistic Regression and Decision Tree. We have trained easyML to achieve the best possible result for any dataset and the output gives user an idea about the best model.

## Design :
Our package has 5 modules. One for each of the 4 machine learning models that we have built and the 5th module combines each ML models together with the best hyperparameters. So, the user has the flexibility of running the desired model by calling the respective modules for machine learning models, or can just run the 5th module to model the data using different ML models with the best hyperparameters. The prediction models has an assumption that the data passed to it is numeric. They are evaluated using MSE,RMSE,R2 on both training and test sets. The classification models has an assumption that the target variable has only two classes i.e binary classification problem. They are evaluated using accuracy, precision, recall, specificity, negative predictive value, f1 score, confusion matrix on both training and test sets.

#### 1.	LINEAR_REGRESSION.py
This module implements the Linear Regression algorithm. It is a classic ML algorithm, which estimates the relationship between a quantitative dependent variable and two or more independent variables using a straight line. The weights for Linear Regression are obtained by using gradient descent approach. It has a class “LinearRegression”, which requires the following as mandatory parameters.
•	X – set of features
•	y – target variable
•	learningRate – parameter required for gradient descent algorithm
•	tolerance – parameter for stopping gradient descent algorithm
The class also has default parameters like,
•	maxIteration=50000  –  maximum allowable iterations for model to converge
•	erro r='rmse' – evaluation criteria for the gradient descent algorithm 
•	lambda_Val = 0.0001 – parameter required for cost derivative of the model
The model takes care of scaling the features. The runModel() function does everything for you. 

#### 2.	LOGISTIC_REGRESSION.py 
This module implements a special type of Generalized Linear model called Logistic Regression which predicts the categorical target variable. This module also uses gradient descent approach. It has a class “LogisticRegression”, which requires the following as mandatory parameters.
•	X – set of features
•	y – target variable
The class also has default parameters like, 
•	learningRate = 0.01, Learning rate of the gradient descent algorithm
•	maxIteration = 100, Maximum No. of iterations for the gradient descent
•	fitIntercept - boolean which indicates if base X0 feature vector is added or not
 The run_model () function does everything for you. 

#### 3.	LASSO_REGRESSION.py
Much like Linear Regression, Lasso Regression uses weights to find a linear relationship between the input variable and the output variable. Instead this model uses a sub gradient algorithm to help find the weights. It has a class “LassoRegression”, which requires the following as mandatory parameters.
•	X – set of features
•	Y – target variable
•	alpha - the parameter which balances the amount of emphasis given to minimizing RSS vs minimizing sum of square of coefficients
•	num_iters – Number of times the sub gradient algorithm loops for to find the weights
•	learningRate – parameter required for gradient descent algorithm
The runModel() function combines all the earlier functions and executes them so the user only has to write one line of code does everything for you. 

#### 4.	DECISION_TREE.py
This module implements the classification algorithm called Decision Trees.They are a non-parametric supervised learning method used for classification. The model predicts the value of a target variable by learning simple decision rules inferred from the data features.
The feature importance and tree splits for decision tree are obtained by using Iterative Dichotomiser 3 approach. It has a class “DecisionTree”, which requires no mandatory parameters. But the class has default parameters like,
•	min_sample_split = 20 – parameter required for number of trees split
•	max_depth = 100 – parameter for stopping decision tree algorithm
•	n_feats = None (all the features) – parameter for number of features to be used for building the trees
The features passed to the model can either be numeric or categorical. The model uses Entropy and Information gain to determine the Parent and child node. The run_model () function does everything for you. 

#### 5.	ML_MODEL.py
In this module, we have integrated all the other 4 machine learning modules together. It has a class “ML” which requires a mandatory parameter 
•	data – pandas dataframe with the target variable in the last column
The MODEL () function does everything to you. It identifies the type of problem and runs all the algorithms present for that type of problem with the best hyperparameters. It also prints the result of all the models that ran.


## Usage
The user has the flexibility of choosing which algorithm to run. For eg: Let’s assume, we have to model classification problem of breast cancer. The user just uses the DECISION_TREE module  from the containers folder and the run_model method models the data. Similarly, the user can utilize the individual modules separately, if they are clear with the model of their choice. The images below shows the usage of all the 4 machine learning modules used separately separately. However, if the user just wants to run all the models that could fit for their dataset, they could use the ML_MODEL module from the containers folder. The image below shows the usage.

## Discussion :
This package is based off the already existing python library Scikit-Learn. With Data Science becoming an up-and-coming field, we wanted to make a package that will help the user test and find predictions very easily. Though we have achieved our objective of making things easy for the user, we have provided little flexibility to the user by assuming many things. The data is assumed to be numeric for prediction problems. No encoding is taken care of within the code. Another assumption is that, the target variable has to be in the last column of the data frame when using ML class. Also, we have built classification models only for binary classification. All these assumptions and limitations can be improved as a future work. In addition to that, we are planning to elevate our package by adding more modules for different machine learning algorithms.


