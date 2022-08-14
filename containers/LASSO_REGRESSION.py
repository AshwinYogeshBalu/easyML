import numpy as np
import pandas as pd
import math


class LassoRegression:
    """
    LassoRegression class is used to fit and evaluate Lasso regression algorithm to the given data
    Methods:
        __init__()
        create_function()
        split()
        add_intercept_ones()
        standardize()
        subgradient_descent()
        normal_equation_lasso_regression()
        predict()
        sse()
        mse()
        rmse()
        r2()
        runModel()
    """
    
    def __init__(self, X, Y, alpha, num_iters, learning_rate):
        """
        Purpose  :  Initialises the parameters
        Parameter  : 
        X - Features
        Y - Target variable
        alpha - the parameter which balances the amount of emphasis given to 
        minimizing RSS vs minimizing sum of square of coefficients
        num_iters - Number of times the sub gradient algorithm loops for to find the weights
        learning_rate - parameter required for gradient descent algorithm
        """
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        
    def create_function(self, X_b, theta):
        '''
        Parameters
        ----------
        X_b : An Array of the 
        theta : TYPE
            DESCRIPTION.

        Returns
        -------
        a function of w1x1 + w2x2+....+ wnxn
        '''
        return np.dot(X_b, theta)
    
    def split(self, X, Y):
        '''
        Parameters
        ----------
        X : the full data frame of features
        Y : The variable we are predicting

        Returns
        -------
        X_train : splits X into a set to train the model
        Y_train : splits Y into a set to train the model
        X_test : splits X into a set to test the model
        Y_test : splits Y into a set to compare it to the predictions with X_test
        '''
        np.random.seed(0)
        train_size = math.floor(int(.7 * X.shape[0]))
        X_train = X[:train_size]
        Y_train = Y[:train_size]
        X_test = X[train_size:]
        Y_test = Y[train_size:]
        
        return X_train, Y_train, X_test, Y_test 
    
    def add_intercept_ones(self,X):
        '''
        Parameters
        ----------
        X : a dataframe of the features

        Returns
        -------
        X_b : adds an intercept term
        '''
        intercept_ones = np.ones((len(X),1))
        X_b = np.c_[intercept_ones,X]
        return X_b
    
    def standardize(self, X_train, X_test):
        '''
        Parameters
        ----------
        X_train : training set of features
        X_test : test set of features

        Returns
        -------
        X_train_s : The standarized version of X_train
        X_test_s : The standarized version of X_test
        mean : The mean of X_train
        std : The std of X_trin
        '''
        mean = np.mean(X_train, 0)
        std = np.std(X_train, 0)
        X_train_s = (X_train - mean) / std 
        X_test_s = (X_test - mean) / std 
        return X_train_s, X_test_s, mean, std
    
    def subgradient_descent(self, X, Y, theta, number_of_iterations, learning_rate):
        '''
        Parameters
        ----------
        X : features data
        Y : target variable data
        theta : the Initial weights that you are starting with, will change and adjust throughout the loop
        number_of_iterations : How many times the for loop runs for
        learning_rate : small positive value, often in the range between 0.0 and 1.0

        Returns
        -------
        intercept : the bias
        weights : weights for each feature
        '''
        X_b = self.add_intercept_ones(X)
        for i in range(self.num_iters):

            
            f = self.create_function(X_b,theta)
            Y_predicted = f
            error = Y-Y_predicted
            elastic_mse = 1/(Y.size) * np.dot(error.T, error) + 1 * np.sum(np.abs(theta))
            

            elastic_gradient = -(2/Y.size) * X_b.T.dot((Y - Y_predicted)) + 1 * np.sign(theta) 
            theta = theta - self.learning_rate * elastic_gradient
            intercept = theta[0]
            weights = theta[1:]
            
        return intercept, weights
    
    def normal_equation_lasso_regression(self, X, Y,alpha):
        '''
        Parameters
        ----------
        X : features data
        Y : target variable data
        alpha : the shrinkage value

        Returns
        -------
        theta_optimal : Identifies optimal initial thetas
        '''
        intercept_ones = np.ones((len(X),1))        
        X_b = np.c_[intercept_ones,X]              
        I = np.identity(X_b.shape[1])              
        I[0][0] = 0                                
        theta_optimal = np.linalg.inv(X_b.T.dot(X_b) + alpha * I).dot(X_b.T).dot(Y) 
        return theta_optimal    
        
    def predict(self, X, weights,intercept):
        '''
        Parameters
        ----------
        X : all the features
        weights : Weights for each X
        intercept : the intercept

        Returns
        -------
        The prediction which is each x times it respective weights then the products added together along with the intercept
        '''
        return X.dot(weights) + intercept
        
    def sse(self, X, Y, weights,intercept):
        '''
        Parameters
        ----------
        X : all the features
        Y : target variable
        weights : weights for each feature
        intercept : the bias

        Returns
        -------
        The sum of squared error
        '''
        Y_hat = self.predict(X, weights,intercept)
        return ((Y_hat-Y)**2).sum()

    def mse(self, X, Y, weights,intercept):
        '''
        Parameters
        ----------
        X : all the features
        Y : target variable
        weights : weights for each feature
        intercept : the bias

        Returns
        -------
        The mean squared error
        '''
        Y_hat = self.predict(X, weights,intercept)
        return ((Y_hat-Y)**2).mean() 

    def rmse(self, X, Y, weights,intercept):
        '''
        Parameters
        ----------
        X : all the features
        Y : target variable
        weights : weights for each feature
        intercept : the bias

        Returns
        -------
        The root mean squared error
        '''
        Y_hat = self.predict(X, weights,intercept)
        return np.sqrt(self.sse(X, Y, weights, intercept) / X.shape[0])

    def r2(self, X,Y, weights,intercept):
        '''
        Parameters
        ----------
        X : all the features
        Y : target variable
        weights : weights for each feature
        intercept : the bias

        Returns
        -------
        The r squared value
        '''
        actual = Y
        prediction = self.predict(X, weights,intercept)
        ssr = sum((actual - prediction) ** 2) 
        sst = sum((actual - actual.mean()) ** 2) 
        return 1 - ssr/sst
    
    def runModel(self):
        '''
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        X_train, Y_train, X_test,  Y_test = self. split(self.X, self.Y)
        X_train_s, X_test_s, mean, std = self.standardize(X_train, X_test)
        theta = self.normal_equation_lasso_regression(X_train_s, Y_train, self.alpha)
        intercept, weights = self.subgradient_descent(X_train_s, Y_train, theta, self.num_iters, self.learning_rate)
        Y_pred = self.predict(X_test_s, weights, intercept)
    
        print('\n####################MSE##########################')
        print('Training data: {}'.format(self.mse(X_train_s,Y_train, weights,intercept)))
        print('Testing data: {}'.format(self.mse(X_test_s,Y_test, weights,intercept)))
    
        print('\n####################RMSE##########################')
        print('Training data: {}'.format(self.rmse(X_train_s,Y_train, weights,intercept)))
        print('Testing data: {}'.format(self.rmse(X_test_s,Y_test, weights,intercept)))
    
        print('\n####################R2##########################')
        print('Training data: {}'.format(self.r2(X_train_s,Y_train, weights,intercept)))
        print('Testing data: {}'.format(self.r2(X_test_s,Y_test, weights,intercept))) 


