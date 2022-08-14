import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from random import seed
from random import randrange
from random import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



class LinearRegression:
    """
    LinearRegression class is used to fit and evaluate Linear regression algorithm to the given data
    Methods:
        __init__(self, min_samples_split, max_depth, n_feats)
        splitToTrainTest()
        add_x0()
        normalize()
        normalizeTestData()
        rank()
        checkMatrix()
        checkInvertibility()
        gradientDescent()
        predict()
        sse()
        mse()
        rmse()
        r2()
        costFunction()
        costDerivative()
        runModel()
    """

    def __init__(self, X, y, learningRate, tolerance, maxIteration=50000, error='rmse', lambda_Val = 0.0001):
        """
        Purpose  :  Initialises the parameters
        Parameter  : 
        X - Features
        y - Target variable
        learningRate - learning rate of the algorithm
        tolerance - parameter for stopping gradient descent algorithm
        maxIteration - Maximum No. of iterations for the gradient descent
        error - error criteria to evaluate gradient descent
        lambda_Val - adjusting parameter for costderivative function
        """
        self.X = X
        self.y = y
        self.learningRate = learningRate
        self.tolerance = tolerance
        self.maxIteration = maxIteration
        self.error = error
        self.lambda_Val = lambda_Val
        
        
    def splitToTrainTest(self):
        """
        Purpose :  Function which is used to implement the train test split of the input data. 
        It performs like train_test_split of sklearn

        Parameters
        ----------
        x - features data
        y - target column data

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        np.random.seed(0)                  
        indices = np.random.permutation(len(self.X))       
        data_test_size = int(len(self.X) * 0.3)
        train_indices = indices[data_test_size:]
        test_indices = indices[:data_test_size]
        X_train = self.X[train_indices]
        y_train = self.y[train_indices]
        X_test = self.X[test_indices]
        y_test = self.y[test_indices]
        return X_train, X_test, y_train, y_test
        
    
    def add_x0(self, X):
        """
        Purpose : Creates X0 feature vector of dimension M x 1 and concatenates X0 feature vector with the feature making intercept
        """
        return np.column_stack([np.ones([X.shape[0], 1]),X])
    
    def normalize(self, X):
        """
        Purpose :  Normalizes the features of training data

        Parameters
        ----------
        X - features in training set
        
        Returns
        -------
        X_norm - normalized features
        mean - mean of each columns
        std - standard deviation of each columns
        """
        mean= np.mean(X, 0)
        std= np.std(X,0)
        X_norm = (X-mean)/std
        X_norm = self.add_x0(X_norm)
        
        return X_norm, mean, std
    
    def normalizeTestData(self, X_test, train_mean, train_std):
        """
        Purpose :  Normalizes the features of test data based on mean,std from training data

        Parameters
        ----------
        X_test - features in test set
        train_mean - mean of each columns from training data
        train_std - standard deviation of each columns from training data
        
        Returns
        -------
        X_norm - normalized features of test data
        """
        X_norm = (X_test-train_mean)/train_std
        X_norm = self.add_x0(X_norm)
        return X_norm
    
    def rank(self, X, eps = 1e-12):
        u, s, vh = np.linalg.svd(X)
        return len([x for x in s if abs(x) > eps])
    
    def checkMatrix(self, X):
        X_rank = np.linalg.matrix_rank(X)
        if X_rank == min(X.shape[0],X.shape[1]):
            self.fullRank = True
        else:
            self.fullRank = False
            
            
    def checkInvertibility(self, X):
        if X.shape[0] < X.shape[1]:
            self.lowRank = True
        else:
            self.lowRank = False            
            
    def gradientDescent(self, X, y):
        """
        determines the weights by running gradient descent approach

        Parameters
        ----------
        X : all the features
        y : target values (actual)

        Returns
        -------
        None
        """

        self.error_sequence = []
        last = float('inf')
        for i in tqdm(range(self.maxIteration)):
            self.w = self.w - self.learningRate * self.costDerivative(X,y)
            if self.error == 'rmse':
                cur = self.rmse(X, y)
            else:
                cur = self.sse(X,y)
            diff = last - cur
            last= cur
            self.error_sequence.append(cur)
            
            if diff<self.tolerance:
                break 
    
    def predict(self, X):
        """
        Predicts the target value based on the input features

        Parameters
        ----------
        X : all the features

        Returns
        -------
        The predicted value.
        """
        return X.dot(self.w)
    
    def sse(self, X, y):
        """
        Parameters
        ----------
        X : all the features
        Y : target variable
        weights : weights for each feature
        intercept : the bias

        Returns
        -------
        The sum of squared error
        """
        y_hat = self.predict(X)
        return ((y_hat-y)**2).sum()
    
    def mse(self,X,y):
        """
        Parameters
        ----------
        X : all the features
        Y : target variable
        weights : weights for each feature
        intercept : the bias

        Returns
        -------
        The mean squared error
        """
        y_hat = self.predict(X)
        return ((y_hat-y)**2).mean()
        
    def rmse(self, X, y):
        """
        Parameters
        ----------
        X : all the features
        Y : target variable
        weights : weights for each feature
        intercept : the bias

        Returns
        -------
        The root mean squared error
        """
        y_hat = self.predict(X)
        return np.sqrt(self.sse(X, y) / X.shape[0])
    
    def r2(self,X,y):
        """
        Parameters
        ----------
        X : all the features
        Y : target variable
        weights : weights for each feature
        intercept : the bias

        Returns
        -------
        The r squared value
        """
        actual = y
        prediction = self.predict(X)
        ssr = sum((actual - prediction) ** 2) 
        sst = sum((actual - actual.mean()) ** 2) 
        return 1 - ssr/sst
    
    def costFunction(self, X, y):
        return self.sse(X,y)/2
    
    def costDerivative(self,X,y):
        y_hat = self.predict(X)
        gradient = (y_hat - y).dot(X) + self.lambda_Val        
        return gradient
    
    
    def runModel(self):
        """
        Purpose :  Function for running the model using train and test data and for comparing the results
        Parameters : None
        return : None
        """
        self.X_train, self.X_test, self.y_train, self.y_test = self.splitToTrainTest()
        self.X_train, self.mean, self.std = self.normalize(self.X_train)
        self.X_test = self.normalizeTestData(self.X_test,self.mean,self.std)
        #self.checkMatrix(self.X_train)
        #self.checkInvertibility(self.X_train)
        
      
        print("\n*******************On solving using gradient descent*******************\n")
        self.w = np.ones(self.X_train.shape[1],dtype=np.float64) * 0
        self.gradientDescent(self.X_train, self.y_train)
        #self.w = self.gradientDescent(self.w,self.X_train, self.y_train)
            
        print('The weights are as follows:')
        print(self.w)
        
        print('\n####################MSE##########################')
        print('Training data: {}'.format(self.mse(self.X_train,self.y_train)))
        print('Testing data: {}'.format(self.mse(self.X_test,self.y_test)))
        
        print('\n####################RMSE##########################')
        print('Training data: {}'.format(self.rmse(self.X_train,self.y_train)))
        print('Testing data: {}'.format(self.rmse(self.X_test,self.y_test)))
        
        print('\n####################R2##########################')
        print('Training data: {}'.format(self.r2(self.X_train,self.y_train)))
        print('Testing data: {}'.format(self.r2(self.X_test,self.y_test)))
     
