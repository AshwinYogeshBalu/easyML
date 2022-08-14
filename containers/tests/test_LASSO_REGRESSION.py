import unittest
from containers.LASSO_REGRESSION import LassoRegression 
import pandas as pd
import numpy as np


def read_data():
    df = pd.read_csv("./datasets/mtcars.csv")
    df = df[["cyl", "wt", "disp", "hp", "mpg" ]] 
    X = df.iloc[:, :4].values
    Y = df.iloc[:, -1].values
    return X,Y

def weights():
    X,Y = read_data()
    x = LassoRegression(X, Y, alpha = 1, num_iters = 3000, learning_rate=0.001)
    X_train,Y_train, X_test, Y_test = x.split(X, Y)
    X_train_s, X_test_s, mean, std = x.standardize(X_train, X_test)
    theta = x.normal_equation_lasso_regression(X_train_s, Y_train, 1)
    intercept, weights = x.subgradient_descent(X_train_s, Y_train, theta, 3000, .001)
    w = intercept + weights
    return w

def normalize():
    X,Y = read_data()
    x = LassoRegression(X, Y, alpha = 1, num_iters = 3000, learning_rate=0.001)
    X_train,Y_train, X_test, Y_test = x.split(X, Y)
    X_train_s, X_test_s, mean, std = x.standardize(X_train, X_test)
    return mean, std


def predict(test_x1):
    X,Y = read_data()
    x = LassoRegression(X, Y, alpha = 1, num_iters = 3000, learning_rate=0.001)
    X_train,Y_train, X_test, Y_test = x.split(X, Y)
    X_train_s, X_test_s, mean, std = x.standardize(X_train, X_test)
    theta = x.normal_equation_lasso_regression(X_train_s, Y_train, 1)
    intercept, weights = x.subgradient_descent(X_train_s, Y_train, theta, 3000, .001)
    w = intercept + weights
    x.predict(X_test_s, weights, intercept)
    X1 = (test_x1 - mean)/std
    value = X1.T.dot(weights) + intercept
    return value

class TestLassoRegression(unittest.TestCase):
    def test_weights(self):
        model_weights = weights()
        example_weights = np.array([18.03111543, 16.75514727, 19.48161498, 18.36069128])
        np.testing.assert_array_almost_equal(model_weights,example_weights,decimal=6)

    
    def test_normalize(self):
        model_mean,model_std = normalize()
        mean = np.array([6.18181818,   3.36154545, 232.57727273, 135.04545455])
        std = np.array([1.6958871 ,   1.00372926, 123.48320435,  56.43779945]) 
        np.testing.assert_array_almost_equal(model_mean,mean,decimal=6)
        np.testing.assert_array_almost_equal(model_std,std,decimal=6)


    def test_predict(self):
        test_x1 = np.array([  8.   ,   3.52 , 318.   , 150.   ]) 
        model_prediction = predict(test_x1)
        actual = 17.197381462978285
        np.testing.assert_array_almost_equal(model_prediction,actual,decimal=6)



