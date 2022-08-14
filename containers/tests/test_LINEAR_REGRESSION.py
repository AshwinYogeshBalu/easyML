import unittest
from containers.LINEAR_REGRESSION import LinearRegression
import pandas as pd
import numpy as np


def read_data():
	data = pd.read_excel("./datasets/ENB2012_data.xlsx",index=0)
	return data

def weights():
	data = read_data()
	x = LinearRegression(data.values[:, 0:-1], data.values[:, -1],learningRate=0.0004, tolerance=0.005)
	x.runModel()
	return x.w

def normalize():
	data = read_data()
	x = LinearRegression(data.values[:, 0:-1], data.values[:, -1],learningRate=0.0004, tolerance=0.005)
	x.runModel()
	return x.mean,x.std

def predict(test_x1):
	data = read_data()
	x = LinearRegression(data.values[:, 0:-1], data.values[:, -1],learningRate=0.0004, tolerance=0.005)
	x.runModel()
	X1 = (test_x1 - x.mean)/x.std
	X1 = np.insert(X1, 0, 1)
	value = X1.T.dot(x.w)
	return value


class TestLinearRegression(unittest.TestCase):
	def test_weights(self):
		model_weights = weights()
		example_weights = np.array([22.32891971,  0.11659162, -1.17092647,  2.91366333, -2.54247066,4.32069222, -0.1271735 ,  2.58302229,  0.27069972])
		np.testing.assert_array_almost_equal(model_weights,example_weights,decimal=6)

	
	def test_normalize(self):
		model_mean,model_std = normalize()
		mean = np.array([7.62342007e-01, 6.73340149e+02, 3.19274164e+02, 1.77032993e+02,5.23698885e+00, 3.46840149e+00, 2.38011152e-01, 2.81970260e+00])
		std = np.array([ 0.10512383, 88.09791731, 43.09352263, 44.97913295,  1.74995163,1.10420186,  0.13382541,  1.52884967]) 
		np.testing.assert_array_almost_equal(model_mean,mean,decimal=6)
		np.testing.assert_array_almost_equal(model_std,std,decimal=6)


	def test_predict(self):
		test_x1 = np.array([-0.97353762,  0.978001  , -0.01796473,  0.96638162, -0.99259249, -0.42419915,  1.21044906,  0.1179301 ])
		model_prediction = predict(test_x1)
		actual = 21.06260843216776
		np.testing.assert_array_almost_equal(model_prediction,actual,decimal=6)




