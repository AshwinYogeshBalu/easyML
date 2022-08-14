from containers.LOGISTIC_REGRESSION import LogisticRegression
import pandas as pd
import numpy as np
import unittest


def read_data():
	data = pd.read_csv('./datasets/breast-cancer.csv')
	return data

def weights():
	data = read_data()
	lr = LogisticRegression(data.iloc[:,0:-1],data.iloc[:,-1],learningRate=0.1, maxIteration=200000)
	lr.run_model()
	return lr.weights


def predict(test_x1):
	data = read_data()
	lr = LogisticRegression(data.iloc[:,0:-1],data.iloc[:,-1],learningRate=0.1, maxIteration=200000)
	lr.run_model()
	test_x1 = np.insert(test_x1, 0, 1)
	value = int(lr.sigmoid(np.dot(test_x1,lr.weights)))
	return value


class TestLogisticRegression(unittest.TestCase):
	def test_weights(self):
		model_weights = weights()
		example_weights = np.array([ -69.89886784, -459.39060111,  -78.32344236, -879.16253398,
			16.75694692,   11.22702794,   60.58202372,   85.03184055,
			34.47790854,   22.15187838,    3.32035941,  -13.99199483,
			-81.5075801 ,  111.72394505,  124.30864843,    1.05297932,
			13.73652759,   18.55951365,    4.40049136,    4.51536553,
			1.05045651, -500.45421237,  378.55450341,  403.06989027,
			41.42241219,   21.9114728 ,  207.48535264,  253.57139102,
			71.11143245,   64.41588109,   19.42777716])
		for i in range(len(model_weights)):
			assert abs(model_weights[i]) - abs(example_weights[i]) <= 1


	def test_predict(self):
		test_x1  =  np.array([13.61, 24.98, 88.05, 582.7, 0.09487999999999999, 0.08511, 0.08625,
			0.04489, 0.1609, 0.058710000000000005, 0.4565, 1.29, 2.861, 43.14,
			0.005872, 0.01488, 0.02647, 0.009921, 0.01465, 0.002355, 16.99,
			35.27, 108.6, 906.5, 0.1265, 0.1943, 0.3169, 0.1184, 0.2651,
			0.07397000000000001])
		test_y1 = 0
		model_prediction = predict(test_x1)
		assert model_prediction == test_y1

