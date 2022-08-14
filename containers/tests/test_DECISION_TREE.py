from containers.DECISION_TREE import DecisionTree
import pandas as pd
import numpy as np
import unittest


def read_data():
	data = pd.read_csv('./datasets/breast-cancer.csv')
	return data


def predict(test):
	data = read_data()
	clf = DecisionTree(max_depth=100)
	clf.run_model(data.iloc[:,0:-1],data.iloc[:,-1])
	value = clf.predict(test)
	return value


class TestDecisionTree(unittest.TestCase):
	def test_predict(self):
		test_x1 = np.array([1.340e+01, 2.052e+01, 8.864e+01, 5.567e+02, 1.106e-01, 1.469e-01,
			1.445e-01, 8.172e-02, 2.116e-01, 7.325e-02, 3.906e-01, 9.306e-01,
			3.093e+00, 3.367e+01, 5.414e-03, 2.265e-02, 3.452e-02, 1.334e-02,
			1.705e-02, 4.005e-03, 1.641e+01, 2.966e+01, 1.133e+02, 8.444e+02,
			1.574e-01, 3.856e-01, 5.106e-01, 2.051e-01, 3.585e-01, 1.109e-01])
		test_x2=np.array([1.321e+01, 2.525e+01, 8.410e+01, 5.379e+02, 8.791e-02, 5.205e-02,
			2.772e-02, 2.068e-02, 1.619e-01, 5.584e-02, 2.084e-01, 1.350e+00,
			1.314e+00, 1.758e+01, 5.768e-03, 8.082e-03, 1.510e-02, 6.451e-03,
			1.347e-02, 1.828e-03, 1.435e+01, 3.423e+01, 9.129e+01, 6.329e+02,
			1.289e-01, 1.063e-01, 1.390e-01, 6.005e-02, 2.444e-01, 6.788e-02])
		arr = np.stack((test_x1, test_x1), axis=0)
		test_y1 = np.array([1,1])
		model_prediction = predict(arr)
		for i in range(len(test_y1)):
			assert test_y1[i] == model_prediction[i]



