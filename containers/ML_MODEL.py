from containers.LINEAR_REGRESSION import *
from containers.LOGISTIC_REGRESSION import *
from containers.LASSO_REGRESSION import *
from containers.DECISION_TREE import *
import pandas as pd


class ML():
	"""
	ML class is used to fit the data into the models that it could fit in and evaluate the different models performance

	Methods:
		__init__()
		MODEL()
	"""

	def __init__(self,data):
		"""
		Purpose  :  Initialises the parameters

		Parameter  : 
		data - pandas dataframe with the target variable in the last columns
		"""
		self.data = data


	def MODEL(self):
		"""
		Purpose  :  Run different models that could fit for the data and evaluate each models
		Parameter  : 
		None

		Return:
		None
		"""
		target_variable_unique_count = self.data.iloc[:,-1].nunique()
		if target_variable_unique_count == 2:
			print(' It is a classification problem')
			print('\n\n\n------------------------------------------------------------------------------------')
			print('------------------------------------------------------------------------------------')
			print('************************RUNNING LOGISTIC REGRESSION MODEL***************************')
			print('------------------------------------------------------------------------------------')
			print('------------------------------------------------------------------------------------\n')

			lr = LogisticRegression(self.data.iloc[:,0:-1],self.data.iloc[:,-1],learningRate=0.1, maxIteration=200000)
			lr.run_model()

			print('\n\n\n------------------------------------------------------------------------------------')
			print('------------------------------------------------------------------------------------')
			print('************************RUNNING DECISION TREE MODEL*********************************')
			print('------------------------------------------------------------------------------------')
			print('------------------------------------------------------------------------------------\n')

			dt = DecisionTree(max_depth=100, min_samples_split=20)
			dt.run_model(self.data.iloc[:,0:-1],self.data.iloc[:,-1])

		else:
			print('It is a prediction problem')
			print('\n\n\n------------------------------------------------------------------------------------')
			print('------------------------------------------------------------------------------------')
			print('********************************RUNNING MLR MODEL***********************************')
			print('------------------------------------------------------------------------------------')
			print('------------------------------------------------------------------------------------\n')


			GD = LinearRegression(self.data.values[:, 0:-1], self.data.values[:, -1],
				learningRate=0.0004, tolerance=0.005)
			GD.runModel()


			print('\n\n\n------------------------------------------------------------------------------------')
			print('------------------------------------------------------------------------------------')
			print('************************RUNNING LASSO REGRESSION MODEL******************************')
			print('------------------------------------------------------------------------------------')
			print('------------------------------------------------------------------------------------\n')


			Lasso = LassoRegression(X = self.data.values[:, 0:-1], Y = self.data.values[:, -1], alpha = 1, 
				num_iters = 3000, learning_rate = .001)
			Lasso.runModel()
