import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class LogisticRegression:
    """
    LogisticRegression class is used to fit and evaluate logistic regression algorithm to the given data
 
    Methods:
        __init__(self, min_samples_split, max_depth, n_feats)
        splitToTrainTest()
        add_x0()
        sigmoid()
        costFunction()
        train()
        predict_prob()
        predict()
        accuracy()
        model_eval()
        splitToTrainTest()
        run_model()
    """

    def __init__(self, X, y, learningRate = 0.01, maxIteration = 100,fitIntercept =True, flag = False) :
        """
        Purpose  :  Initialises the parameters
        Parameter  : 
        X - Features
        y - Target variable
        learningRate - learning rate of the algorithm
        maxIteration - Maximum No. of iterations for the gradient descent
        fitIntercept - boolean which indicates if base X0 feature vector is added or not
        """
        self.X = X
        self.y = y
        self.learningRate = learningRate
        self.maxIteration = maxIteration
        self.fitIntercept = fitIntercept
        self.flag = flag
        
   
    def addX0(self, X):
        """
        Purpose : Creates X0 feature vector of dimension M x 1 and concatenates X0 feature vector with the feature making intercept
        """
        return np.column_stack([np.ones([X.shape[0], 1]), X])
        
    def sigmoid(self, z):
        """
        Purpose : Defines the Logit function, based on which we make predictions
        Parameter :  z - product of the features with weights
        Returns : sig - which is the probability of the attachment to the class
        """
        sig = 1 / (1+np.exp(-z))
        return sig
        
    def costFunction(self, X, y):
        """
        Purpose : 
        Maximum Likelihood Estimation is a way to finding the best possible parameters which make the observed data most probable. 
        This is done by finding parameters θ that maximize the likelihood function.
        """
        pred = -y * np.log(h) - (1 - y) * np.log(1 - h) 
        cost = pred.mean()
        return cost

        
    def train(self, X, y):
        """
        Purpose : Function for training the algorithm. We are using Gradient descent algorithm.
        Parameters : 
        X - Features
        y - Target variable (1/0)
        return : none
        """
        if self.fitIntercept:
            X = self.addX0(X)  

        self.weights = np.zeros(X.shape[1])  
        
        for i in range(self.maxIteration):  
            z = np.dot(X, self.weights)  
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.weights -= self.learningRate * gradient
            
            if (self.flag == True and i % 10000 == 0):
                z = np.dot(X, self.weights)
                h = self.sigmoid(z)
                print(f'loss: {self.costFunction(h, y)} \t')

    def predict_prob(self, X):  
        if self.fitIntercept:
            X = self.addX0(X)
        return self.sigmoid(np.dot(X, self.weights))
    
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
    
    def accuracy(self,y,y_hat):
        """
        Purpose : Accuracy of the model
        Parameters : 
        y - predicted values
        y_hat - actual values
        return : accuracy value
        """
        acc = (y == y_hat).mean()
        return acc
    
    def model_eval(self,pred,original): 
        """
        Purpose : evaluate the model based on various etrics
        Parameters : 
        pred - predicted values
        original - actual values
        return : precision,recall,specificity,negative_pred_value,f1,matrix
        """        
        matrix=np.zeros((2,2))
        for i in range(len(pred)):
            if int(pred[i])==1 and int(original[i])==1: 
                matrix[0,0]+=1 #True Positives
            elif int(pred[i])==0 and int(original[i])==1:
                matrix[0,1]+=1 #False Negatives
            elif int(pred[i])==1 and int(original[i])==0:
                matrix[1,0]+=1 #False Positives
            elif int(pred[i])==0 and int(original[i])==0:
                matrix[1,1]+=1 #True Negatives
    
        precision=matrix[0,0]/(matrix[0,0]+matrix[0,1])
        recall=matrix[0,0]/(matrix[0,0]+matrix[1,0])
        specificity=matrix[1,1]/(matrix[0,1]+matrix[1,1])
        negative_pred_value=matrix[1,1]/(matrix[1,0]+matrix[1,1])
        f1=2*(precision*recall)/(precision+recall)
        return precision,recall,specificity,negative_pred_value,f1,matrix
    
    def splitToTrainTest(self,x,y):
        """
        Purpose :  Function which is used to implement the train test split of the input data. 
        It performs like train_test_split of sklearn
        Parameters : 
        x - features data
        y - target column data
        return : X_train, X_test, y_train, y_test
        """
        np.random.seed(0)                  
        indices = np.random.permutation(len(x))       
        data_test_size = int(len(x) * 0.2)
        train_indices = indices[data_test_size:]
        test_indices = indices[:data_test_size]
        X_train = x[train_indices]
        y_train = y[train_indices]
        X_test = x[test_indices]
        y_test = y[test_indices]
        return X_train, X_test, y_train, y_test
    
    def run_model(self):
        """
        Purpose :  Function for running the model using train and test data and for comparing the results
        Parameters : None
        return : None
        """
        self.X = np.array(self.X)
        dummies = pd.get_dummies(self.y)
        dummies.drop(dummies.columns[-1],axis=1,inplace=True)
        value = dummies.columns
        self.y = np.array(dummies.iloc[:,0])
        print('The target class {} is encoded as 1 and the other class is encoded as 0 \n'.format(value[0]))
        
        X_train, X_test, y_train, y_test = self.splitToTrainTest(self.X, self.y)
        self.train(X_train, y_train)
        
        #On testing data
        y_pred = [int(round(x)) for x in self.predict_prob(X_test).flatten()]
        test_accuracy = self.accuracy(y_pred,y_test)
        test_precision,test_recall,test_specificity,test_negative_pred_value,test_f1,test_matrix = self.model_eval(y_pred,y_test)
        #On training data
        y_pred = [int(round(x)) for x in self.predict_prob(X_train).flatten()]
        train_accuracy = self.accuracy(y_pred,y_train)
        train_precision,train_recall,train_specificity,train_negative_pred_value,train_f1,train_matrix = self.model_eval(y_pred,y_train)
        
        print('\n####################ACCURACY##########################')
        print('Training data: {}'.format(train_accuracy))
        print('Testing data: {}'.format(test_accuracy))
        print('\n####################PRECISION##########################')
        print('Training data: {}'.format(train_precision))
        print('Testing data: {}'.format(test_precision))
        print('\n####################RECALL##########################')
        print('Training data: {}'.format(train_recall))
        print('Testing data: {}'.format(test_recall))
        print('\n####################SPECIFICITY##########################')
        print('Training data: {}'.format(train_specificity))
        print('Testing data: {}'.format(test_specificity))
        print('\n############NEGATIVE PREDICTIVE VALUE####################')
        print('Training data: {}'.format(train_negative_pred_value))
        print('Testing data: {}'.format(test_negative_pred_value))
        print('\n######################F1 SCORE##########################')
        print('Training data: {}'.format(train_f1))
        print('Testing data: {}'.format(test_f1))
        print('\n###############CONFUSION MATRIX##########################')
        print('Training data:\n{}'.format(train_matrix))
        print('\nTesting data:\n{}'.format(test_matrix))
        
        

               