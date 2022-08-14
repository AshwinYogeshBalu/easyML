import numpy as np
import pandas as pd
from collections import Counter


def entropy(y):
    hist = np.bincount(y)
    ps = hist/len(y)
    return(-np.sum([p * np.log2(p) for p in ps if p>0]))


class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
  
    def is_leaf_node(self):
        return(self.value is not None)


class DecisionTree:
    """
    DecisionTree class is used to fit and evaluate decision tree algorithm to the given data
 
    Methods:
        __init__(self, min_samples_split, max_depth, n_feats)
        fit()
        _grow_tree()
        _best_criteria()
        _best_criteria()
        _information_gain()
        _split()
        predict()
        _traverse_tree()
        _most_common_label()
        accuracy()
        model_eval()
        splitToTrainTest()
        run_model()
    """
    def __init__(self, min_samples_split=20, max_depth=100, n_feats=None):
        """
        DecisionTree Class Constructor to initialize the object.
 
        Input Arguments: 
        min_samples_split must be int, 
        max_depth must be int, 
        n_feats must be int
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        """This method returns number of features to be used for growing the tree"""
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """This method returns best features and threshold to grow the tree"""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        #stopping criteria
        if(depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return(Node(value=leaf_value))
    
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        #greedy search
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        left_idxs, right_idxs = self._split(X[:,best_feat],best_thresh)

        left = self._grow_tree(X[left_idxs,:], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs,:], y[right_idxs], depth+1)
        return(Node(best_feat, best_thresh, left, right))
  

    def _best_criteria(self, X, y, feat_idxs):
        """This method returns split index and threshold"""
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
        for threshold in thresholds:
            gain = self._information_gain(y, X_column, threshold)
            if(gain>best_gain):
                best_gain = gain
                split_idx = feat_idx
                split_thresh = threshold
        return(split_idx, split_thresh)

    def _information_gain(self, y, X_column, split_threh):
        """This method returns information gain"""
        #parent entropy
        parent_entropy = entropy(y)

        #generate split
        left_idxs, right_idxs = self._split(X_column, split_threh)
        if(len(left_idxs == 0) or len(right_idxs)==0):
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r

        ig = parent_entropy - child_entropy
        return ig
  
    def _split(self, X_column, split_threh):
        left_idxs = np.argwhere(X_column <= split_threh).flatten()
        right_idxs = np.argwhere(X_column > split_threh).flatten()
        return(left_idxs, right_idxs)
  
    def predict(self, X):
        '''returns prediction'''
        #traverse tree
        return(np.array([self._traverse_tree(x, self.root) for x in X]))

    def _traverse_tree(self, x, node):
        """This method traverse through the trees and returns right or left node."""
        if(node.is_leaf_node()):
            return(node.value)

        if(x[node.feature] <= node.threshold):
            return(self._traverse_tree(x, node.left))
        return(self._traverse_tree(x, node.right))

    def _most_common_label(self, y):
        """This method returns most common label."""
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return(most_common)
  
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

    def run_model(self, X, y): 
        """
        Purpose :  Function for running the model using train and test data and for comparing the results
        Parameters : None
        return : None
        """
        X = np.array(X)
        dummies = pd.get_dummies(y)
        dummies.drop(dummies.columns[-1],axis=1,inplace=True)
        value = dummies.columns
        y = np.array(dummies.iloc[:,0])
        print('The target class {} is encoded as 1 and the other class is encoded as 0 \n'.format(value[0]))
        X_train, X_test, y_train, y_test = self.splitToTrainTest(X, y)
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        x_pred = self.predict(X_train)
  
        #On testing data
        test_accuracy = self.accuracy(y_pred,y_test)
        test_precision,test_recall,test_specificity,test_negative_pred_value,test_f1,test_matrix = self.model_eval(y_pred,y_test)

        #On training data
        train_accuracy = self.accuracy(x_pred,y_train)
        train_precision,train_recall,train_specificity,train_negative_pred_value,train_f1,train_matrix = self.model_eval(x_pred,y_train)
    
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

               

