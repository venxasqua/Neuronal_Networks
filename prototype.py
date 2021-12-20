import numpy as np

"""
We will build a prototype of neuronal network with 2 layers
where the first layer have the same numbers of neurons as the number of features in the data set
and the secondlayer/output layer will only have one neurons
Since this is a prototype I willl only use the logistic activation function in every neurons.
"""


class Prototype:
    def __init__(self,learning_rate,steps):
        self.learning_rate : int = learning_rate
        self.steps : int = steps
    def fit(self,X_train,Y_train):
        X_train = X_train.T
        Y_train = Y_train.T
        np.random.seed(42)
        W = [0,0]
        b = [0,0]
        Z = [0,0]
        sigmoid = [0,0]
        W[0] = np.random.randn(X_train.shape[0],X_train.shape[0])
        W[1] = np.random.randn(1,X_train.shape[0])
        b[0] = np.zeros(X_train.shape[0],1)
        b[1] = np.zeros(1,1)
        layer =[0,0]
        """ First compute the forward propagation"""
        #first layer
        Z[0] = W[0]*X_train+b[0]
        sigmoid[0] = 1/(1+np.exp(-Z[0]))
        #second layer
        Z[1] = W[1]*sigmoid[0]+b[1]
        sigmoid[1] = 1(1+np.exp(-Z[1]))
        cost_function = -np.mean(Y_train*np.log(sigmoid[1]+(1-Y_train)*np.log(1-sigmoid[1])))
        """ now compute the gradient using backward propagation"""
        
    def predict(self):
        pass