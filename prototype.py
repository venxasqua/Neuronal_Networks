import numpy as np

"""
We will build a prototype of neuronal network with 2 layers
where the first layer have the same numbers of neurons as the number of features in the data set
and the secondlayer/output layer will only have one neurons
Since this is a prototype I willl only use the logistic activation function in every neurons.
"""


class Prototype:
    def __init__(self,learning_rate=0.01,steps=10000):
        self.learning_rate : int = learning_rate
        self.steps : int = steps
    def fit(self,X_train,Y_train):
        X_train = X_train.T
        Y_train = Y_train.T
        np.random.seed(42)
        self.W = [0,0]
        self.b = [0,0]
        #Z = [0,0]
        dZ = [0,0]
        dW = [0,0]
        db = [0,0]
        sigmoid = [0,0]
        self.W[0] = np.random.randn(X_train.shape[0],X_train.shape[0])
        self.W[1] = np.random.randn(1,X_train.shape[0])
        self.b[0] = np.zeros((X_train.shape[0],1))
        self.b[1] = np.zeros((1,1))
        for i in range(self.steps):
            """ First compute the forward propagation"""

            #first layer
            #Z[0] = self.W[0].dot(X_train) + self.b[0]
            sigmoid[0] = 1/(1 + np.exp(-(self.W[0].dot(X_train) + self.b[0])))

            #second layer
            #Z[1] = self.W[1].dot(sigmoid[0])+self.b[1]
            sigmoid[1] = 1/(1+np.exp(-(self.W[1].dot(sigmoid[0])+self.b[1])))
            

            cost_function = -np.mean(Y_train*np.log(sigmoid[1])+(1-Y_train)*np.log(1-sigmoid[1]))
            """backward propagation"""
            dZ[1] = sigmoid[1] - Y_train
            dW[1] = 1/X_train.shape[0]*dZ[1].dot(sigmoid[0].T)
            db[1] = 1/X_train.shape[0]*np.sum(dZ[1],axis=1,keepdims=True)
            dZ[0] = self.W[1].T.dot(dZ[1])*(sigmoid[0]*(1-sigmoid[0]))
            dW[0] = 1/X_train.shape[0]*dZ[0].dot(X_train.T)
            db[0] = 1/X_train.shape[0]*np.sum(dZ[0],axis=1,keepdims=True)
            
            """update via gradient descent"""
            self.W[0] = self.W[0] - self.learning_rate * dW[0]
            self.b[0] = self.b[0] - self.learning_rate * db[0]
            self.W[1] = self.W[1] - self.learning_rate * dW[1]
            self.b[1] = self.b[1] - self.learning_rate * db[1]
        return (cost_function, sigmoid[1])



    def predict(self,X_test):
        prediction = X_test.T
        for i in range(2):
            prediction = self.W[i].dot(prediction) + self.b[i]
            prediction = 1/(1+np.exp(-prediction))
        return prediction.T
if __name__ == "__main__":
    X_train =np.array([[1],[2],[3]])
    Y_train = np.array([0,1,1]).reshape((3,1))
    model = Prototype()
    print(model.fit(X_train,Y_train))