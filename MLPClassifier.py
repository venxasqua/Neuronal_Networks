import numpy as np
import pandas as pd
class MLPClassifier:
    def __init__(self,hidden_layer_size,alpha=0.0001,learning_rate=0.11,max_iter=20000, random_state=42):
        self.hidden_layer_size : tuple = hidden_layer_size
        alpha : float = alpha
        self.learning_rate: float = learning_rate
        self.max_iter : int = max_iter
        self.random_state = random_state
    def fit(self,X_train : np.array, Y_train : np.array):
        """ some preparation"""
        np.random.seed(self.random_state)
        X_train = X_train.T  
        Y_train = Y_train.T
        
        self.W = list(range(len(self.hidden_layer_size)+1))
        self.b = list(range(len(self.hidden_layer_size)+1))
        dZ = list(range(len(self.hidden_layer_size)+1))
        dW = list(range(len(self.hidden_layer_size)+1))
        db = list(range(len(self.hidden_layer_size)+1))
        sigmoid = list(range(len(self.hidden_layer_size)+1))

        self.W[0] = np.random.randn(self.hidden_layer_size[0],X_train.shape[0])
        self.b[0] = np.zeros((self.hidden_layer_size[0],1))
        self.W[len(self.hidden_layer_size)] = np.random.randn(Y_train.shape[0],self.hidden_layer_size[len(self.hidden_layer_size)-1])
        self.b[len(self.hidden_layer_size)] = np.zeros((Y_train.shape[0],1))
        for i in range(1,len(self.hidden_layer_size)):
            self.W[i] = np.random.randn(self.hidden_layer_size[i],self.hidden_layer_size[i-1])
            self.b[i] = np.zeros((self.hidden_layer_size[i],1))

        #------------------------------------------------------------------------------------------------------------------------------
        for m in range(self.max_iter):
            """forward propagation"""
            sigmoid[0] = 1/(1 + np.exp(-(self.W[0].dot(X_train) + self.b[0])))
            for j in range(1,len(self.hidden_layer_size)+1):
                sigmoid[j] = 1/(1 + np.exp(-(self.W[j].dot(sigmoid[j-1]) + self.b[j])))

            cost_function = -np.mean(Y_train*np.log(sigmoid[len(self.hidden_layer_size)])+(1-Y_train)*np.log(1-sigmoid[len(self.hidden_layer_size)]))
            """backward propagation"""
            dZ[len(self.hidden_layer_size)] = sigmoid[len(self.hidden_layer_size)] - Y_train
            dW[len(self.hidden_layer_size)] = 1/X_train.shape[0]*dZ[len(self.hidden_layer_size)].dot(sigmoid[len(self.hidden_layer_size)-1].T)
            db[len(self.hidden_layer_size)] = 1/X_train.shape[0]*np.sum(dZ[len(self.hidden_layer_size)],axis=1,keepdims=True)
            for k in range(len(self.hidden_layer_size)-1,0,-1):
                dZ[k] = self.W[k+1].T.dot(dZ[k+1])*(sigmoid[k]*(1-sigmoid[k]))
                dW[k] = 1/X_train.shape[0]*dZ[k].dot(sigmoid[k-1].T)
                db[k] = 1/X_train.shape[0]*np.sum(dZ[k],axis=1,keepdims=True)
            dZ[0] = self.W[1].T.dot(dZ[1])*(sigmoid[0]*(1-sigmoid[0]))
            dW[0] = 1/X_train.shape[0]*dZ[0].dot(X_train.T)
            db[0] = 1/X_train.shape[0]*np.sum(dZ[0],axis=1,keepdims=True)
            """update"""
            #print(dW)
            for l in range(len(self.hidden_layer_size)+1):
                self.W[l] = self.W[l] - self.learning_rate*dW[l]
                self.b[l] = self.b[l] - self.learning_rate*db[l]

        return cost_function,sigmoid[len(self.hidden_layer_size)]



    def predict(X_test : np.array):
        pass


if __name__ == "__main__":
    X_train =np.array([[-100,-10],[0,2],[10,15],[-30,-1],[1,-2],[20,15]])
    Y_train = np.array([0,0,1,0,0,1])
    Y_train=Y_train.reshape((Y_train.shape[0],1))

    model = MLPClassifier((2,))
    print(model.fit(X_train,Y_train))