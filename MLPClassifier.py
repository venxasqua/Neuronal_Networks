import numpy as np
import pandas as pd
class MLPClassifier:
    def __init__(self,hidden_layer_size,alpha=0.0001,learning_rate=0.1,
                max_iter=2000, random_state=42,momentum = 0.9,use_momentum = True):

        self.hidden_layer_size : tuple = hidden_layer_size
        self.alpha : float = alpha
        self.learning_rate: float = learning_rate
        self.max_iter : int = max_iter
        self.random_state = random_state
        self.momentum = momentum
        if not use_momentum:
             self.momentum = 0

    def fit(self,X_train : np.array, Y_train : np.array):
        """ X_train is an array of dimension (m,n),
            where m is the size of the training set and
            n is the numbers of feature
            Y_train is an array of dimension (m,1) with value of type int"""
        """ some preparation"""
        np.random.seed(self.random_state)
        X_train = X_train.T  
 
        """ we get the dummies matrix for Y_train so, if we want to identify
            between k-object,then Y_train below wwill be a (k-1,m) matrix
            so that if an first object has verylow activasion function on each row
            then this is belongs to the first object
        """
        Y_train = pd.get_dummies(
                                Y_train.reshape(Y_train.shape[0])
                                ).values.T
        
        self.W = list(range(len(self.hidden_layer_size)+1))

        self.b = list(range(len(self.hidden_layer_size)+1))

        dZ = list(range(len(self.hidden_layer_size)+1))

        dW = [0]*(len(self.hidden_layer_size)+1)#list(range(len(self.hidden_layer_size)+1))

        db = [0]*(len(self.hidden_layer_size)+1)#list(range(len(self.hidden_layer_size)+1))

        sigmoid = list(range(len(self.hidden_layer_size)+1))

        self.W[0] = np.random.randn(
                                    self.hidden_layer_size[0],X_train.shape[0]
                                    )

        self.b[0] = np.zeros((self.hidden_layer_size[0],1))

        self.W[len(self.hidden_layer_size)] = np.random.randn(
                                                        Y_train.shape[0],
                                                        self.hidden_layer_size[
                                                            len(self.hidden_layer_size)-1
                                                             ])

        self.b[len(self.hidden_layer_size)] = np.zeros((Y_train.shape[0],1))

        for i in range(1,len(self.hidden_layer_size)):

            self.W[i] = np.random.randn(self.hidden_layer_size[i],
                                        self.hidden_layer_size[i-1]
                                        )

            self.b[i] = np.zeros((self.hidden_layer_size[i],1))

        #------------------------------------------------------------------------------------------------------------------------------
        """now we compute the activation function of each neuron on every layer"""
        """ we do here a simple version just break after max_iter"""

        for m in range(self.max_iter):

            """forward propagation"""

            sigmoid[0] = 1/(1 + np.exp(-(self.W[0].dot(X_train) + self.b[0])))

            for j in range(1,len(self.hidden_layer_size)):

                sigmoid[j] = 1/(1 + np.exp(-(self.W[j].dot(sigmoid[j-1]) + self.b[j])))
            z = self.W[len(self.hidden_layer_size)].dot(sigmoid[len(self.hidden_layer_size)-1]
                                                        ) + self.b[len(self.hidden_layer_size)]
            z = np.exp(z)
            sigmoid[len(self.hidden_layer_size)] = z/z.sum(axis=0)
            #computer the cost function
            """I put the cost function here, but you can also put at the end of  this methode.
                The purpose was, that maybe some one want to plot the cost function after each iteration"""
            cost_function = -(Y_train*np.log(sigmoid[len(self.hidden_layer_size)])).sum()/Y_train.shape[1]

            """backward propagation"""
            dZ[len(self.hidden_layer_size)] = sigmoid[len(self.hidden_layer_size)] - Y_train

            dW[len(self.hidden_layer_size)] = (self.momentum * dW[len(self.hidden_layer_size)] + 
                            (1-self.momentum)* 1/X_train.shape[1]*dZ[len(self.hidden_layer_size)
                                                                    ].dot(sigmoid[len(
                                                                        self.hidden_layer_size)-1].T))

            db[len(self.hidden_layer_size)] = (self.momentum * db[len(self.hidden_layer_size)] + 
                                                (1 - self.momentum) * 1/X_train.shape[1]*np.sum(
                                                dZ[len(self.hidden_layer_size)],axis=1,keepdims=True))
        

            for k in range(len(self.hidden_layer_size)-1,0,-1):
                dZ[k] = self.W[k+1].T.dot(dZ[k+1])*(sigmoid[k]*(1-sigmoid[k]))

                dW[k] = (self.momentum * dW[k] + (1-self.momentum)
                                                 * 1/X_train.shape[1]*dZ[k].dot(sigmoid[k-1].T))

                db[k] = (self.momentum * db[k] + (1-self.momentum) 
                                                * 1/X_train.shape[1]*np.sum(dZ[k],axis=1,keepdims=True))


            dZ[0] = self.W[1].T.dot(dZ[1])*(sigmoid[0]*(1-sigmoid[0]))

            dW[0] = (self.momentum * dW[0] + (1-self.momentum)
                                            * 1/X_train.shape[1]*dZ[0].dot(X_train.T))

            db[0] = (self.momentum * db[0] + (1-self.momentum)
                                            * 1/X_train.shape[1]*np.sum(dZ[0],axis=1,keepdims=True))

            """update using gradient descent and regularization"""
            """we use here the frobenius norm(L2 norm of a matrix) this can be viewed as weight decay"""
            """we use momentum to so that the gradient descent method converts faster"""
            for l in range(len(self.hidden_layer_size)+1):

                self.W[l] = (1-self.learning_rate*self.alpha/X_train.shape[1]
                            )*self.W[l] - self.learning_rate*dW[l]

                self.b[l] = (1-self.learning_rate*self.alpha/X_train.shape[1]
                            )*self.b[l] - self.learning_rate*db[l]

        

        return cost_function



    def predict(self,X_test : np.array,threshold=0.6):
        
        prediction_array = X_test.T

        for i in range(len(self.hidden_layer_size)+1):

            prediction_array = self.W[i].dot(prediction_array) + self.b[i]

            prediction_array = 1/(1+np.exp(-prediction_array))
        return prediction_array.T.argmax(axis=1)

if __name__ == "__main__":
    X_train =np.array([[-100,-10],[0,2],[100,150],[-30,-1],[1,-2],[200,105],[4,5],[2,3]])
    Y_train = np.array([0,1,2,0,1,2,1,1])
    Y_train=Y_train.reshape((Y_train.shape[0],1))

    model = MLPClassifier((2,3,2),alpha=0.000001,max_iter=10000)
    print(model.fit(X_train,Y_train))
    X_test = np.array([[-40,-10],[3,2],[100,150],[-5,-2],[2,-2],[15,25]])
    d=model.predict(X_test)
    print(d)