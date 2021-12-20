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
    def fit(self):
        pass