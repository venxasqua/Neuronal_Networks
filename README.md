# Neuronal_Networks

This is my first learning project to Neuronal Network.

There is some thing Iwant to say here.
First, my Neuronal network can be used for Multi-class Classification.
Here, I used one vs all. So if you want to recognize c number of classes, then Y_train is a (m,1) dimensional matrix, where m is the training size, and Y-train has values between 0 and c-1. 
Note here that the last layer (out-put layer) will have only c-1 neurons, so that if the acitivision function of the last layer is to low, then this belong to the first class.

Secondly, if you want to use Softmax Regression instead, then you need to use Normalization first. So we dont want to get a very high/small value.