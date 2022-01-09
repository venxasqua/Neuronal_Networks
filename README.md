# Neuronal_Networks
This is my first learning project to Neuronal Network.

requirements:
numba, numpy and pandas


There is some thing Iwant to say here.

First, my Neuronal network can be used for Multi-class Classification.

So if you want to recognize c number of classes, then Y_train is a (m,1) dimensional matrix, where m is the training size, and Y_train has values between 0 and c-1. 
By default you use mode == "multi-task", so if you want to detect multiple things at once and if the classes you want to dectect is not there then it returns -1, you also should to change threshold.
The second mode you can use is "softmax", this will returns the class with the highest possibility.

You also can improve the program by normalize each layer and do mini-batch gradient descent, but dont use stochastic descent, since you will lose the vectorize programming.

I will ad mini-btach gradient descent so, one use parallel programming to make the computing faster. I also will add a test program.
