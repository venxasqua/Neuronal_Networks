# Neuronal_Networks
This is my first learning project to Neuronal Network.

requirements:
numba, numpy and pandas


There is some thing Iwant to say here.

First, my Neuronal network can be used for Multi-class Classification.
Here, I used one vs all. So if you want to recognize c number of classes, then Y_train is a (m,1) dimensional matrix, where m is the training size, and Y_train has values between 0 and c-1. 
By default you use mode == "multi-task", so if you want to for exaple to detect multiple things at once and if the classes you want to dectect is not there then it returns -1, you also should to change threshold.
The second mode you can use is "softmax", this will returns the class with the highest possibility.


