import numpy as np


x = np.zeros((3,3,3))
y = np.ones((3,3))
y[1,1]=0
y = y==1
print(x.shape)
print(y.shape)
x[1, y] = 3
print(x)