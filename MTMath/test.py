import numpy as np
from powerlaw import cumulative_distribution_function
from numpy import searchsorted, unique

data = np.array([1, 2, 3, 4, 5])
x, F = cumulative_distribution_function(data)
print(F)

data = [10, 20, 20, 50, 20]
n = len(data)
CDF = searchsorted(data, data, side="left") / len(data)
print(CDF)
CDF = searchsorted(data, data, side="right") / len(data)
print(CDF)
