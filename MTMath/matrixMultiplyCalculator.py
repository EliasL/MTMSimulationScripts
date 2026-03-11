import numpy as np

# Just a sheet to do some simple calculations

m1 = np.array(((1, 0), (0, -1)))
m2 = np.array(((0, 1), (1, 0)))
m3 = np.array(((1, -1), (0, 1)))
e = np.array(((1, 0), (-1, 1)))
print(e @ m1)
print(e @ m1 @ m2)
print(e @ m1 @ m2 @ m3)
e2 = e @ m1 @ m2 @ m3
print(e2.T @ e2)
print(e @ m1 @ m2 @ m3 @ m1 @ m2)
print(e2.T @ e2)
