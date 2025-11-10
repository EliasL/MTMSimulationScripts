import numpy as np


# w-functions cause warnings
# s-functions are "safe" and do not cause warnings


# def w1():
#     p1 = np.array([[0.0, 1], [1, 1]])
#     A0 = p1.T @ p1


# w1()


"""
This does not cause warning
p1 = np.array([[1, 1], [0.0, 1]])
A0 = p1.T @ p1
"""


# def s1():
#     p1 = np.array([[0, 1], [1, 1]])
#     A0 = p1.T @ p1


# s1()


# def s2():
#     p1 = np.array([[0, 1.0], [1, 1]], dtype=float)
#     A0 = p1.T @ p1


# s2()


def s3():
    p1 = np.array([[1.0, 0.3], [0, 1.0]])
    A0 = p1.T @ p1


s3()
