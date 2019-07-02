import numpy as np


def tridiag(l=-1, c=-4, r=-1, size=3):
    m = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            if i == j:
                m[i, j] = c
            elif i == j-1:
                m[i, j] = l
            elif i == j+1:
                m[i, j] = r
    return m


def hilb(size=3):
    m = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            m[i, j] = i+j+1

    return m
