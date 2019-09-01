import numpy as np


def identity(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


ACTIVATIONS = [identity, sigmoid]
