import numpy as np

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def discounted_cumsum(arr, gamma):
    discounted = arr[-1:]
    for i in reversed(range(len(arr) - 1)):
        discounted.append(arr[i]+ discounted[-1] * gamma)
    return discounted[::-1]

