import numpy as np
from task1 import matriz_datos, y

def gradient(A, x, b):
    return 2 * A.T @ (A @ x - b)

def gradient_descent(start, A, b, learn_rate, iters):
    x = np.array(start, dtype=float)
    for _ in range(iters):
        eval_grad = gradient(A, x, b)
        diff = -learn_rate * eval_grad
        x += diff
    return x.tolist()

A = np.array([[1, 2], [3, 4]])
b = np.array([1, 2])

iterations = 300
print(gradient_descent([1, 1], matriz_datos, y, 0.5, iterations))