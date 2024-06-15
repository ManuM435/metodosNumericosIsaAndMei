import numpy as np

np.random.seed(9923)

def gradient(A, x, b):
    return 2 * A.T @ (A @ x - b)

def gradient_descent(start, A, b, learn_rate, iters):
    x = np.array(start, dtype=float)
    for _ in range(iters):
        eval_grad = gradient(A, x, b)
        diff = -learn_rate * eval_grad
        x += diff
    return x.tolist()

n = 5
d = 100

# Generar una matriz A de tamaño n x d con valores aleatorios
A = np.random.rand(n, d)

# Generar un vector b de tamaño n con valores aleatorios
b = np.random.rand(n)

# Inicializar el vector x con d elementos
start = np.random.rand(d)

#Calcula el step 1/lambda max que es el maximo autovalor del hessiano de F(x), AtA
hessiano = A.T @ A
autovalores_h = np.linalg.eigvals(hessiano)

# Toma la parte real de los autovalores
autovalores_reales = np.real(autovalores_h)


step = 1/max(autovalores_reales)
print(step)

iterations = 300
print(gradient_descent(start, A, b, step, iterations))
print(len(gradient_descent(start, A, b, step, iterations)))