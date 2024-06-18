import numpy as np
import matplotlib.pyplot as plt
 

np.random.seed(9923)

def costFunction(A, b, x):
    return (A@x - b).T @ (A@x - b)
    

def costFunctionL2(A, b, x, delta2):
    result = costFunction(A, b, x) +  delta2* np.linalg.norm(x)**2
    print(np.linalg.norm(x)**2)
    return result

def gradient(A, x, b):
    return 2 * A.T @ (A @ x - b)

def gradientL2(A, x, b, delta2):
    return 2 * A.T @ (A @ x - b) + 2 * delta2 * x

def gradientDescent(start, A, b, learn_rate, iters, x_truth, delta):
    x_f = np.array(start, dtype=float)
    x_f2 = np.array(start, dtype=float)
    trajectory_f = [x_f.copy()]
    trajectory_f2 = [x_f2.copy()]
    error_por_iter_f = []
    error_por_iter_f2 = []
    cost_por_iter_f = []
    cost_por_iter_f2 = []
    for _ in range(iters):
        eval_grad_f = gradient(A, x_f, b)
        x_f += - (learn_rate * eval_grad_f)
        error_f = np.linalg.norm(x_f - x_truth)
        cost_f = costFunction(A, b, x_f)
        error_por_iter_f.append(error_f)
        cost_por_iter_f.append(cost_f)
        trajectory_f.append(x_f.copy())

        eval_grad_f2 = gradientL2(A, x_f, b, delta)
        x_f2 += - (learn_rate * eval_grad_f2)
        error_f2 = np.linalg.norm(x_f2 - x_truth)
        cost_f2 = costFunctionL2(A, b, x_f, delta)
        error_por_iter_f2.append(error_f2)
        cost_por_iter_f2.append(cost_f2)
        trajectory_f2.append(x_f2.copy())

    return (trajectory_f, error_por_iter_f, cost_por_iter_f), (trajectory_f2, error_por_iter_f2, cost_por_iter_f2)

def groundTruthFinder(matrix, b):
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    S_inv = np.zeros((U.shape[0], U.shape[1]))
    for i in range(len(S)):
        if S[i] != 0:
            S_inv[i, i] = 1/S[i]
    x = Vt.T @ S_inv @ U.T @ b
    sigmaMax = S[0]
    return x, sigmaMax

def stepInator(A):
    hessian = 2 * A.T @ A
    eigenVals = np.linalg.eigvals(hessian)
    realEigens = eigenVals.real
    maxEigen = max(realEigens)
    step = 1/maxEigen
    return step

n = 5
d = 100

def randomMatrixGenerator(n, d):
    A = np.random.randn(n, d)
    b = np.random.randn(n)
    return A, b

A, b = randomMatrixGenerator(n, d)

# Inicializar el vector x con d elementos
start = np.random.randint(0, 10, d)

step = stepInator(A)

iterations = 1000

def plot_error(errorPerIter, errorL2PerIter):
    plt.plot(range(len(errorPerIter)), errorPerIter, label='Original')
    plt.plot(range(len(errorL2PerIter)), errorL2PerIter, label='L2 Regularized')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error of Gradient Descent')
    plt.yscale('log')
    plt.legend()
    plt.show()

def plot_cost(costPerIter, costL2PerIter):
    plt.plot(range(len(costPerIter)), costPerIter, label='Original')
    plt.plot(range(len(costL2PerIter)), costL2PerIter, label='L2 Regularized')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost of Gradient Descent')
    plt.yscale('log')
    plt.legend()
    plt.show()

def costFunction2D(A, b, x_range, y_range):
    Z = np.zeros((len(x_range), len(y_range)))
    for ix, x in enumerate(x_range):
        for iy, y in enumerate(y_range):
            Z[ix, iy] = costFunction(A, b, np.array([x, y]))
    return Z


def plot_isocost_and_trajectory(A, b, trajectory_f, trajectory_f2):
    x_range = np.linspace(-10, 10, 100)
    y_range = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z_f = costFunction2D(A, b, x_range, y_range)
    Z_f2 = costFunction2D(A, b, x_range, y_range)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.contour(X, Y, Z_f, levels=20)
    plt.plot([x[0] for x in trajectory_f], [x[1] for x in trajectory_f], 'r-')
    plt.title('Isocost curve and trajectory for f')

    plt.subplot(1, 2, 2)
    plt.contour(X, Y, Z_f2, levels=20)
    plt.plot([x[0] for x in trajectory_f2], [x[1] for x in trajectory_f2], 'r-')
    plt.title('Isocost curve and trajectory for f2')

    plt.show()

# Use the functions
x_truth, sigmaMax = groundTruthFinder(A, b)
delta2 = 0.01*sigmaMax  # Choose an appropriate value for delta
(final_x_f, errorPerIter_f, costPerIter_f), (final_x_f2, errorPerIter_f2, costPerIter_f2) = gradientDescent(start, A, b, step, iterations, x_truth, delta2)
# print(final_x_f)
# print(final_x_f2)


# plot_error(errorPerIter_f, errorPerIter_f2)

plot_cost(costPerIter_f, costPerIter_f2)


plot_isocost_and_trajectory(A, b, final_x_f, final_x_f2)

print('Autovalor mas imp', sigmaMax)
#graficar la norma 2 de x (se deberia achicar ) y la de Ax - b