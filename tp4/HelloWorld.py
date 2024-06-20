import numpy as np
import matplotlib.pyplot as plt
 

np.random.seed(19)
# xd con esta seed se van las iteraciones 
# claramente varian en base a la seed que usemos. A lo mejor podemos mostrar en Apendice como con algunas seeds varia, con otras no

def sol(A, b, x):
    return (A @ x - b)

def costFunction(A, b, x):
    return (sol(A, b, x)).T @ (sol(A, b, x))

def costFunctionL2(A, b, x, delta2):
    result = costFunction(A, b, x) +  delta2 * np.linalg.norm(x)**2
    return result

def gradient(A, x, b):
    return 2 * A.T @ sol(A, b, x)

def gradientL2(A, x, b, delta2):
    return gradient(A, x, b) + 2 * delta2 * x

def gradientDescent(start, A, b, learn_rate, iters, delta, svd_truth):
    x_f, x_f2 = np.array(start, dtype=float), np.array(start, dtype=float)
    trajectory_f, trajectory_f2 = [x_f.copy()], [x_f2.copy()]
    costF1, costF2 = [], []
    xFNorms, xF2Norms = [], []
    svdFnorms, svdF2Norms = [], []

    for _ in range(iters):
        # Definir los x
        evalF1 = gradient(A, x_f, b)
        x_f += - (learn_rate * evalF1)
        evalF2 = gradientL2(A, x_f2, b, delta)
        x_f2 += - (learn_rate * evalF2)

        # Calcular Normas de x_f & x_f2
        xFNorms.append((np.linalg.norm(x_f)))
        xF2Norms.append((np.linalg.norm(x_f2)))

        # Calcular Costos
        cost_f = costFunction(A, b, x_f)
        costF1.append(cost_f)
        cost_f2 = costFunction(A, b, x_f2)
        costF2.append(cost_f2)

        # Calcular Trajectories
        trajectory_f.append(x_f.copy())
        trajectory_f2.append(x_f2.copy())

        #Calcular el error relativo entre la X u Xsvd
        svdFnorms.append(np.linalg.norm(x_f - svd_truth)/np.linalg.norm(svd_truth))
        svdF2Norms.append(np.linalg.norm(x_f2 - svd_truth)/np.linalg.norm(svd_truth))

    return (trajectory_f, costF1), (trajectory_f2, costF2), (xFNorms, xF2Norms), (svdFnorms, svdF2Norms)


def SVDFinder(matrix, b):
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

def randomMatrixGenerator(n, d):
    A = np.random.randn(n, d)
    b = np.random.randn(n)
    return A, b

# Plot Functions

def plotFvF2(F1List, F2List, Category, Title, scale):
    plt.plot(range(len(F1List)), F1List, label='Original')
    plt.plot(range(len(F2List)), F2List, label='L2 Regularized')
    plt.xlabel('Iteration')
    plt.ylabel(Category)
    plt.title(Title)
    plt.yscale(scale)
    plt.legend()
    plt.show()



# Definimo' Lo' Parametro's
n = 5
d = 100

A, b = randomMatrixGenerator(n, d)

# Inicializar el vector x con d elementos
startOriginal = np.random.randint(0, 10, d)

stepOriginal = stepInator(A)

iterations = 4000




# Use the functions
svd_truth, sigmaMax = SVDFinder(A, b)
deltaOriginal = 0.01 * sigmaMax  # Choose an appropriate value for delta

(final_x_f, costPerIter_f), (final_x_f2, costPerIter_f2), (xFNorms, xF2Norms), (svdFnorms, svdF2Norms) = gradientDescent(startOriginal, A, b, stepOriginal, iterations, deltaOriginal, svd_truth)

# # Graf'em Norms
# plotFvF2(xFNorms, xF2Norms, 'Norm', 'Norm of x', scale='linear')

# # Graf'em Costos
# plotFvF2(costPerIter_f, costPerIter_f2, 'Cost', 'Cost of Gradient Descent', scale='log')

# # Graf'em relative error SVD
# plotFvF2(svdFnorms, svdF2Norms, 'Relative Error SVD', 'Relative Error between x and x_svd', scale='log')





# Plot The Functions with Different Starting Conditions
starts = [startOriginal]
for s in range(1, 6):
    starter = np.random.randint(s*2 + 1, s**2 + 5, d)
    starts.append(starter)
print(starts)




# Plot The Functions with Different Step Sizes
steps = []


# Plot Cost Function F2 convergence with varying delta values
deltas = [0.1, 1, deltaOriginal, 24, 100]




# TODO
# (Maybe) plotear como varia la convergencia con distintos valores de Delta 
# plot_isocost_and_trajectory(A, b, final_x_f, final_x_f2)
