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
        diff = learn_rate * evalF1
        x_f += - (diff)
        evalF2 = gradientL2(A, x_f2, b, delta)
        x_f2 += - (learn_rate * evalF2)

        # Norms
        x_fNorm = np.linalg.norm(x_f)
        x_f2Norm = np.linalg.norm(x_f2)

        # Calcular Normas de x_f & x_f2
        xFNorms.append(x_fNorm)
        xF2Norms.append(x_f2Norm)

        error1 = x_f - svd_truth
        error2 = x_f2 - svd_truth

        relativeError1 = np.linalg.norm(error1)/np.linalg.norm(svd_truth)
        relativeError2 = np.linalg.norm(error2)/np.linalg.norm(svd_truth)

        #Calcular el error relativo entre la X u Xsvd
        svdFnorms.append(relativeError1)
        svdF2Norms.append(relativeError2)


        # Calcular Costos
        cost_f = costFunction(A, b, x_f)
        costF1.append(cost_f)
        cost_f2 = costFunction(A, b, x_f2)
        costF2.append(cost_f2)

        # Calcular Trajectories
        trajectory_f.append(x_f)
        trajectory_f2.append(x_f2)

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
    plt.plot(range(len(F1List)), F1List, label='Original', color="orchid")
    plt.plot(range(len(F2List)), F2List, label='L2 Regularized', color="mediumvioletred")
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


# # Uncomment this for graphs
# Graf'em Norms
plotFvF2(xFNorms, xF2Norms, 'Norm', 'Norm of Approximation Solutions', scale='linear')

# Graf'em Costos
plotFvF2(costPerIter_f, costPerIter_f2, 'Cost', 'Cost of Gradient Descent', scale='log')

plt.plot(range(len(costPerIter_f2)), costPerIter_f2, label='L2 Regularized', color="mediumvioletred")
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost of F2')
plt.yscale('log')
plt.legend()
plt.show()

# Graf'em relative error SVD
plotFvF2(svdFnorms, svdF2Norms, 'Relative Error SVD', 'Relative Error between Solution and Pseudoinverse', scale='log')





# # Plot The Functions with Different Starting Conditions
# starts = [startOriginal]
# x1_vals_start, x2_vals_start = [], []
# x1_cost_start, x2_cost_start = [], []

# startHigh = np.random.randint(50, 80, d)
# starts.append(startHigh)

# startLow = np.random.randint(-80, -50, d)
# starts.append(startLow)

# for starting in starts:
#     (x1f_sts, cost1f_sts), (x1f2_sts, cost1f2_sts), _, _ = gradientDescent(starting, A, b, stepOriginal, iterations, deltaOriginal, svd_truth)
#     x1_vals_start.append(x1f_sts)
#     x2_vals_start.append(x1f2_sts)
#     x1_cost_start.append(cost1f_sts)
#     x2_cost_start.append(cost1f2_sts)

# # Plot the convergence of the cost functions with the 5 different starting conditions
# plt.plot(range(len(x1_cost_start[0])), x1_cost_start[0], label='Original F', color="forestgreen", linestyle='--', alpha=0.85)
# plt.plot(range(len(x2_cost_start[0])), x2_cost_start[0], label='Original F2', color="lime", linestyle='--', alpha=0.85)
# plt.plot(range(len(x1_cost_start[1])), x1_cost_start[1], label='High Start F', color="darkorange")
# plt.plot(range(len(x2_cost_start[1])), x2_cost_start[1], label='High Start F2', color="orangered")
# plt.plot(range(len(x1_cost_start[2])), x1_cost_start[2], label='Low Start F', color="royalblue")
# plt.plot(range(len(x2_cost_start[2])), x2_cost_start[2], label='Low Start F2', color="navy")
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.title('Cost of Gradient Descent with Different Starting Conditions')
# plt.yscale('log')
# plt.legend(loc='upper right')
# plt.show()


# # Plot The Functions with Different Step Sizes
# steps = [stepOriginal]
# x1_vals_step, x2_vals_step = [], []
# x1_cost_step, x2_cost_step = [], []

# stepHigh = stepOriginal * 1.92
# steps.append(stepHigh)

# stepLow = stepOriginal / 1.92
# steps.append(stepLow)

# step_iters = 4000

# for stepping in steps:
#     (x1f_stp, cost1f_stp), (x1f2_stp, cost1f2_stp), _, _ = gradientDescent(startOriginal, A, b, stepping, step_iters, deltaOriginal, svd_truth)
#     x1_vals_step.append(x1f_stp)
#     x2_vals_step.append(x1f2_stp)
#     x1_cost_step.append(cost1f_stp)
#     x2_cost_step.append(cost1f2_stp)

# # Plot the convergence of the cost functions with the 5 different steps
# plt.plot(range(len(x1_cost_step[0])), x1_cost_step[0], label='Original F', color="forestgreen", linestyle='--', alpha=0.85)
# plt.plot(range(len(x2_cost_step[0])), x2_cost_step[0], label='Original F2', color="lime", linestyle='--', alpha=0.85)
# plt.plot(range(len(x1_cost_step[1])), x1_cost_step[1], label='High Step F', color="darkorange")
# plt.plot(range(len(x2_cost_step[1])), x2_cost_step[1], label='High Step F2', color="orangered")
# plt.plot(range(len(x1_cost_step[2])), x1_cost_step[2], label='Low Step F', color="royalblue")
# plt.plot(range(len(x2_cost_step[2])), x2_cost_step[2], label='Low Step F2', color="navy")
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.title('Cost of Gradient Descent with Different Step Sizes')
# plt.yscale('log')
# plt.legend(loc='upper right')
# plt.show()


# # Plot Cost Function F2 convergence with varying delta values
# deltas = [deltaOriginal]
# x1_vals_delta, x1_cost_delta = [], []
# x2_vals_delta, x2_cost_delta = [], []

# deltaHigh = (deltaOriginal * 60) ** 2
# deltas.append(deltaHigh)

# deltaLow = (deltaOriginal / 30) ** 2
# deltas.append(deltaLow)

# for deltaing in deltas:
#     (x1f1_dlt, cost1f1_dlt), (x1f2_dlt, cost1f2_dlt), _, _ = gradientDescent(startOriginal, A, b, stepOriginal, iterations, deltaing, svd_truth)
#     x1_vals_delta.append(x1f1_dlt)
#     x1_cost_delta.append(cost1f1_dlt)
#     x2_vals_delta.append(x1f2_dlt)
#     x2_cost_delta.append(cost1f2_dlt)

# (x1f_dltOg, cost1f_dltOg), _, _, _ = gradientDescent(startOriginal, A, b, stepOriginal, iterations, deltaOriginal, svd_truth)

# # Plot the convergence of the cost functions with the 5 different deltas
# plt.plot(range(len(x1_cost_delta[0])), x1_cost_delta[0], label='Original F', color="forestgreen", linestyle='--', alpha=0.85)
# plt.plot(range(len(x2_cost_delta[0])), x2_cost_delta[0], label='Original F2', color="lime", linestyle='--', alpha=0.85)
# plt.plot(range(len(x2_cost_delta[1])), x2_cost_delta[1], label='High Delta F2', color="orangered")
# plt.plot(range(len(x2_cost_delta[2])), x2_cost_delta[2], label='Low Delta F2', color="navy")
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.title('Cost of Gradient Descent with Different Delta Values')
# plt.yscale('log')
# plt.legend(loc='upper right')
# plt.show()




