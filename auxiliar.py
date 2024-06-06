import math
import numpy as np

def expoDist(distance, sigma):
    return np.exp(-distance ** 2 / (2 * sigma ** 2))

def eucledian_distance(sigma, matrix):
    matriz_similaridad = [[0 for _ in range(len(matrix))] for _ in range(len(matrix))]

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            distancia = math.sqrt(sum((matrix[i][k] - matrix[j][k]) ** 2 for k in range(len(matrix[i]))))
            matriz_similaridad[i][j] = expoDist(distancia, sigma)
    return matriz_similaridad


def PCAinator(Matriz, Dimensions):
    U, S, Vt = np.linalg.svd(Matriz, full_matrices=False)
    U_reduced = U[:, :Dimensions]
    S_reduced = np.diag(S[:Dimensions])
    matrix_reduced = np.dot(U_reduced, S_reduced)

    return matrix_reduced, Vt[:Dimensions, :]


def frobeniusNorm(X):
    norm = 0
    for i in range(len(X)):
        for j in range(len(X[0])):
            norm += X[i][j] ** 2
    return np.sqrt(norm)


def redimensionalizerInator(image, dimension):
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    U_reduced = U[:, :dimension]
    S_reduced = np.diag(S[:dimension])
    Vt_reduced = Vt[:dimension, :]
    image_reduced = U_reduced @ S_reduced @ Vt_reduced
    return image_reduced


def frobeniusRelativeError(OriginalMatrix, dimension):
    reduced_matrix = redimensionalizerInator(OriginalMatrix, dimension)
    error = frobeniusNorm(OriginalMatrix - reduced_matrix) / frobeniusNorm(OriginalMatrix)
    return error

def frobeniusMaximumError(image_list, dimension):
    errors = []
    for image in image_list:
        error = frobeniusRelativeError(image, dimension)
        errors.append(error)
    return max(errors)

def errorByDimensions(image_list, max_dimension):
    errors = []
    for i in range(1, max_dimension + 1):
        error = frobeniusMaximumError(image_list, i)
        errors.append(error)
    return errors

