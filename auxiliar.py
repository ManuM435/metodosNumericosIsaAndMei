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