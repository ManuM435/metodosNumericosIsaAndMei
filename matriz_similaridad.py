import csv
import math
import matplotlib.pyplot as plt
import numpy as np

# Lee todas las filas del archivo CSV
vectores = []
with open('ISA/dataset.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Ignora la primera fila
    for fila in reader:   
        vectores.append([float(i) if i else 0 for i in fila[1:]])

# Crea una matriz de similaridad vacía
matriz_similaridad = [[0 for _ in range(len(vectores))] for _ in range(len(vectores))]

# Define el valor de σ
sigma = 20  # Ajusta este valor según sea necesario


# Calcula la distancia euclidiana entre cada par de vectores y aplica la función de kernel
for i in range(len(vectores)):
    for j in range(len(vectores)):
        distancia = math.sqrt(sum((vectores[i][k] - vectores[j][k]) ** 2 for k in range(len(vectores[i]))))
        matriz_similaridad[i][j] = np.exp(-distancia ** 2 / (2 * sigma ** 2))

# Grafica la matriz de similaridad como una imagen
plt.imshow(matriz_similaridad, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()