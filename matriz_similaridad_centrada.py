import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import statistics

# Lee todas las filas del archivo CSV y calcula la matriz centrada
promedio_columnas = []
with open('ISA/dataset.csv', 'r') as f:
    dataset = list(csv.reader(f))
    for i in range(1, len(dataset[0])):  # Comienza desde la segunda columna
        columna = [float(fila[i]) for fila in dataset[1:]]  # Comienza desde la segunda fila
        columna_sin_none = [x for x in columna if x is not None]
        promedio_columnas.append(statistics.mean(columna_sin_none))
    
    matriz_datos = []
    for fila in dataset[1:]:  # Comienza desde la segunda fila
        matriz_datos.append([float(dato) for dato in fila[1:]])  # Comienza desde la segunda columna

matriz_mediana = [[matriz_datos[i][j] - promedio_columnas[j] for j in range(len(matriz_datos[0]))] for i in range(len(matriz_datos))]

# Crea una matriz de similaridad vacía
matriz_similaridad = [[0 for _ in range(len(matriz_mediana))] for _ in range(len(matriz_mediana))]

# Define el valor de σ
sigma = 50  # Ajusta este valor según sea necesario

# Calcula la distancia euclidiana entre cada par de vectores y aplica la función de kernel
for i in range(len(matriz_mediana)):
    for j in range(len(matriz_mediana)):
        distancia = math.sqrt(sum((matriz_mediana[i][k] - matriz_mediana[j][k]) ** 2 for k in range(len(matriz_mediana[i]))))
        matriz_similaridad[i][j] = np.exp(-distancia ** 2 / (2 * sigma ** 2))

# Grafica la matriz de similaridad como una imagen
plt.imshow(matriz_similaridad, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()