import csv
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

matriz_mediana = np.array([[matriz_datos[i][j] - promedio_columnas[j] for j in range(len(matriz_datos[0]))] for i in range(len(matriz_datos))])

# Define el valor de σ
sigma = 10  # Ajusta este valor según sea necesario

# Descomposición SVD
U, S, Vt = np.linalg.svd(matriz_mediana)

# Reducir la dimensionalidad a dos
U_reducida = U[:, :2]
S_reducida = np.diag(S[:2])
Vt_reducida = Vt[:2, :]

# # Reducir la dimensionalidad a seis
# U_reducida = U[:, :6]
# S_reducida = np.diag(S[:6])
# Vt_reducida = Vt[:6, :]

# # Reducir la dimensionalidad a diez
# U_reducida = U[:, :10]
# S_reducida = np.diag(S[:10])
# Vt_reducida = Vt[:10, :]

# Reconstruir la matriz
matriz_reconstruida = U_reducida @ S_reducida @ Vt_reducida

# Crea una matriz de similaridad vacía
matriz_similaridad_reducida = np.zeros((len(matriz_reconstruida), len(matriz_reconstruida)))

# Calcula la distancia euclidiana entre cada par de vectores y aplica la función de kernel
for i in range(len(matriz_reconstruida)):
    for j in range(len(matriz_reconstruida)):
        distancia = np.linalg.norm(matriz_reconstruida[i] - matriz_reconstruida[j])
        matriz_similaridad_reducida[i][j] = np.exp(-distancia ** 2 / (2 * sigma ** 2))

# Graficar la matriz de similaridad reducida
plt.subplot(1, 2, 2)
plt.imshow(matriz_similaridad_reducida, cmap='hot', interpolation='nearest')
plt.colorbar()

plt.show()