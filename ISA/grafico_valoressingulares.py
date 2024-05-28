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

# Generar el gráfico
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(S) + 1), S)
plt.title('Valores singulares de la matriz S')
plt.xlabel('Índice del valor singular')
plt.ylabel('Valor del valor singular')
plt.show()