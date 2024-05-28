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


# Generar el gráfico de dispersión
plt.figure(figsize=(10, 5))
plt.scatter(U_reducida[:, 0], U_reducida[:, 1])
plt.title('Proyección de los datos en los dos primeros autovectores')
plt.xlabel('Autovector 1')
plt.ylabel('Autovector 2')
plt.show()