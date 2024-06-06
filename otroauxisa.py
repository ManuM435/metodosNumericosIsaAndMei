import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics
import auxiliar as aux

# Abrir el archivo csv
promedio_columnas = []
with open('dataset.csv', 'r') as f:
    dataset = list(csv.reader(f))
    for i in range(1, len(dataset[0])):  # Comienza desde la segunda columna
        columna = [float(fila[i]) for fila in dataset[1:]]  # Comienza desde la segunda fila
        columna_sin_none = [x for x in columna if x is not None]
        promedio_columnas.append(statistics.mean(columna_sin_none))
    
    matriz_datos = []
    for fila in dataset[1:]:  # Comienza desde la segunda fila
        matriz_datos.append([float(dato) for dato in fila[1:]])  # Comienza desde la segunda columna
vectores = []
with open('dataset.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Ignora la primera fila
    for fila in reader:   
        vectores.append([float(i) if i else 0 for i in fila[1:]])
with open('y.txt', 'r') as f:
    y = [float(line) for line in f]

# Calcular Matriz Mediana
matriz_mediana = np.array(matriz_datos) - np.mean(matriz_datos)

# Calcular la matriz de similaridad centrada
similarity_matrix_centered = aux.eucledian_distance(15, matriz_mediana)
#Grafica la matriz de similaridad centrada como una imagen
plt.imshow(similarity_matrix_centered, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Matriz de Similaridad Centrada')
plt.show()