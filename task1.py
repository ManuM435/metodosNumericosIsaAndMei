import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics
import math
import auxiliar as aux
from scipy.ndimage import zoom

# Abrir el archivo csv
promedio_columnas = []

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



# Calcular Matriz Mediana
matriz_mediana = [[matriz_datos[i][j] - promedio_columnas[j] for j in range(len(matriz_datos[0]))] for i in range(len(matriz_datos))]




# Esto es para graficar la matriz original

# matriz_promedio_columnas = []
# for fila in matriz_datos:
#     matriz_promedio_columnas.append([dato - promedio_columnas[i] for i, dato in enumerate(fila)])

# matriz_mediana = [[matriz_datos[i][j] - promedio_columnas[j] for j in range(len(matriz_datos[0]))] for i in range(len(matriz_datos))]

# # primera_columna_mediana = [fila[0] for fila in matriz_mediana]
# # print(primera_columna_mediana)
# # Calcular el factor de zoom para cada dimensión
# zoom_factor = [2000 / len(matriz_mediana), 2000 / len(matriz_mediana[0])]

# # Redimensionar la matriz con interpolación de orden cero
# matriz_mediana_resized = zoom(matriz_mediana, zoom_factor, order=0)

# # Mostrar la matriz redimensionada como una imagen
# fig, ax = plt.subplots(figsize=(6, 6)) 
# ax.imshow(matriz_mediana, cmap='hot', interpolation='none')
# plt.show()

# primera_columna = [fila[0] for fila in matriz_mediana]
# print(primera_columna)



def eucledian_distance(sigma, matrix):
    matriz_similaridad = [[0 for _ in range(len(matrix))] for _ in range(len(matrix))]

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            distancia = math.sqrt(sum((matrix[i][k] - matrix[j][k]) ** 2 for k in range(len(matrix[i]))))
            matriz_similaridad[i][j] = np.exp(-distancia ** 2 / (2 * sigma ** 2))
    return matriz_similaridad

# # Crea una matriz de similaridad vacía
# matriz_similaridadUncent = [[0 for _ in range(len(vectores))] for _ in range(len(vectores))]

# # Define el valor de σ
# sigma = 20  # Ajusta este valor según sea necesario

# # Calcula la distancia euclidiana entre cada par de vectores y aplica la función de kernel
# for i in range(len(vectores)):
#     for j in range(len(vectores)):
#         distancia = math.sqrt(sum((vectores[i][k] - vectores[j][k]) ** 2 for k in range(len(vectores[i]))))
#         matriz_similaridadUncent[i][j] = np.exp(-distancia ** 2 / (2 * sigma ** 2))



similarity_matrix = eucledian_distance(20, vectores)

# Grafica la matriz de similaridad como una imagen
plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()



# Esto es para hacer la matriz de similaridad Similaridad Centrada

# # Crea una matriz de similaridad vacía
# matriz_similaridadCent = [[0 for _ in range(len(matriz_mediana))] for _ in range(len(matriz_mediana))]

# # Define el valor de σ
# sigma = 50  # Ajusta este valor según sea necesario

# # Calcula la distancia euclidiana entre cada par de vectores y aplica la función de kernel
# for i in range(len(matriz_mediana)):
#     for j in range(len(matriz_mediana)):
#         distancia = math.sqrt(sum((matriz_mediana[i][k] - matriz_mediana[j][k]) ** 2 for k in range(len(matriz_mediana[i]))))
#         matriz_similaridadCent[i][j] = np.exp(-distancia ** 2 / (2 * sigma ** 2))

similarity_matrix_centered = eucledian_distance(50, matriz_mediana)

# Grafica la matriz de similaridad como una imagen
plt.imshow(similarity_matrix_centered, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()






# Valores Singulares

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