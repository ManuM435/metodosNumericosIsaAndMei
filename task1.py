import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics
import auxiliar as aux
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D


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

# Esto de arriba es de la matriz original


# # Calcular la matriz de similaridad
# similarity_matrix = aux.eucledian_distance(20, vectores)

# # Grafica la matriz de similaridad como una imagen
# plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')
# plt.title('Matriz de Similaridad')
# plt.show()


# # Calcular la matriz de similaridad centrada
# similarity_matrix_centered = aux.eucledian_distance(50, matriz_mediana)

# # Grafica la matriz de similaridad centrada como una imagen
# plt.imshow(similarity_matrix_centered, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')
# plt.title('Matriz de Similaridad Centrada')
# plt.show()





# # Down Here van Valores Singulares (Barritas que muestran la significancia de los valores de d) y PCA (Dispersion 3D) 

 # Descomposición SVD
U, S, Vt = np.linalg.svd(matriz_mediana)

# # Grafico de Significancia de Dimensiones
# plt.figure(figsize=(10, 5))
# plt.bar(range(1, len(S) + 1), S)
# plt.title('Valores singulares de la matriz S')
# plt.xlabel('Índice del valor singular')
# plt.ylabel('Valor del valor singular')
# plt.show()


# Para PCA y Dispersion 2D
U_reducida = U[:, :2]
S_reducida = np.diag(S[:2])
Vt_reducida = Vt[:2, :]


# # Generar el gráfico de dispersión
# plt.figure(figsize=(10, 5))
# plt.scatter(U_reducida[:, 0], U_reducida[:, 1])
# plt.title('Proyección de los datos en los dos primeros autovectores')
# plt.xlabel('Autovector 1')
# plt.ylabel('Autovector 2')
# plt.show()


# # Para PCA y Dispersion 3D
# U_reducida3 = U[:, :3]
# S_reducida3 = np.diag(S[:3])
# Vt_reducida3 = Vt[:3, :]


# # Grafico de Dispersion 3D (PCA)
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(U_reducida3[:, 0], U_reducida3[:, 1], U_reducida3[:, 2])
# ax.set_title('Proyección de los datos en los tres primeros autovectores')
# ax.set_xlabel('Autovector 1')
# ax.set_ylabel('Autovector 2')
# ax.set_zlabel('Autovector 3')
# plt.show()



# # Aca graficamos la matriz de similaridad con distintos valores de d

# # # Generar el diccionario
# # dict_autovector = {i: v for i, v in enumerate(Vt[0])}

# # # Ordenar el diccionario de mayor a menor
# # dict_autovector_ordenado = dict(sorted(dict_autovector.items(), key=lambda item: item[1], reverse=True))

# # # Imprimir el diccionario ordenado
# # for key, value in dict_autovector_ordenado.items():
# #     print(f'Índice: {key}, Valor: {value}')

# # Reducir la dimensionalidad a dos
# U_reduced = U[:, :2]
# S_reduced = np.diag(S[:2])
# Vt_reduced = Vt[:2, :]

# # # Reducir la dimensionalidad a seis
# # U_reduced = U[:, :6]
# # S_reduced = np.diag(S[:6])
# # Vt_reduced = Vt[:6, :]

# # # Reducir la dimensionalidad a diez
# # U_reduced = U[:, :10]
# # S_reduced = np.diag(S[:10])
# # Vt_reduced = Vt[:10, :]

# # Reconstruir la matriz
# matriz_reconstruida = U_reduced @ S_reduced @ Vt_reduced

# sim_matrix_reduced = aux.eucledian_distance(10, matriz_reconstruida)

# # Graficar la matriz de similaridad reducida
# plt.subplot(1, 2, 2)
# plt.imshow(sim_matrix_reduced, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')
# plt.title('Reduced Similarity Matrix')
# plt.show()


#Cuadrados minimos

# Calculate the mean of y
mean_y = sum(y) / len(y)

# Subtract the mean from each value in y to center it
y_centered = [value - mean_y for value in y]

# Convert S from a 1D array to a 2D diagonal matrix
S_diag = np.diag(S)

# Convert matriz_mediana to a NumPy array
matriz_mediana_np = np.array(matriz_mediana)

# Create an empty matrix of the right shape
S_inv = np.zeros_like(matriz_mediana_np.T)

# Fill the diagonal of S_inv with the reciprocals of the non-zero values of S_diag
S_inv[:S_diag.shape[0], :S_diag.shape[0]] = np.linalg.inv(S_diag)

# Calculate the pseudoinverse of the original matrix
X_pseudoinverse = Vt.T @ S_inv @ U.T

# Calculate beta
beta = X_pseudoinverse @ y_centered

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Calculate the predictions of y using beta
y_pred = matriz_mediana_np @ beta + mean_y

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the real values of y
ax.scatter(U[:, 0], U[:, 1], y, color='b')

# Create a grid over the range of U values
grid_x, grid_y = np.meshgrid(np.linspace(min(U[:, 0]), max(U[:, 0]), num=100),
                             np.linspace(min(U[:, 1]), max(U[:, 1]), num=100))

# Calculate the corresponding z values for the grid
grid_z = beta[0] + grid_x * beta[1] + grid_y * beta[2] + mean_y

# Plot the hyperplane
ax.plot_surface(grid_x, grid_y, grid_z, color='r', alpha=0.5)

# Set the labels
ax.set_xlabel('Autovector 1')
ax.set_ylabel('Autovector 2')
ax.set_zlabel('y')

# Show the plot
plt.show()