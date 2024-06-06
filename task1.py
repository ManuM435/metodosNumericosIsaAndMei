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


def PCAinator(Matriz, Dimensions):
    U, S, Vt = np.linalg.svd(Matriz, full_matrices=False)
    U_reduced = U[:, :Dimensions]
    S_reduced = np.diag(S[:Dimensions])
    matrix_reduced = np.dot(U_reduced, S_reduced)

    return matrix_reduced


def pseudoinvCalculatorInator(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_inv = np.diag([1/S if S != 0 else 0 for S in S])
    X_pseudoinverse = Vt.T @ S_inv @ U.T
    return X_pseudoinverse

# Calcular Matriz Mediana
matriz_mediana = np.array(matriz_datos) - np.mean(matriz_datos)



# # Esto es para graficar la matriz original
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
#Esto de arriba es de la matriz original




# def reducedSimilarityMatrixer(sigma, originalMatrix, dimensions):
#     U, S, Vt = np.linalg.svd(originalMatrix, full_matrices=False)
#     U_reduced = U[:, :dimensions]
#     S_reduced = np.diag(S[:dimensions])
#     Vt_reduced = Vt[:dimensions, :]
#     matrix_reduced = U_reduced @ S_reduced @ Vt_reduced
#     similarity_matrix_reduced = aux.eucledian_distance(sigma, matrix_reduced)
#     return similarity_matrix_reduced

# Plot similarity matrices with different dimensions
dimensions_list = [2, 6, 10, 106]
theSigma = 50

# for dimensions in dimensions_list:
#     similarity_matrix_reduced = reducedSimilarityMatrixer(theSigma, matriz_mediana, dimensions)
#     plt.imshow(similarity_matrix_reduced, cmap='hot', interpolation='nearest')
#     plt.colorbar()
#     plt.xlabel('X Axis')
#     plt.ylabel('Y Axis')
#     plt.title(f'Similarity Matrix with Different Dimensio Dimensions')
#     plt.show()








 # Descomposición SVD
U, S, Vt = np.linalg.svd(matriz_mediana, full_matrices=False)

# Grafico de Significancia de Dimensiones para SVD
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(S) + 1), S)
plt.title('Valores singulares de la matriz S')
plt.xlabel('Índice del valor singular')
plt.ylabel('Valor del valor singular')
plt.show()



# Apendice 1.3

# Generar el diccionario
dict_autovector = {i: abs(v) for i, v in enumerate(Vt[0])}

# Ordenar el diccionario de mayor a menor
dict_autovector_ordenado = dict(sorted(dict_autovector.items(), key=lambda item: item[1], reverse=True))

# Crear un gráfico de barras de los valores
plt.bar(dict_autovector_ordenado.keys(), dict_autovector_ordenado.values())
plt.title('Valores (en módulo) de cada feature en el primer autovector de Vt')
plt.xlabel('Número de la feature')
plt.ylabel('Valor del autovector')
plt.show()





# # Generar el gráfico de dispersión

# Z, vt = PCAinator(matriz_mediana, 2)
# plt.figure(figsize=(10, 5))
# plt.scatter(Z[:, 0], Z[:, 1])
# plt.title('Proyección de los datos en los dos primeros autovectores')
# plt.xlabel('Autovector 1')
# plt.ylabel('Autovector 2')
# plt.show()


# # # Para PCA y Dispersion 3D
# # U_reducida3 = U[:, :3]
# # S_reducida3 = np.diag(S[:3])
# # Vt_reducida3 = Vt[:3, :]


# # # Grafico de Dispersion 3D (PCA)
# # fig = plt.figure(figsize=(10, 5))
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(U_reducida3[:, 0], U_reducida3[:, 1], U_reducida3[:, 2])
# # ax.set_title('Proyección de los datos en los tres primeros autovectores')
# # ax.set_xlabel('Autovector 1')
# # ax.set_ylabel('Autovector 2')
# # ax.set_zlabel('Autovector 3')
# # plt.show()















# Cuadrados minimos

# Create a 3D plot

datos = matriz_mediana
labels = y - np.mean(y)

def pseudoinvCalculatorInator(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_inv = np.diag([1/S if S != 0 else 0 for S in S])
    X_pseudoinverse = Vt.T @ S_inv @ U.T
    return X_pseudoinverse


def hyperPlanePlotInator(data, labels, dimensions):
    reduced_mat = PCAinator(data, dimensions)
    pseudoinverse= pseudoinvCalculatorInator(reduced_mat)
    beta = pseudoinverse @ labels

    grid_x, grid_y = np.meshgrid(np.linspace(min(reduced_mat[:, 0]), max(reduced_mat[:, 0]), 50),
                     np.linspace(min(reduced_mat[:, 1]), max(reduced_mat[:, 1]), 50))
    grid_z =  grid_x * beta[0] + grid_y * beta[1]

    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(reduced_mat[:, 0], reduced_mat[:, 1], labels, color='b')
    # plt.colorbar(scatter)

    # ax.plot_surface(grid_x, grid_y, grid_z, color='r', alpha=0.5)
    # ax.set_xlabel('AV 1')
    # ax.set_ylabel('AV 2')
    # ax.set_zlabel('Labels')
    # ax.set_title('el grafiquinho del hiperplaninho')
    # plt.show()

    
    error = np.linalg.norm(reduced_mat @ beta - labels)
    return error


# hyperPlanePlotInator(matriz_mediana, labels, 2)

# # Trying Different Dimensions

# max_dim = 107

# errors = []
# for i in range(2, max_dim):
#     topa = hyperPlanePlotInator(matriz_mediana, labels, i)
#     errors.append(topa)

# # Plot the errors
# plt.figure(figsize=(10, 7))
# plt.plot(range(2, max_dim), errors, marker='o')
# plt.xlabel('Dimensions')
# plt.ylabel('Error')
# plt.title('Error by Dimensions')
# plt.grid()
# plt.show()