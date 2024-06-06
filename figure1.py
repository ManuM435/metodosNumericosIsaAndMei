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

    return matrix_reduced, Vt[:Dimensions, :]

# Calcular Matriz Mediana
matriz_mediana = np.array(matriz_datos) - np.mean(matriz_datos)

# Esto es para graficar la matriz original

matriz_promedio_columnas = []
for fila in matriz_datos:
    matriz_promedio_columnas.append([dato - promedio_columnas[i] for i, dato in enumerate(fila)])

matriz_mediana = [[matriz_datos[i][j] - promedio_columnas[j] for j in range(len(matriz_datos[0]))] for i in range(len(matriz_datos))]

# primera_columna_mediana = [fila[0] for fila in matriz_mediana]
# print(primera_columna_mediana)
# Calcular el factor de zoom para cada dimensión
zoom_factor = [2000 / len(matriz_mediana), 2000 / len(matriz_mediana[0])]

# Redimensionar la matriz con interpolación de orden cero
matriz_mediana_resized = zoom(matriz_mediana, zoom_factor, order=0)

# # Mostrar la matriz redimensionada como una imagen
# fig, ax = plt.subplots(figsize=(6, 6)) 
# ax.imshow(matriz_mediana, cmap='hot', interpolation='none')
# plt.show()

# primera_columna = [fila[0] for fila in matriz_mediana]
# print(primera_columna)

#Esto de arriba es de la matriz original


# Calcular la matriz de similaridad
# similarity_matrix = aux.eucledian_distance(20, vectores)

# #Grafica la matriz de similaridad como una imagen
# plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')
# plt.title('Matriz de Similaridad')
# plt.show()


# # Calcular la matriz de similaridad centrada
# similarity_matrix_centered = aux.eucledian_distance(15, matriz_mediana)

# #Grafica la matriz de similaridad centrada como una imagen
# plt.imshow(similarity_matrix_centered, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')
# plt.title('Matriz de Similaridad Centrada')
# plt.show()





# # Down Here van Valores Singulares (Barritas que muestran la significancia de los valores de d) y PCA (Dispersion 3D) 

 # Descomposición SVD
U, S, Vt = np.linalg.svd(matriz_mediana, full_matrices=False)

# # Grafico de Significancia de Dimensiones
# plt.figure(figsize=(10, 5))
# plt.bar(range(1, len(S) + 1), S)
# plt.title('Valores singulares de la matriz S')
# plt.xlabel('Índice del valor singular')
# plt.ylabel('Valor del valor singular')
# plt.show()

# # Calculate the total sum of eigen values in S
# total_sum = sum(S)

# # Calculate the sum of the first two eigen values in S
# first_two_sum = sum(S[:2])

# # Calculate the percentage represented by the first two eigen values
# percentage = (first_two_sum / total_sum) * 100

# # Print the percentage
# print(f"The first two eigen values represent {percentage:.2f}% of the total sum.")


# # Para PCA y Dispersion 2D
# U_reducida = U[:, :2]
# S_reducida = np.diag(S[:2])
# Vt_reducida = Vt[:2, :]


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



# # Aca graficamos la matriz de similaridad con distintos valores de d

# # Generar el diccionario
# dict_autovector = {i: v for i, v in enumerate(Vt[0])}

# # Ordenar el diccionario de mayor a menor
# dict_autovector_ordenado = dict(sorted(dict_autovector.items(), key=lambda item: item[1], reverse=True))

# # Imprimir el diccionario ordenado
# for key, value in dict_autovector_ordenado.items():
#     print(f'Índice: {key}, Valor: {value}')

# # Crear un gráfico de barras de los valores
# plt.bar(dict_autovector_ordenado.keys(), dict_autovector_ordenado.values())
# plt.title('Valores del primer autovector')
# plt.show()

# Reducir la dimensionalidad a dos

U_reduced2 = U[:, :2]
S_reduced2 = np.diag(S[:2])
Vt_reduced2 = Vt[:2, :]

# Reducir la dimensionalidad a seis
U_reduced6 = U[:, :6]
S_reduced6 = np.diag(S[:6])
Vt_reduced6 = Vt[:6, :]

# Reducir la dimensionalidad a diez
U_reduced10 = U[:, :10]
S_reduced10 = np.diag(S[:10])
Vt_reduced10 = Vt[:10, :]


matriz_reconstruida2 = U_reduced2 @ S_reduced2 @ Vt_reduced2
matriz_reconstruida6 = U_reduced6 @ S_reduced6 @ Vt_reduced6
matriz_reconstruida10 = U_reduced10 @ S_reduced10 @ Vt_reduced10

sigm = 20

matriz_simil2 = aux.eucledian_distance(sigm, matriz_reconstruida2)
matriz_simil6 = aux.eucledian_distance(sigm, matriz_reconstruida6)
matriz_simil10 = aux.eucledian_distance(sigm, matriz_reconstruida10)
matriz_simil106 = aux.eucledian_distance(sigm, matriz_mediana)

# Create a 2x2 figure with the 4 matriz_simil subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Set the spacing between subplots
fig.subplots_adjust(hspace=0.5, wspace=0.5)

# Plot matriz_simil2
axs[0, 0].imshow(matriz_simil2, cmap='hot', interpolation='nearest')
axs[0, 0].set_title('Matriz Similaridad (d=2)')
axs[0, 0].set_xlabel('Sample Number')
axs[0, 0].set_ylabel('Sample Number')

# Plot matriz_simil6
axs[0, 1].imshow(matriz_simil6, cmap='hot', interpolation='nearest')
axs[0, 1].set_title('Matriz Similaridad (d=6)')
axs[0, 1].set_xlabel('Sample Number')
axs[0, 1].set_ylabel('Sample Number')

# Plot matriz_simil10
axs[1, 0].imshow(matriz_simil10, cmap='hot', interpolation='nearest')
axs[1, 0].set_title('Matriz Similaridad (d=2)')
axs[1, 0].set_xlabel('Sample Number')
axs[1, 0].set_ylabel('Sample Number')

# Plot matriz_simil106
axs[1, 1].imshow(matriz_simil106, cmap='hot', interpolation='nearest')
axs[1, 1].set_title('Matriz Similaridad (Original, d=106)')
axs[1, 1].set_xlabel('Sample Number')
axs[1, 1].set_ylabel('Sample Number')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()





# #Cuadrados minimos

# # # Create a dictionary to store the order of values in beta
# # beta_order = {}

# # # Iterate over the values in beta and store their order in the dictionary
# # for i, value in enumerate(beta):
# #     beta_order[i+1] = abs(value)

# # # Sort the dictionary by the absolute values of the values in descending order
# # beta_order = {k: v for k, v in sorted(beta_order.items(), key=lambda item: item[1], reverse=True)}

# # # Print the dictionary
# # print(beta_order)




# # Create a 3D plot

# datos = matriz_mediana
# labels = y - np.mean(y)


# def pseudoinvCalculatorInator(X):
#     U, S, Vt = np.linalg.svd(X, full_matrices=False)
#     S_inv = np.diag([1/S if S != 0 else 0 for S in S])
#     X_pseudoinverse = Vt.T @ S_inv @ U.T
#     return X_pseudoinverse


# def hyperPlanePlotInator(data, labels, dimensions):
#     reduced_mat = PCAinator(data, dimensions)
#     pseudoinverse= pseudoinvCalculatorInator(reduced_mat)
#     beta = pseudoinverse @ labels

#     grid_x, grid_y = np.meshgrid(np.linspace(min(reduced_mat[:, 0]), max(reduced_mat[:, 0]), 50),
#                      np.linspace(min(reduced_mat[:, 1]), max(reduced_mat[:, 1]), 50))
#     grid_z =  grid_x * beta[0] + grid_y * beta[1]

#     # fig = plt.figure(figsize=(10, 7))
#     # ax = fig.add_subplot(111, projection='3d')
#     # scatter = ax.scatter(reduced_mat[:, 0], reduced_mat[:, 1], labels, color='b')
#     # plt.colorbar(scatter)

#     # ax.plot_surface(grid_x, grid_y, grid_z, color='r', alpha=0.5)
#     # ax.set_xlabel('AV 1')
#     # ax.set_ylabel('AV 2')
#     # ax.set_zlabel('Labels')
#     # ax.set_title('el grafiquinho del hiperplaninho')
#     # plt.show()

    
#     error = np.linalg.norm(reduced_mat @ beta - labels)
#     return error


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