import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics
import auxiliar as aux
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D

# Un par de avisos antes de correr el codigo sin repasarlo (estos avisos se repiten en el codigo de abajo tambien)
# Correr el codigo apenas luego de abrirlo va a tardar mucho debido a las lineas 100-103, y sus plots debajo
# Lo mismo para comentar either la linea 300 o las 330-342 para evitar mas de 100 pop-ups de graficos

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


def pseudoinvCalculatorInator(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_inv = np.diag([1/S if S != 0 else 0 for S in S])
    X_pseudoinverse = Vt.T @ S_inv @ U.T
    return X_pseudoinverse

# Calcular Matriz Mediana
matriz_mediana = np.array(matriz_datos) - np.mean(matriz_datos)

# Descomposición SVD
U, S, Vt = np.linalg.svd(matriz_mediana, full_matrices=False)



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






# para graficar las matrices de similaridad con diferentes dimensiones

# def reducedSimilarityMatrixer(sigma, originalMatrix, dimensions):
#     U, S, Vt = np.linalg.svd(originalMatrix, full_matrices=False)
#     U_reduced = U[:, :dimensions]
#     S_reduced = np.diag(S[:dimensions])
#     Vt_reduced = Vt[:dimensions, :]
#     matrix_reduced = U_reduced @ S_reduced @ Vt_reduced
#     similarity_matrix_reduced = aux.eucledian_distance(sigma, matrix_reduced)
#     return similarity_matrix_reduced

# # Plot similarity matrices with different dimensions
# dimensions_list = [2, 6, 10, 4]
# theSigma = 50

# fig, axs = plt.subplots(2, 2, figsize=(10, 10))


# # Aviso Importante: Correr estas funciones puede tardar mucho, si se desea correr el codigo rapido, comentar estas 4 lineas y sus plots
# similarity_matrix_d2 = reducedSimilarityMatrixer(theSigma, matriz_mediana, dimensions_list[0])
# similarity_matrix_d6 = reducedSimilarityMatrixer(theSigma, matriz_mediana, dimensions_list[1])
# similarity_matrix_d10 = reducedSimilarityMatrixer(theSigma, matriz_mediana, dimensions_list[2])
# similarity_matrix_d106 = aux.eucledian_distance(theSigma, matriz_mediana)


# axs[0, 0].imshow(similarity_matrix_d2, cmap='hot', interpolation='none')
# axs[0, 0].set_title('Similariy Matrix with 2 Dimensions')
# axs[0, 0].set_xlabel('Sample Number')
# axs[0, 0].set_ylabel('Sample Number')

# axs[0, 1].imshow(similarity_matrix_d6, cmap='hot', interpolation='none')
# axs[0, 1].set_title('Similariy Matrix with 6 Dimensions')
# axs[0, 1].set_xlabel('Sample Number')
# axs[0, 1].set_ylabel('Sample Number')

# axs[1, 0].imshow(similarity_matrix_d10, cmap='hot', interpolation='none')
# axs[1, 0].set_title('Similariy Matrix with 10 Dimensions')
# axs[1, 0].set_xlabel('Sample Number')
# axs[1, 0].set_ylabel('Sample Number')

# axs[1, 1].imshow(similarity_matrix_d106, cmap='hot', interpolation='none')
# axs[1, 1].set_title('Similariy Matrix with 106 Dimensions')
# axs[1, 1].set_xlabel('Sample Number')
# axs[1, 1].set_ylabel('Sample Number')

# plt.tight_layout()
# plt.show()


# si este de arriba no sirve, usar este de abajo que es mas rustico pero ya comprobado de sobra que funciona
# (PS: que tarde mucho no significa que no sirva...)

# U_reduced2 = U[:, :2]
# S_reduced2 = np.diag(S[:2])
# Vt_reduced2 = Vt[:2, :]

# # Reducir la dimensionalidad a seis
# U_reduced6 = U[:, :6]
# S_reduced6 = np.diag(S[:6])
# Vt_reduced6 = Vt[:6, :]

# # Reducir la dimensionalidad a diez
# U_reduced10 = U[:, :10]
# S_reduced10 = np.diag(S[:10])
# Vt_reduced10 = Vt[:10, :]


# matriz_reconstruida2 = U_reduced2 @ S_reduced2 @ Vt_reduced2
# matriz_reconstruida6 = U_reduced6 @ S_reduced6 @ Vt_reduced6
# matriz_reconstruida10 = U_reduced10 @ S_reduced10 @ Vt_reduced10

# sigm = 20

# matriz_simil2 = aux.eucledian_distance(sigm, matriz_reconstruida2)
# matriz_simil6 = aux.eucledian_distance(sigm, matriz_reconstruida6)
# matriz_simil10 = aux.eucledian_distance(sigm, matriz_reconstruida10)
# matriz_simil106 = aux.eucledian_distance(sigm, matriz_mediana)

# # Create a 2x2 figure with the 4 matriz_simil subplots
# fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# # Plot matriz_simil2
# axs[0, 0].imshow(matriz_simil2, cmap='hot', interpolation='nearest')
# axs[0, 0].set_title('Similarity Matrix (d=2)')
# axs[0, 0].set_xlabel('Sample Number')
# axs[0, 0].set_ylabel('Sample Number')

# # Plot matriz_simil6
# axs[0, 1].imshow(matriz_simil6, cmap='hot', interpolation='nearest')
# axs[0, 1].set_title('Similarity Matrix (d=6)')
# axs[0, 1].set_xlabel('Sample Number')
# axs[0, 1].set_ylabel('Sample Number')

# # Plot matriz_simil10
# axs[1, 0].imshow(matriz_simil10, cmap='hot', interpolation='nearest')
# axs[1, 0].set_title('Similarity Matrix (d=10)')
# axs[1, 0].set_xlabel('Sample Number')
# axs[1, 0].set_ylabel('Sample Number')

# # Plot matriz_simil106
# axs[1, 1].imshow(matriz_simil106, cmap='hot', interpolation='nearest')
# axs[1, 1].set_title('Similarity Matrix (Original, d=106)')
# axs[1, 1].set_xlabel('Sample Number')
# axs[1, 1].set_ylabel('Sample Number')

# # Adjust the spacing between subplots
# plt.tight_layout()

# # Show the figure
# plt.show()

# hasta aca todo este bloque comentado era el codigo mas rustico original para graficar las de similaridad
# que ya da miedo sacarlo por si el otro llega a tener un error...







# # Grafico de Significancia de Dimensiones para SVD
# plt.figure(figsize=(10, 5))
# plt.bar(range(1, len(S) + 1), S)
# plt.title('Valores singulares de la Matriz')
# plt.xlabel('Índice del valor singular')
# plt.ylabel('Valor del valor singular')
# plt.show()




# # Grafico de valores absolutos de primer autovector
# # Generar el diccionario
# dict_autovector = {i: abs(v) for i, v in enumerate(Vt[0])}

# # Ordenar el diccionario de mayor a menor
# dict_autovector_ordenado = dict(sorted(dict_autovector.items(), key=lambda item: item[1], reverse=True))

# # Crear un gráfico de barras de los valores
# plt.bar(dict_autovector_ordenado.keys(), dict_autovector_ordenado.values())
# plt.title('Valores absolutos de cada columna en primera fila de Vt')
# plt.show()





# # Generar el gráfico de dispersión

# Z, vtAux = aux.PCAinator(matriz_mediana, 2)
# plt.figure(figsize=(10, 5))
# plt.scatter(Z[:, 0], Z[:, 1])
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




# Cuadrados minimos

# Create a 3D plot

# Primero centrar la etiqueta
labels = y - np.mean(y)

def pseudoinvCalculatorInator(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_inv = np.diag([1/S if S != 0 else 0 for S in S])
    X_pseudoinverse = Vt.T @ S_inv @ U.T
    return X_pseudoinverse


reduci = aux.PCAinator(matriz_mediana, 2)
pseudi = pseudoinvCalculatorInator(reduci)
betita = pseudi @ labels
print(betita)