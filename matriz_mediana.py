import csv
import statistics
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

promedio_columnas = []
with open('ISA\dataset.csv', 'r') as f:
    dataset = list(csv.reader(f))
    for i in range(1, len(dataset[0])):  # Comienza desde la segunda columna
        columna = [float(fila[i]) for fila in dataset[1:]]  # Comienza desde la segunda fila
        columna_sin_none = [x for x in columna if x is not None]
        promedio_columnas.append(statistics.mean(columna_sin_none))
    
    matriz_datos = []
    for fila in dataset[1:]:  # Comienza desde la segunda fila
        matriz_datos.append([float(dato) for dato in fila[1:]])  # Comienza desde la segunda columna

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

# Mostrar la matriz redimensionada como una imagen
fig, ax = plt.subplots(figsize=(6, 6)) 
ax.imshow(matriz_mediana, cmap='hot', interpolation='none')
plt.show()

primera_columna = [fila[0] for fila in matriz_mediana]
print(primera_columna)