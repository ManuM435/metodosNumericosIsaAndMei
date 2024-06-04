import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import auxiliar as aux

# Load the images and convert them into vectors
images = []
for filename in os.listdir('datasets_imgs'):
    img = Image.open(os.path.join('datasets_imgs', filename))
    img_vector = np.array(img).flatten()
    images.append(img_vector)

# Stack the vectors to form a data matrix
data_matrix = np.vstack(images)

# Perform a singular value decomposition (SVD) on the data matrix
U, S, Vt = np.linalg.svd(data_matrix, full_matrices=False)

# Get the dimension of the images (assuming they are square)
p = int(np.sqrt(images[0].shape[0]))







# Reconstruct the images from the low-dimensional representation and visualize them
# dimensions = [2, 10, 24]  # The dimensions to use for reconstruction
# for i in range(len(data_matrix)):
#     plt.figure(figsize=(10, 10))
#     for j, d in enumerate(dimensions):
#         U_reduced = U[:, :d]
#         S_reduced = np.diag(S[:d])
#         Vt_reduced = Vt[:d, :]
#         matrix_reconstructed = U_reduced @ S_reduced @ Vt_reduced

#         # Visualize the reconstructed image
#         img_reconstructed = matrix_reconstructed[i].reshape((p, p))  # Assuming the images are size p x p
#         plt.subplot(1, 4, j+1)
#         plt.imshow(img_reconstructed, cmap='gray')
#         plt.title(f'Reconstructed with d={d} dimensions')
#     plt.suptitle(f'Image {i} Reconstruction')
#     plt.show()

# # Initialize the similarity matrix
# similarity_matrix = np.zeros((len(images), len(images)))

# for d in dimensions:
#     # Perform a reduced SVD on the data matrix
#     U_reduced = U[:, :d]
#     S_reduced = np.diag(S[:d])
#     Vt_reduced = Vt[:d, :]
#     data_matrix_reduced = U_reduced @ S_reduced @ Vt_reduced

#     # Initialize the similarity matrix
#     similarity_matrix = aux.eucledian_distance(2000, data_matrix_reduced)

#     # Visualize the similarity matrix
#     plt.figure(figsize=(10, 10))
#     plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
#     plt.colorbar(label='Similarity')
#     plt.title(f'Similarity Matrix (d={d})')
#     plt.show()

# # Set the dimension to 24
# d = 24

# # Perform a reduced SVD on the data matrix
# U_reduced = U[:, :d]
# S_reduced = np.diag(S[:d])
# Vt_reduced = Vt[:d, :]
# data_matrix_reduced = U_reduced @ S_reduced @ Vt_reduced

# # Initialize a figure
# fig, axs = plt.subplots(4, 5, figsize=(10, 10))

# # Flatten the axes
# axs = axs.flatten()

# # Loop over each image
# for i in range(len(data_matrix_reduced)):
#     # Reconstruct the image
#     img_reconstructed = data_matrix_reduced[i].reshape((p, p))  # Assuming the images are size p x p

#     # Add the image to the plot
#     axs[i].imshow(img_reconstructed, cmap='gray')
#     axs[i].axis('off')  # Hide the axes

# # Remove the last unused subplot
# fig.delaxes(axs[-1])

# # Show the plot
# plt.suptitle('All Images at d=24')
# plt.show()










# 2.4


def frobeniusNorm(X):
    norm = 0
    for i in range(len(X)):
        for j in range(len(X[0])):
            norm += X[i][j] ** 2
    return np.sqrt(norm)


def redimensionalizerInator(image, dimension):
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    U_reduced = U[:, :dimension]
    S_reduced = np.diag(S[:dimension])
    Vt_reduced = Vt[:dimension, :]
    image_reduced = U_reduced @ S_reduced @ Vt_reduced
    return image_reduced


def frobeniusError(OriginalMatrix, dimension):
    reduced_matrix = redimensionalizerInator(OriginalMatrix, dimension)
    error = frobeniusNorm(OriginalMatrix - reduced_matrix)
    return error

def frobeniusMaximumError(image_list, dimension):
    errors = []
    for image in image_list:
        error = frobeniusError(image, dimension)
        errors.append(error)
    return min(errors)

def errorByDimensions(image_list, max_dimension):
    errors = []
    for i in range(1, max_dimension + 1):
        error = frobeniusMaximumError(image_list, i)
        errors.append(error)
    return errors

# Load the images
images2 = []
for filename in os.listdir('datasets_imgs'):
    img = Image.open(os.path.join('datasets_imgs', filename))
    images2.append(np.array(img))

# Calculate the maximum errors for each dimension
max_dimension = 24
max_errors = errorByDimensions(images2, max_dimension)

# Plot the maximum errors
plt.figure(figsize=(10, 7))
plt.plot(range(1, max_dimension + 1), max_errors, marker='o')
plt.xlabel('Dimensions')
plt.ylabel('Maximum Frobenius Error')
plt.title('Maximum Frobenius Error by Dimensions')
plt.grid()
plt.yscale('log')  # Set the y-axis scale to logarithmic
plt.show()



