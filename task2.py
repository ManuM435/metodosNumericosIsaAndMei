import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import auxiliar as aux
from auxiliar import PCAinator

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
dimensions = [24, 16, 10, 4]  # The dimensions to use for reconstruction
amountofsamples = 5
# Create a figure
fig, axs = plt.subplots(amountofsamples, len(dimensions), figsize=(14, 14))

for i in range(amountofsamples):
    for j, d in enumerate(dimensions):
        U_reduced = U[:, :d]
        S_reduced = np.diag(S[:d])
        Vt_reduced = Vt[:d, :]
        matrix_reconstructed = U_reduced @ S_reduced @ Vt_reduced

        # Visualize the reconstructed image
        img_reconstructed = matrix_reconstructed[i+4].reshape((p, p))  # Assuming the images are size p x p
        axs[i, j].imshow(img_reconstructed, cmap='gray')
        axs[i, j].axis('off')  # Turn off the axes

        # Add a label to the first subplot of each column
        if i == 0:
            axs[i, j].set_title(f'd={d}')

# Add a title to the figure
fig.suptitle('Image Reconstruction with Different Dimensions')

# Adjust the space between subplots

plt.show()

# Add a title to the figure
fig.suptitle('Image Reconstruction with Different Dimensions')
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Create a figure
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

for i, d in enumerate(dimensions):
    # Perform a reduced SVD on the data matrix
    U_reduced = U[:, :d]
    S_reduced = np.diag(S[:d])
    Vt_reduced = Vt[:d, :]
    data_matrix_reduced = U_reduced @ S_reduced @ Vt_reduced

    # Initialize the similarity matrix
    similarity_matrix = aux.eucledian_distance(2000, data_matrix_reduced)

    # Visualize the similarity matrix
    ax = axs[i//2, i%2]
    im = ax.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    ax.set_title(f'Similarity Matrix (d={d})')

    # Set x and y labels
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Sample Number')

    # Set x and y ticks to be integers
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax, label='Similarity')

plt.tight_layout(h_pad=0.5)  # Adjust the horizontal space between subplots
plt.show()
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










# # 2.4


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


def frobeniusRelativeError(OriginalMatrix, dimension):
    reduced_matrix = redimensionalizerInator(OriginalMatrix, dimension)
    error = frobeniusNorm(OriginalMatrix - reduced_matrix) / frobeniusNorm(OriginalMatrix)
    return error

def frobeniusMaximumError(image_list, dimension):
    errors = []
    for image in image_list:
        error = frobeniusRelativeError(image, dimension)
        errors.append(error)
    return min(errors)

def errorByDimensions(image_list, max_dimension):
    errors = []
    for i in range(1, max_dimension + 1):
        error = frobeniusMaximumError(image_list, i)
        errors.append(error)
    return errors

# Load the images
images2, imagesforMat2 = [], []

for filename in os.listdir('datasets_imgs_02'):
    img2 = Image.open(os.path.join('datasets_imgs_02', filename))
    images2.append(np.array(img2))

for filename in os.listdir('datasets_imgs_02'):
    img2m = Image.open(os.path.join('datasets_imgs_02', filename))
    img_vector2 = np.array(img2m).flatten()
    imagesforMat2.append(img_vector2)

# Stack the vectors to form a data matrix
data_matrix2 = np.vstack(imagesforMat2)



# Calculate the maximum errors for each dimension
max_dimension = 24
max_errors = errorByDimensions(images2, max_dimension)

# Plot the maximum errors
plt.figure(figsize=(10, 7))
plt.fill_between(range(1, max_dimension + 1), max_errors, color='skyblue', alpha=0.4)
plt.plot(range(1, max_dimension + 1), max_errors, marker='o', color='blue')
plt.axhline(y=0.1, color='r', linestyle='--')  # Add red dotted line at y=0.1
plt.xlabel('Dimensions')
plt.ylabel('Maximum Frobenius Relative Error')
plt.title('Maximum Frobenius Error by Dimensions')
plt.xticks(range(1, max_dimension + 1))  # Set x-axis ticks for every whole value between 1 and 24
plt.xlim(1, max_dimension)  # Set the x-axis limit to remove empty space on the right
plt.ylim(0, 0.6)  # Set the y-axis limit to show the full range of errors
plt.grid(axis='x')  # Add vertical grid lines
plt.show()




ladimension = 7

# Aprender una representacion (plotear Vt)

# Get the shape of the original images
img_shape = np.array(img).shape

# Plot the first three eigenvectors
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i in range(3):
    # Reshape the eigenvector to have the same shape as the original images
    eigenvector_img = Vt[i].reshape(img_shape)

    # Normalize the eigenvector image for better visualization
    eigenvector_img = (eigenvector_img - np.min(eigenvector_img)) / (np.max(eigenvector_img) - np.min(eigenvector_img))

    # Plot the eigenvector image
    axs[i].imshow(eigenvector_img, cmap='gray')
    axs[i].set_title(f'Autovector {i+1}')
    axs[i].axis('off')

fig.suptitle('Primeros tres autovectores de Vt representados como imágenes')

plt.show()