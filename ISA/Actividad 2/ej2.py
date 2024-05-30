import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the images and convert them into vectors
images = []
for filename in os.listdir('/Users/isabelcastaneda/Documents/GitHub/metodosNumericosIsaAndMei/ISA/Actividad 2/datasets_imgs'):
    img = Image.open(os.path.join('/Users/isabelcastaneda/Documents/GitHub/metodosNumericosIsaAndMei/ISA/Actividad 2/datasets_imgs', filename))
    img_vector = np.array(img).flatten()
    images.append(img_vector)

# Stack the vectors to form a data matrix
data_matrix = np.vstack(images)

# Perform a singular value decomposition (SVD) on the data matrix
U, S, Vt = np.linalg.svd(data_matrix, full_matrices=False)

# Get the dimension of the images (assuming they are square)
p = int(np.sqrt(images[0].shape[0]))

# Reconstruct the images from the low-dimensional representation and visualize them
dimensions = [9, 10, 11, 12, 13, 24]  # The dimensions to use for reconstruction
for i in range(len(data_matrix)):
    plt.figure(figsize=(10, 10))
    for j, d in enumerate(dimensions):
        U_reduced = U[:, :d]
        S_reduced = np.diag(S[:d])
        Vt_reduced = Vt[:d, :]
        matrix_reconstructed = U_reduced @ S_reduced @ Vt_reduced

        # Visualize the reconstructed image
        img_reconstructed = matrix_reconstructed[i].reshape((p, p))  # Assuming the images are size p x p
        plt.subplot(2, 3, j+1)
        plt.imshow(img_reconstructed, cmap='gray')
        plt.title(f'Reconstructed with d={d} dimensions')
    plt.suptitle(f'Image {i} Reconstruction')
    plt.show()