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
dimensions = [2, 10, 24]  # The dimensions to use for reconstruction
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

# Initialize the similarity matrix
similarity_matrix = np.zeros((len(images), len(images)))

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

# Set the dimension to 24
d = 24

# Perform a reduced SVD on the data matrix
U_reduced = U[:, :d]
S_reduced = np.diag(S[:d])
Vt_reduced = Vt[:d, :]
data_matrix_reduced = U_reduced @ S_reduced @ Vt_reduced

# Initialize a figure
fig, axs = plt.subplots(4, 5, figsize=(10, 10))

# Flatten the axes
axs = axs.flatten()

# Loop over each image
for i in range(len(data_matrix_reduced)):
    # Reconstruct the image
    img_reconstructed = data_matrix_reduced[i].reshape((p, p))  # Assuming the images are size p x p

    # Add the image to the plot
    axs[i].imshow(img_reconstructed, cmap='gray')
    axs[i].axis('off')  # Hide the axes

# Remove the last unused subplot
fig.delaxes(axs[-1])

# Show the plot
plt.suptitle('All Images at d=24')
plt.show()

import numpy as np
import os
from PIL import Image

# Load the images
image_files = os.listdir('/Users/isabelcastaneda/Documents/GitHub/metodosNumericosIsaAndMei/datasets_imgs_02')
images = [np.array(Image.open('/Users/isabelcastaneda/Documents/GitHub/metodosNumericosIsaAndMei/datasets_imgs_02/' + file)) for file in image_files]

# Convert the list of images to a data matrix
data_matrix_2 = np.array(images).reshape(len(images), -1)

# Apply SVD
U, S, Vt = np.linalg.svd(data_matrix_2, full_matrices=False)

# Calculate the total variance of the original data matrix
total_variance = np.linalg.norm(data_matrix_2, 'fro')**2

# Initialize the sum of squared singular values up to d
sum_of_squared_singular_values_up_to_d = 0

# Loop over the singular values
for d in range(len(S)):
    # Add the square of the current singular value
    sum_of_squared_singular_values_up_to_d += S[d]**2

    # Calculate the proportion of the total variance
    proportion_of_total_variance = sum_of_squared_singular_values_up_to_d / total_variance

    # If the proportion of the total variance is greater than 0.9 (i.e., the error is less than 10%),
    # then we have found the minimum number of dimensions
    if proportion_of_total_variance >= 0.9:
        break

print(f"The minimum number of dimensions is {d+1}")



