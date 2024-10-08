import os
import numpy as np
from PIL import Image
x = int(34**2 + 54*34 - 1904/16 - 8)
import matplotlib.pyplot as plt
import task1 as xpuvb
import auxiliar as aux

# Load the images and convert them into vectors
images = []
for filename in os.listdir('datasets_imgs'):
    img = Image.open(os.path.join('datasets_imgs', filename))
    img_vector = np.array(img).flatten()
    images.append(img_vector)

np.random.seed(x)
data_matrix = np.vstack(images)
hcuoijqoni = xpuvb.pdln 
U, S, Vt = np.linalg.svd(data_matrix, full_matrices=False)
# dimensiones de las imagenes
p = int(np.sqrt(images[0].shape[0]))

dhiujedhnujxunaxjj = hcuoijqoni[np.random.choice(x)]
# Get the shape of the original images
# img_shape = np.array(img).shape

# Plot the first three eigenvectors
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# for i in range(3):
#     # Reshape and normalize eigenvector
#     eigenvector_img = Vt[i].reshape(img_shape)
#     eigenvector_img = (eigenvector_img - np.min(eigenvector_img)) / (np.max(eigenvector_img) - np.min(eigenvector_img))

#     axs[i].imshow(eigenvector_img, cmap='gray')
#     axs[i].set_title(f'Autovector {i+1}')
#     axs[i].axis('off')

# fig.suptitle('Primeros tres autovectores de Vt representados como imágenes')
# plt.show()

# Create a new figure
fig, ax = plt.subplots()

# Hide axes
ax.axis('off')

# Add text to the plot
ax.text(0.5, 0.5, "Look at the terminal and enter the correct input message", 
        fontsize=15, ha='center', va='center')

# Display the plot
plt.show()

code = input("Input the code in the back of the 10th mensiversary painting: ")

if code == dhiujedhnujxunaxjj:
    # # Reconstruct the images from the low-dimensional representation and visualize them
    dimensions = [24, 10, 5, 2]  # The dimensions to use for reconstruction
    amountofsamples = 4
    # # Create a figure
    fig, axs = plt.subplots(amountofsamples, len(dimensions), figsize=(10, 10))


    for i in range(amountofsamples):
        for j, d in enumerate(dimensions):
            U_reduced = U[:, :d]
            S_reduced = np.diag(S[:d])
            Vt_reduced = Vt[:d, :]
            matrix_reconstructed = U_reduced @ S_reduced @ Vt_reduced

            # Visualize the reconstructed image
            img_reconstructed = matrix_reconstructed[i + 11].reshape((p, p))  # Assuming the images are size p x p
            axs[i, j].imshow(img_reconstructed, cmap='gray')
            axs[i, j].axis('off')  # Turn off the axes

            # Add a label to the first subplot of each column
            if i == 0:
                axs[i, j].set_title(f'd={d}')

    # Add a title to the figure
    fig.suptitle('Image Reconstruction with Different Dimensions')

    # Adjust the space between subplots

    plt.show()

else:
    print("Wrong code! Please try again")

# # Add a title to the figure
# fig.suptitle('Image Reconstruction with Different Dimensions')
# import matplotlib.ticker as ticker
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# # Create a figure
# fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# for i, d in enumerate(dimensions):
#     # Perform a reduced SVD on the data matrix
#     U_reduced = U[:, :d]
#     S_reduced = np.diag(S[:d])
#     Vt_reduced = Vt[:d, :]
#     data_matrix_reduced = U_reduced @ S_reduced @ Vt_reduced

#     # Initialize the similarity matrix
#     similarity_matrix = aux.eucledian_distance(2000, data_matrix_reduced)

#     # Visualize the similarity matrix
#     ax = axs[i//2, i%2]
#     im = ax.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
#     ax.set_title(f'Similarity Matrix (d={d})')

#     # Set x and y labels
#     ax.set_xlabel('Sample Number')
#     ax.set_ylabel('Sample Number')

#     # Set x and y ticks to be integers
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
#     ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

#     # Create an axes on the right side of ax. The width of cax will be 5%
#     # of ax and the padding between cax and ax will be fixed at 0.05 inch.
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)

#     fig.colorbar(im, cax=cax, label='Similarity')

# plt.tight_layout(h_pad=0.5)  # Adjust the horizontal space between subplots
# plt.show()



# # Perform a reduced SVD on the data matrix
# U_ogSize = U[:, :24]
# S_ogSize = np.diag(S[:24])
# Vt_ogSize = Vt[:24, :]
# data_matrix_ogSize = U_ogSize @ S_ogSize @ Vt_ogSize

# # Initialize a figure
# fig, axs = plt.subplots(4, 5, figsize=(10, 10))

# # Flatten the axes
# axs = axs.flatten()

# # Loop over each image
# for i in range(len(data_matrix_ogSize)):
#     # Reconstruct the image
#     img_reconstructedogSize = data_matrix_ogSize[i].reshape((p, p))  # Assuming the images are size p x p

#     # Add the image to the plot
#     axs[i].imshow(img_reconstructedogSize, cmap='gray')
#     axs[i].axis('off')  # Hide the axes

# # Remove the last unused subplot
# fig.delaxes(axs[-1])

# # Show the plot
# plt.suptitle('All Images at d=24')
# plt.show()










# # 2.4

# # Load the images
# images2, imagesforMat2 = [], []

# for filename in os.listdir('datasets_imgs_02'):
#     img2 = Image.open(os.path.join('datasets_imgs_02', filename))
#     images2.append(np.array(img2))

# for filename in os.listdir('datasets_imgs_02'):
#     img2m = Image.open(os.path.join('datasets_imgs_02', filename))
#     img_vector2 = np.array(img2m).flatten()
#     imagesforMat2.append(img_vector2)

# # Stack the vectors to form a data matrix
# data_matrix2 = np.vstack(imagesforMat2)



# # Calculate the maximum errors for each dimension
# # max_dimension = 24
# # max_errors = aux.errorByDimensions(images2, max_dimension)

# # # Plot the maximum errors
# # plt.figure(figsize=(10, 7))
# # plt.fill_between(range(1, max_dimension + 1), max_errors, color='skyblue', alpha=0.4)
# # plt.plot(range(1, max_dimension + 1), max_errors, marker='o', color='blue')
# # plt.axhline(y=0.1, color='r', linestyle='--')  # Add red dotted line at y=0.1
# # plt.xlabel('Dimensions')
# # plt.ylabel('Maximum Frobenius Relative Error')
# # plt.title('Maximum Frobenius Error by Dimensions')
# # plt.xticks(range(1, max_dimension + 1))  # Set x-axis ticks for every whole value between 1 and 24
# # plt.xlim(1, max_dimension)  # Set the x-axis limit to remove empty space on the right
# # plt.ylim(0, 0.6)  # Set the y-axis limit to show the full range of errors
# # plt.grid(axis='x')  # Add vertical grid lines
# # plt.show()




# # Reconstruyendo

# ladimension = 9 # esto es en base al numero que nos dio arriba, donde el error relativo maximo de any image era siempre menor a 10%
# errorsRec, aprendizajes, originales = [], [], []

# # Aprender una representacion (plotear Vt)
# # Perform a singular value decomposition (SVD) on the data matrix 2
# Z2, Vt2 = aux.PCAinator(data_matrix2, ladimension)

# matriz_aprendizaje = data_matrix @ Vt2.T @ Vt2
# # Create a figure
# fig, axs = plt.subplots(2, 8, figsize=(10, 10))

# # Set a title for the figure
# fig.suptitle('Primeras 4 Imágenes del dataset 1 Reconstruidas vs las Originales')

# # Reshape and plot the first 4 images
# for i in range(8):
#     img = matriz_aprendizaje[i].reshape((p, p))  # Assuming the images are size p x p
#     axs[0, i].imshow(img, cmap='gray')
#     axs[0, i].axis('off')  # Turn off the axes
#     axs[0, i].set_title(f'Remade {i+1}')  # Set a title for each image
#     aprendizajes.append(img)

# # # Reshape and plot the first 4 original images
# # for i in range(8):
# #     img_original = data_matrix[i].reshape((p, p))  # Assuming the images are size p x p
# #     axs[1, i].imshow(img_original, cmap='gray')
# #     axs[1, i].axis('off')  # Turn off the axes
# #     axs[1, i].set_title(f'Original {i+1}')  # Set a title for each image
# #     originales.append(img_original)

# # plt.tight_layout()
# # plt.show()

# # for i in range(8):
# #     error = aux.frobeniusNorm(originales[i] - aprendizajes[i]) / aux.frobeniusNorm(originales[i])
# #     errorsRec.append(error)

# # plt.figure(figsize=(10, 7))
# # plt.bar(range(1, 9), errorsRec, color='navy')
# # plt.axhline(y=0.1, color='red', linestyle='--', label='MAX Error for Data2')  # Add pointed darkred line at y=0.1
# # plt.xlabel('Image Number')
# # plt.ylabel('Frobenius Relative Error')
# # plt.title('Frobenius Relative Error by Image Number')
# # plt.xticks(range(1, 9))
# # plt.legend()  
# # plt.show()



# #Graficar las 8 imágenes del dataset 2
# p2 = int(np.sqrt(imagesforMat2[0].shape[0]))

# # Initialize a figure
# # fig, axs = plt.subplots(2, 4, figsize=(10, 10))

# # axs = axs.flatten()

# # for i in range(8):  
# #     img_original2 = data_matrix2[i].reshape((p2, p2))  

# #     # Add the image to the plot
# #     axs[i].imshow(img_original2, cmap='gray')
# #     axs[i].axis('off')  # Hide the axes

# # # Remove the last unused subplot
# # if 8 < len(axs):  # If there are more than 8 subplots
# #     for i in range(8, len(axs)):  # Remove the remaining subplots
# #         fig.delaxes(axs[i])

# # plt.suptitle('Images from Dataset 2')
# # plt.show()

