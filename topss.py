import numpy as np

import matplotlib.pyplot as plt

# Generate random matrices
matrix1 = np.random.rand(3, 3)
matrix2 = np.random.rand(3, 3)
matrix3 = np.random.rand(3, 3)
matrix4 = np.random.rand(3, 3)

# Create a 2x2 subplot
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# Plot the matrices
axs[0, 0].imshow(matrix1)
axs[0, 0].set_title('Matrix 1')
axs[0, 0].set_xlabel('Sample Number')
axs[0, 0].set_ylabel('Sample Number')

axs[0, 1].imshow(matrix2)
axs[0, 1].set_title('Matrix 2')
axs[0, 1].set_xlabel('Sample Number')
axs[0, 1].set_ylabel('Sample Number')

axs[1, 0].imshow(matrix3)
axs[1, 0].set_title('Matrix 3')
axs[1, 0].set_xlabel('Sample Number')
axs[1, 0].set_ylabel('Sample Number')

axs[1, 1].imshow(matrix4)
axs[1, 1].set_title('Matrix 4')
axs[1, 1].set_xlabel('Sample Number')
axs[1, 1].set_ylabel('Sample Number')

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()