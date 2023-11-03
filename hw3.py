from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

# load the dataset from the provided .npy file, center it around the origin, and return it as a numpy array of floats
def load_and_center_dataset(filename):
    # Your implementation goes here!
    x = np.load(filename) # load the file
    x_center = x - np.mean(x, axis=0) # subtract the mean from each data point to center the data

    return x_center

# calculate and return the covariance matrix of the dataset as a numpy matrix (d × d array)
def get_covariance(dataset):
    # Your implementation goes here!
    n, d = dataset.shape
    # n is number of images
    # d is number of dimensions (column size)
    cov_matrix = np.dot(np.transpose(dataset), dataset) / (n-1)

    return cov_matrix

# perform eigendecomposition on the covariance matrix S and return a diagonal matrix (numpy array) 
# with the largest m eigenvalues on the diagonal in descending order,
# and a matrix (numpy array) with the corresponding eigenvectors as columns.
def get_eig(S, m):
    # Your implementation goes here!
    w, v = eigh(S)
    # largest m eigenvalues of S as a m-by-m diagonal matrix Λ
    top_eig_indices = np.argsort(w)[-m:]

    # Return the largest m eigenvalues of S as a m-by-m diagonal matrix Λ, in descending order,
    # and the corresponding normalized eigenvectors as columns in a d-by-m matrix U.

    # rearrange the output of eigh to get the eigenvalues in decreasing order
    # make sure to keep the eigenvectors in the corresponding columns after that rearrangement
    eig_val = np.diag(w[top_eig_indices][::-1])
    eig_vec = v[:, top_eig_indices][:, ::-1]

    return eig_val, eig_vec

# similar to get_eig, but instead of returning the first m, 
# return all eigenvalues and corresponding eigenvectors in a similar format
# that explain more than a prop proproportion of the variance 
# (specifically, please make sure the eigenvalues are returned in descending order)
def get_eig_prop(S, prop):
    # Your implementation goes here!
    w, v = eigh(S)

    # sort that we have eigenvalues in descending order
    sorted_indices = np.argsort(w)[::-1]
    w = w[sorted_indices]
    v = v[:, sorted_indices]

    # calculate the proportion of variance in the dataset explained by the ith eigenvector
    total_variance = sum(w)
    prop_variance = w / total_variance

    index = np.where(prop_variance > prop)[0]

    # return the eigenvalues as a diagonal matrix, and the corresponding eigenvectors as columns in a matrix
    all_eig_val = np.diag(w[index])
    all_eig_vec = v[:, index]

    # return the diagonal matrix of eigenvalues first, then the eigenvectors in corresponding columns.
    return all_eig_val, all_eig_vec

# project each d × 1 image into your m-dimensional subspace (spanned by m vectors of size d × 1)
# and return the new representation as a d × 1 numpy array.
def project_image(image, U):
    # Your implementation goes here!
    project_image = np.dot(U, np.dot(U.T, image))

    return project_image

# use matplotlib to display a visual representation of the original image and the projected image side-by-side.
def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency
    # fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    # return fig, ax1, ax2

    # 1. Reshape the images to be 32 × 32
    original = np.reshape(orig, (32, 32))
    projection = np.reshape(proj, (32, 32))

    # Create the subplots
    fig, (ax1, ax2) = plt.subplots(figsize=(9, 3), ncols=2)

    # Title the first subplot (the one on the left) as “Original”
    # and the second (the one on the right) as “Projection”
    ax1.set_title("Original")
    ax2.set_title("Projection")

    #Use imshow with the optional argument aspect='equal' to display the images on the correct axes
    orig_colormap = ax1.imshow(original, aspect='equal')
    proj_colormap = ax2.imshow(projection, aspect='equal')

    #Create a colorbar for each image placed to its right
    fig.colorbar(orig_colormap, ax=ax1)
    fig.colorbar(proj_colormap, ax=ax2)

    #Return the fig, ax1 and ax2 objects used in step 2 from display_image().
    return fig, ax1, ax2