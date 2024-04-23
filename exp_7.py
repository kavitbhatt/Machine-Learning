import numpy as np
from keras.datasets import mnist

def scatter_matrices(X, y, selected_digits):
    n_features = X.shape[1]
    overall_mean = np.mean(X, axis=0)
    S_W = np.zeros((n_features, n_features))
    S_B = np.zeros((n_features, n_features))

    for digit in selected_digits:
        class_data = X[y == digit]
        class_mean = np.mean(class_data, axis=0)
        n = len(class_data)
        deviation = class_mean - overall_mean

        scatter_matrix = np.zeros((n_features, n_features))
        for row in class_data:
            row = row.reshape(n_features, 1)
            scatter_matrix += np.dot(deviation, deviation.T)
        S_B += scatter_matrix * n
        S_W += np.cov(class_data.T, bias=True) * n
    return S_W, S_B

def lda_projection(X, y, selected_digits, n_components):
    S_W, S_B = scatter_matrices(X, y, selected_digits)
    eig_vals, eig_vecs = np.linalg.eigh(np.dot(np.linalg.pinv(S_W), S_B))
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    return eig_vecs[:, :n_components]

# Load MNIST dataset
(X_train, y_train), (_, _) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1)

# Extract only digits 4, 7, and 8
selected_digits = [4, 7, 8]
idx = np.where(np.isin(y_train, selected_digits))
X_train, y_train = X_train[idx], y_train[idx]

# Total number of images for training
total_images = 900

# Shuffle the data
shuffle_idx = np.random.permutation(len(X_train))
X_train, y_train = X_train[shuffle_idx], y_train[shuffle_idx]

# Take the required number of images for training
X_train, y_train = X_train[:total_images], y_train[:total_images]

max_dimension=n_components=len(selected_digits) - 1
# Compute projection matrices
projection_matrices = lda_projection(X_train, y_train, selected_digits, max_dimension)

print("Maximum LDA dimensions:", projection_matrices.shape[1])
print("Projection Matrix:",projection_matrices)

X_red=np.dot(X_train,projection_matrices)