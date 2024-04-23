from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

#loading the dataset
(X1,y1),(X2,y2) = mnist.load_data()

X1 = X1.reshape((-1, 784))
X2 = X2.reshape((-1, 784))

X1 = X1/255 #Normalize by dividing by 255
X2 = X2/255

#FOR MAKING TRAINING DATASET
four = []
seven = []
eight = []
for i in range(60000):
  if y1[i]==4:
    four.append(i)
  elif y1[i]==7:
    seven.append(i)
  elif y1[i]==8:
    eight.append(i)
  else:
    continue

np.random.shuffle(four)
np.random.shuffle(seven)
np.random.shuffle(eight)
X_train = np.concatenate([X1[four[:300]],X1[seven[:300]],X1[eight[:300]]],axis=0)
y_train = np.zeros([900,1])
y_train[0:300]=4
y_train[300:600]=7
y_train[600:900]=8

plt.figure(figsize=(12,4))
plt.suptitle(' IMAGES WITH 784 FEATURES ',fontsize=30)
plt.subplot(1,3,1) 
image = X_train[np.random.randint(0,300)]
image = image.reshape(28,28)
plt.imshow(image,cmap='gray')
plt.axis('off')

plt.subplot(1,3,2) 
image = X_train[np.random.randint(300,600)]
image = image.reshape(28,28)
plt.imshow(image,cmap='gray')
plt.axis('off')

plt.subplot(1,3,3) 
image = X_train[np.random.randint(600,900)]
image = image.reshape(28,28)
plt.imshow(image,cmap='gray')
plt.axis('off')
plt.show() 

shuffle_index = np.random.permutation(900)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

cov_mat = np.cov(X_train.T)
eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
eigen_vectors = eigen_vectors.T
sorted_index = np.argsort(eigen_values)[::-1] #INDEX OF SORTED EIGEN VALUES
sorted_eigenvalues = eigen_values[sorted_index] #SORTED EIGEN VALUES
sorted_eigenvectors = eigen_vectors[sorted_index]
    
eig_vals_total = sum(eigen_values)
eigen_variance = [(i / eig_vals_total) for i in sorted_eigenvalues]
eigen_variance = np.round(eigen_variance,4)
cum_eigen_variance = np.cumsum(eigen_variance)

plt.plot(np.arange(1,784+1), cum_eigen_variance,'-o')
plt.xlabel('Number of components')
plt.ylabel('Cumulative Eigen variance');
plt.title("Eigen Value Variance Plot with all components");
plt.show()

# SORTED EIGEN VECTOR ACCORDING TO SORTED EIGEN VALUES

def PCA(X , num_components):
    A = sorted_eigenvectors[:num_components]# Projection matrix
    X_reduced = np.dot(A,X.T)
    
    # Eigen variance of reduced components
    component_eigen_variance = [i/eig_vals_total for i in sorted_eigenvalues[:num_components]]
    
    # Cumulative Eigen variance of reduced components
    cum_component_eigen_variance = np.cumsum(component_eigen_variance) 
    
    return X_reduced,cum_component_eigen_variance[-1],A

def Inverse_PCA(PCA_X,A,components):
    xre = np.dot(PCA_X.T,A)
    # xre = np.dot(A.T,PCA_X)
    
    plt.figure(figsize=(8,6))
    plt.suptitle(' IMAGES WITH {0} FEATURES '.format(components),fontsize=20)
    
    i1 = np.random.randint(0,300)
    i2 = np.random.randint(0,300)
    i3 = np.random.randint(0,300)
    
    plt.subplot(2,3,1) 
    plt.title("m = {0}".format(components))
    image = xre[i1]
    image = image.reshape(28,28)
    plt.imshow(image,cmap='gray')
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.title("m = {0}".format(components))
    image = xre[i2]
    image = image.reshape(28,28)
    plt.imshow(image,cmap='gray')
    plt.axis('off')

    plt.subplot(2,3,3) 
    plt.title("m = {0}".format(components))
    image = xre[i3]
    image = image.reshape(28,28)
    plt.imshow(image,cmap='gray')
    plt.axis('off')
    
    plt.subplot(2,3,4)
    plt.title("m = 784")
    image = X_train[i1]
    image = image.reshape(28,28)
    plt.imshow(image,cmap='gray')
    plt.axis('off')

    plt.subplot(2,3,5)
    plt.title("m = 784")
    image = X_train[i2]
    image = image.reshape(28,28)
    plt.imshow(image,cmap='gray')
    plt.axis('off')

    plt.subplot(2,3,6)
    plt.title("m = 784")
    image = X_train[i3]
    image = image.reshape(28,28)
    plt.imshow(image,cmap='gray')
    plt.axis('off')
    plt.show() 
    
X_reduced = []
Cum_component_eigen_variance = []
components = [2,10,50,100,200,300]    
Projection_Matrix = []
for i in range(len(components)):
    x1,c1,a1 = PCA(X_train,components[i])
    
    X_reduced.append(x1)
    Cum_component_eigen_variance.append(c1)
    Projection_Matrix.append(a1)
    
    Inverse_PCA(X_reduced[i],Projection_Matrix[i],components[i])
    
    print("No. Of Components ",components[i])
    print("Total variance in data :",Cum_component_eigen_variance[i]*100,'%')
    print("----------------------------------------------------------------")