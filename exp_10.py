import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# ANN training function with K-Fold cross-validation
def ANN_Train_KFold(x, y, epochs, learning_rate = 0.01, k=5):
    kf = KFold(n_splits=k)
    fold = 0
    
    for train_index, test_index in kf.split(x):
        fold += 1
        print(f"Fold #{fold}")
        
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Initialize weights and biases as scalars
        w1, w2 = np.random.randn(), np.random.randn()
        b1, b2 = np.random.randn(), np.random.randn()

        for epoch in range(epochs):
            # Training loop for each epoch
            for i in range(len(x_train)):
                # Forward pass, ensure x[i] is used as a scalar
                z1 = sigmoid(w1 * x_train[i][0] + b1)  # Use [0] to ensure scalar input
                z2 = sigmoid(w2 * z1 + b2)
                
                # Backpropagation
                error = z2 - y_train[i]
                d_z2 = error * sigmoid_derivative(z2)
                
                d_w2 = d_z2 * z1
                d_b2 = d_z2
                
                d_z1 = d_z2 * w2 * sigmoid_derivative(z1)
                d_w1 = d_z1 * x_train[i][0]  # Use [0] to ensure scalar input
                d_b1 = d_z1
                
                # Update weights and biases
                w1 -= learning_rate * d_w1
                b1 -= learning_rate * d_b1
                w2 -= learning_rate * d_w2
                b2 -= learning_rate * d_b2

            # Evaluate after each epoch
            y_pred_train = ANN_Predict(x_train, w1, w2, b1, b2)
            y_pred_test = ANN_Predict(x_test, w1, w2, b1, b2)
            RMSE_Train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            RMSE_Test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
        

            print("\nAfter Iteration ",epoch+1)
            print("Weights : ")
            print("w1 = ",w1)
            print("w2 = ",w2)
            print("\nBiases : ")
            print("b1 = ",b1)
            print("b2 = ",b2)
            print("\nRMSE of Training Dataset : ",RMSE_Train)
            print("RMSE of Testing Dataset : ",RMSE_Test)
            
        # Plot y_pred_train and y_pred_test for each epoch
        
        X_grid = np.arange(min(x), max(x), 0.01)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.plot(X_grid, ANN_Predict(X_grid, w1, w2, b1, b2), color = 'black')
        plt.scatter(x_train, y_pred_train, color = 'red')
        plt.scatter(x_test, y_pred_test, color = 'yellow')
        
        plt.legend(['Fit','Training Dataset','Testing Dataset'])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

def ANN_Predict(x, w1, w2, b1, b2):
    y_pred = np.zeros(len(x))
    for i in range(len(x)):
        z1 = sigmoid(w1 * x[i][0] + b1)
        z2 = sigmoid(w2 * z1 + b2)
        y_pred[i] = z2  # Ensure z2 is a scalar here
    return y_pred

# Generate random data
x = np.random.uniform(0, 10, 100)  # 100 Data Points Generated
y = np.exp(-0.1 * x)  # y = e^(-0.1*x)
x = x.reshape(-1, 1)  # Convert row vector into column vector

# Plot the data
plt.scatter(x, y, c='orange')
plt.xlabel('x')
plt.ylabel('y = e^-0.1x')
plt.legend(['Training Dataset', 'Test Dataset'])
plt.title('Scatter Plot of Whole Dataset for Simple ANN')
plt.show()

# Train the ANN with K-Fold cross-validation
ANN_Train_KFold(x, y, epochs=5, learning_rate=0.01, k=5)  # Using 5 folds and 5 epochs
