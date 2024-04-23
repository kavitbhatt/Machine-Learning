import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Generate dataset
X,y=make_classification(n_samples=100,n_features=5,n_classes=2)

#Pre-process the dataset
y[y==0]=-1
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=20, train_size=80)

# Display overall dataset
plt.scatter(X[:, 0], X[:, 1], c=y, label='Overall Dataset')
plt.title('Scatter plot of Overall Dataset')
plt.show()

# Display Testing and Training Dataset
plt.scatter(X_train[:, 0], X_train[:, 1], c='r')
plt.scatter(X_test[:, 0], X_test[:, 1], c='b')
plt.legend(['Training Dataset','Testing Dataset'])
plt.title('Scatter plot of Training, and Testing Dataset')
plt.show()

def activation_function(activation):
    if activation > 0:
        return 1
    else:
        return -1

def Perceptron(X,y):   
    #initialize weight vector
    weights = np.zeros((X_train.shape[1],1))
    mistakes = 0
    
    for i in range(X_train.shape[0]):
        y_pred = activation_function(np.dot(weights.T,X_train[i]))
        if y_pred != y_train[i]:
            w1 = weights.T + y_train[i]*X_train[i]
            weights = w1.T
            mistakes = mistakes + 1
    return weights,mistakes
                
def predict(weights,X):
    y_pred=[]
    for i in range(X.shape[0]):
        y_pred.append(activation_function(np.dot(weights.T,X[i])))
    return y_pred   

weights,mistakes_train = Perceptron(X_train, y_train)
y_pred = predict(weights,X_test)
mistakes_test = np.sum(y_pred != y_test) 
accuracy = (y_test.shape[0]-mistakes_test)/y_test.shape[0]

print("Mistakes During Training : ",mistakes_train)
print("Mistakes During Testing : ",mistakes_test)
print('Accuracy : ',accuracy*100,'%')

x0_1 = np.amax(X_test[:, 0])
x0_2 = np.amin(X_test[:, 0])
x1_1 = (-weights[0] * x0_1) / weights[1]
x1_2 = (-weights[0] * x0_2) / weights[1]

#Plot Testing and Predicted Dataset with Decision Boundary
plt.plot([x0_1, x0_2], [x1_1, x1_2])
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s = 20)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='x')
plt.legend(['Decision Boundary','Testing Dataset','Predicted Dataset'])
plt.title('Scatter plot of Testing and Predicted Dataset')
plt.show()

cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm, display_labels = [False, True])
cm_display.plot()
plt.title("Confusion Matrix")
plt.show()