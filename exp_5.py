import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

X, y = make_classification(n_samples=100, n_features=5, n_classes=2)
y[y==0]=-1
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=20)

plt.scatter(X_train[:, 0], X_train[:, 1], c='r', marker='o')
plt.scatter(X_test[:, 0], X_test[:, 1], c='b', marker='*')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(['Training Dataset','Testing Dataset'])
plt.title('Scatter plot of training and testing dataset')
plt.show()

def signum(x):
    if x >= 0:
        return 1
    else :
        return -1
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def MCE_gradient(weigths,X):
    gradient = 0
    for i in range(len(X)):
        #gradient= -yi(1-l[yi*W.T*Xi])*l(yi*W.T*Xi)*Xi
        l_x = sigmoid(y_train[i]*np.dot(weigths.T,X[i]))
        gradient += -y_train[i]*(1-l_x)*l_x*X[i]
    return gradient
    
def MCE(stepsize,X_train,y_train):
    weights = np.zeros(X_train.shape[1])  
    mistakes = 0
    
    for i in range(len(X_train)):
        y_pred = signum(np.dot(weights.T,X_train[i]))
        if(y_pred != y_train[i]):
            mistakes += 1
            weights = weights - stepsize * MCE_gradient(weights,X_train) 
    return weights,mistakes

def MCE_predict(weights,X):
    y_pred=[]
    for i in range(X.shape[0]):
        y_pred.append(signum(np.dot(weights.T,X[i])))
    return y_pred  

def LR_gradient(weigths,X):
    gradient = 0
    for i in range(len(X)):
        #gradient= yi(1-l[yi*W.T*Xi])*Xi
        l_x = sigmoid(y_train[i]*np.dot(weigths.T,X[i]))
        gradient += -y_train[i]*(1-l_x)*X[i]
    return gradient

def LR(stepsize,X_train,y_train):
    weights = np.zeros(X_train.shape[1])  
    mistakes = 0
    
    for i in range(len(X_train)):
        y_pred = signum(np.dot(weights.T,X_train[i]))
        if(y_pred != y_train[i]):
            
            mistakes += 1
            weights = weights - stepsize * LR_gradient(weights,X_train) 
    return weights,mistakes

def LR_predict(weights,X):
    y_pred=[]
    for i in range(X.shape[0]):
        y_pred.append(signum(np.dot(weights.T,X[i])))
    return y_pred 

#MCE PART
weights_MCE,mistake_MCE_train = MCE(0.001,X_train,y_train)
y_pred_MCE = MCE_predict(weights_MCE,X_test)
mistakes_MCE_test = np.sum(y_pred_MCE != y_test) 
accuracy_MCE = (y_test.shape[0]-mistakes_MCE_test)/y_test.shape[0]

print("------------------ MCE ------------------")
print("Mistakes During Training : ",mistake_MCE_train)
print("Mistakes During Testing : ",mistakes_MCE_test)
print('Accuracy : ',accuracy_MCE*100,'%')

x0_1 = np.amax(X_test[:,0])
x0_2 = np.amin(X_test[:,0])
x1_1 = (-weights_MCE[0] * x0_1) / weights_MCE[1]
x1_2 = (-weights_MCE[0] * x0_2) / weights_MCE[1]

#Plot Testing and Predicted Dataset with Decision Boundary
plt.plot([x0_1, x0_2], [x1_1, x1_2])
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_MCE, marker='x')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s = 20)
plt.legend(['Decision Boundary','Predicted Dataset','Testing Dataset'])
plt.title('Scatter plot of Testing and Predicted Dataset')
plt.show()

cm = confusion_matrix(y_test, y_pred_MCE)
cm_display = ConfusionMatrixDisplay(cm, display_labels = [True, False])
cm_display.plot()
plt.title("Confusion Matrix")
plt.show()

weights_LR,mistake_LR_train = LR(0.001,X_train,y_train)
y_pred_LR = LR_predict(weights_LR,X_test)
mistakes_LR_test = np.sum(y_pred_LR != y_test) 
accuracy_LR = (y_test.shape[0]-mistakes_LR_test)/y_test.shape[0]

#LR PART
print("------------------ LR ------------------")
print("Mistakes During Training : ",mistake_LR_train)
print("Mistakes During Testing : ",mistakes_LR_test)
print('Accuracy : ',accuracy_LR*100,'%')

x0_1 = np.amax(X_test[:,0])
x0_2 = np.amin(X_test[:,0])
x1_1 = (-weights_LR[0] * x0_1) / weights_LR[1]
x1_2 = (-weights_LR[0] * x0_2) / weights_LR[1]

#Plot Testing and Predicted Dataset with Decision Boundary
plt.plot([x0_1, x0_2], [x1_1, x1_2])
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_LR, marker='x')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s = 20)
plt.legend(['Decision Boundary','Predicted Dataset','Testing Dataset'])
plt.title('Scatter plot of Testing and Predicted Dataset')
plt.show()

cm = confusion_matrix(y_test, y_pred_LR)
cm_display = ConfusionMatrixDisplay(cm, display_labels = [True, False])
cm_display.plot()
plt.title("Confusion Matrix")
plt.show()