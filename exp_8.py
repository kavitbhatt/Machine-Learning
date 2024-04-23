import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

iris = datasets.load_iris()
X = iris.data[:,2:] #PETAL LENGTH AND WIDTH
y = iris.target

scatter = plt.scatter(X[:, 0], X[:, 1], c=y, marker='o')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(['Iris setosa','Iris versicolor','Iris virginica'])
plt.legend(scatter.legend_elements()[0], iris.target_names, title="Classes")
plt.title("Scatter Plot of Whole Dataset")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Multi-Class Softmax Logistic Regression
MLR = LogisticRegression(multi_class='multinomial')
MLR.fit(X_train,y_train)
y_pred_MLR = MLR.predict(X_test)

mistakes_test_MLR = np.sum(y_pred_MLR != y_test) 
accuracy_MLR = (y_test.shape[0]-mistakes_test_MLR)/y_test.shape[0]

print("\nResults for Multi-Class Softmax Logistic Regression:")
print("Mistakes During Testing : ",mistakes_test_MLR)
print('Accuracy : ',accuracy_MLR*100,'%')

DecisionBoundaryDisplay.from_estimator(MLR,X_train,cmap = 'twilight_r'
                                       ,xlabel="Petal length",
                                       ylabel="Petal width")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_MLR, marker='x')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s = 20)
plt.legend(scatter.legend_elements()[0], ['sentosa',
                                          'versicolor',
                                          'virginica',])
plt.title('''Scatter plot of Testing and Predicted Dataset for 
Multi-Class Softmax Logistic Regression''')
plt.show()

# Linear SVM
LSVM = SVC(kernel='linear')
LSVM.fit(X_train, y_train)
y_pred_LSVM = LSVM.predict(X_test)

mistakes_test_LSVM = np.sum(y_pred_LSVM != y_test) 
accuracy_LSVM = (y_test.shape[0]-mistakes_test_LSVM)/y_test.shape[0]

print("\nResults for Linear SVM:")
print("Mistakes During Testing : ",mistakes_test_LSVM)
print('Accuracy : ',accuracy_LSVM*100,'%')

DecisionBoundaryDisplay.from_estimator(LSVM,X_train,cmap = 'twilight_r',
                                       xlabel="Petal length",
                                       ylabel="Petal width")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_LSVM, marker='x')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s = 20)
plt.legend(scatter.legend_elements()[0], ['sentosa',
                                          'versicolor',
                                          'virginica',])
plt.title('Scatter plot of Testing and Predicted Dataset for Linear SVM')
plt.show()

# Soft SVM
SSVM = SVC(kernel='linear')
SSVM.fit(X_train, y_train)
y_pred_SSVM = SSVM.predict(X_test)

mistakes_test_SSVM = np.sum(y_pred_SSVM != y_test) 
accuracy_SSVM = (y_test.shape[0]-mistakes_test_SSVM)/y_test.shape[0]

print("\nResults for Linear SVM:")
print("Mistakes During Testing : ",mistakes_test_SSVM)
print('Accuracy : ',accuracy_SSVM*100,'%')

DecisionBoundaryDisplay.from_estimator(SSVM,X_train,cmap = 'twilight_r',
                                       xlabel="Petal length",
                                       ylabel="Petal width")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_SSVM, marker='x')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s = 20)
plt.legend(scatter.legend_elements()[0], ['sentosa',
                                          'versicolor',
                                          'virginica',])
plt.title('Scatter plot of Testing and Predicted Dataset for Soft SVM')
plt.show()