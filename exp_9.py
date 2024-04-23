import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


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

# Decision Tree Classification
DT_Classification = DecisionTreeClassifier()
DT_Classification.fit(X_train,y_train)
y_pred_DTC = DT_Classification.predict(X_test)

mistakes_test_DTC = np.sum(y_pred_DTC != y_test) 
accuracy_DTC = (y_test.shape[0]-mistakes_test_DTC)/y_test.shape[0]

print("\nResults for Decision Tree Classification: ")
print("Mistakes During Testing : ",mistakes_test_DTC)
print('Accuracy : ',accuracy_DTC*100,'%')

plot_tree(DT_Classification, feature_names=list(iris.feature_names[2:]),  
         class_names=list(iris.target_names),filled=True)
plt.title("Trained Tree of Decision Tree Classification")
plt.show()

# Decision Tree Regression
x = np.random.uniform(-5,5,100) # 100 Data Points Generated
y = np.exp(x) # y = e^x

x = x.reshape(-1, 1) # Convert row vector into column vector
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

plt.scatter(x_train,y_train,c='red')
plt.scatter(x_test,y_test,c='yellow')
plt.xlabel('x')
plt.ylabel('y = e^x')
plt.legend(['Training Dataset','Test Dataset'])
plt.title('Scatter Plot of Whole Dataset for Decision Tree Regression')
plt.show()

DT_Regression = DecisionTreeRegressor()
DT_Regression.fit(x_train,y_train)

y_pred_DTR_train = DT_Regression.predict(x_train)
y_pred_DTR_test = DT_Regression.predict(x_test)

RMSE_DTR_train = np.sqrt(mean_squared_error(y_train,y_pred_DTR_train))
RMSE_DTR_test = np.sqrt(mean_squared_error(y_test,y_pred_DTR_test))

print("\nRMSE OF Decision Tree Regression FOR TRAINING DATASET : ",RMSE_DTR_train)
print("RMSE OF Decision Tree Regression FOR TESTING DATASET : ",RMSE_DTR_test)

X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(X_grid, DT_Regression.predict(X_grid), color = 'black')
plt.scatter(x_train, y_pred_DTR_train, color = 'red')
plt.scatter(x_test, y_pred_DTR_test, color = 'yellow')
plt.title('Decision Tree Regression')
plt.legend(['Decision Tree Regression Line','Training Dataset','Testing Dataset'])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plot_tree(DT_Regression,filled=True)
plt.title("Trained Tree of Decision Tree Regression")
plt.show()