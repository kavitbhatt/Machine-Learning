import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=100, n_features=1, bias = 0, noise =5)
# Scale feature x to range -5…..5
X = np.interp(X, (X.min(), X.max()), (-5, 5))
 # Scale target y to range 15…..-15
y = np.interp(y, (y.min(), y.max()), (15, -15))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=20, train_size=80)

# Display overall dataset
plt.scatter(X,y, label='Overall Dataset')
plt.title('Scatter plot of Overall Dataset')
plt.xlabel("X (Range : -5 to 5)")   
plt.ylabel("Y (Range : 15 to -15)")
plt.show()

# Display Testing and Training Dataset
plt.xlabel("X (Range : -5 to 5)")
plt.scatter(X_train,y_train, color ='r')
plt.scatter(X_test, y_test, color ='b')

plt.ylabel("Y (Range : 15 to -15)")
plt.legend(['Training Dataset','Testing Dataset'])
plt.title('Scatter plot of Training, and Testing Dataset')
plt.show()

X_b = np.c_[np.ones([80,1]),X_train] # X_b = [1,X_train] 

def Linear_Regression_Manual(X,y):
    a = np.dot(X.T,X)
    b = np.dot(X.T,y)
    weights = np.dot(np.linalg.inv(a),b)
    return weights

def Predict(w,X):
    y = np.ones([X.shape[0],1])
    for i in range(X.shape[0]):
        y[i] = w[1]*X[i] + w[0]
    return y 
 
weights_final = Linear_Regression_Manual(X_b,y_train)
y_pred = Predict(weights_final,X_test) 

RMSE_train_manual = mean_squared_error(y_train, Predict(weights_final,X_train))
RMSE_test_manual = mean_squared_error(y_test, Predict(weights_final,X_test))

print("RMSE OF TRAINED MODEL FOR TRAINING DATASET : ",RMSE_train_manual)
print("RMSE OF TRAINED MODEL FOR TESTING DATASET : ",RMSE_test_manual)

# SHOW LINEAR REGRESSION TRAINED MODEL ON SAME SCATTER PLOT 
# WITH TRAINING AND TESTING DATASET
plt.plot(X,Predict(weights_final,X), color ='k')    
plt.scatter(X_train,y_train, color ='r')
plt.scatter(X_test, y_test, color ='b')
plt.xlabel("X (Range : -5 to 5)")
plt.ylabel("Y (Range : 15 to -15)")
plt.legend(['Training Dataset','Testing Dataset'])
plt.title('Linear Regression Using Close Form')
plt.show()

regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred1 = regr.predict(X_test)
plt.plot(X,regr.predict(X), color ='k')    
plt.scatter(X_train,y_train, color ='r')
plt.scatter(X_test, y_test, color ='b')
plt.xlabel("X (Range : -5 to 5)")
plt.ylabel("Y (Range : 15 to -15)")
plt.legend(['Training Dataset','Testing Dataset'])
plt.title('Linear Regression Using Sklearn')
plt.show()

RMSE_train_imported = mean_squared_error(y_train, regr.predict(X_train))
RMSE_test_imported = mean_squared_error(y_test, regr.predict(X_test))

print("\nRMSE OF IMPORTED MODEL FOR TRAINING DATASET : ",RMSE_train_imported)
print("RMSE OF IMPORTED MODEL FOR TESTING DATASET : ",RMSE_test_imported)

print("\nWEIGHTS OF LINEAR REGRESSION CLOSE FORM SOLUTION MODEL : ")
print("w = ",weights_final[1],"  b = ",weights_final[0])

print("\nWEIGHTS OF LINEAR REGRESSION IMPORTED MODEL : ")
print("w = ",regr.coef_[0],"  b = ",regr.intercept_)