import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

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
plt.scatter(X_train,y_train, color ='r')
plt.scatter(X_test, y_test, color ='b')
plt.xlabel("X (Range : -5 to 5)")
plt.ylabel("Y (Range : 15 to -15)")
plt.legend(['Training Dataset','Testing Dataset'])
plt.title('Scatter plot of Training, and Testing Dataset')
plt.show()

X_b = np.c_[np.ones([80,1]),X_train] # X_b = [1,X_train] 

#optimized weight = (XT*X + alpha*I)^-1 * XT*y 
def Ridge_Regression_Manual(X,y,alpha):
    I = np.identity(X.shape[1])
    a = np.dot(X.T,X) + alpha * I 
    b = np.dot(X.T,y)
    weights = np.dot(np.linalg.inv(a),b)
    return weights

# y = wTx + b 
def Predict(w,X):
    y = np.ones([X.shape[0],1])
    for i in range(X.shape[0]):
        y[i] = w[1]*X[i] + w[0]
    return y 

# ALPHA HYPERPARAMATER 0.1,0.5,1.0,1.5
alpha = [0.1,0.5,1.0,1.5]
y_pred_calculated = []
weights_final = []

for i in range(len(alpha)):
    weights_final.append(Ridge_Regression_Manual(X_b,y_train,alpha[i]))
    y_pred_calculated.append(Predict(weights_final[i],X_test))
    
    print("\nWEIGHTS OF RIDGE REGRESSION CLOSE FORM ALPHA = ",alpha[i])
    print("w = ",weights_final[i][1],"  b = ",weights_final[i][0])
    
    regr = Ridge(alpha[i])
    regr.fit(X_train, y_train)
    y_pred_imported = regr.predict(X_test)
    
    print("\nWEIGHTS OF RIDGE REGRESSION SKLEARN ALPHA = ",alpha[i])
    print("w = ",regr.coef_[0],"  b = ",regr.intercept_)
    
    RMSE_train_manual = mean_squared_error(y_train, Predict(weights_final[i],X_train))
    RMSE_test_manual = mean_squared_error(y_test, Predict(weights_final[i],X_test))
    print("\nRMSE OF TRAINED MODEL FOR TRAINING DATASET : ",RMSE_train_manual)
    print("RMSE OF TRAINED MODEL FOR TESTING DATASET : ",RMSE_test_manual)
    
    RMSE_train_imported = mean_squared_error(y_train, regr.predict(X_train))
    RMSE_test_imported = mean_squared_error(y_test, regr.predict(X_test))
    print("\nRMSE OF IMPORTED MODEL FOR TRAINING DATASET : ",RMSE_train_imported)
    print("RMSE OF IMPORTED MODEL FOR TESTING DATASET : ",RMSE_test_imported)
    print("\n")
    # SHOW RIDGE REGRESSION TRAINED MODEL ON SAME SCATTER PLOT 
    # WITH TRAINING AND TESTING DATASET
    plt.plot(X,Predict(weights_final[i],X), color ='k')    
    plt.scatter(X_train,y_train, color ='r')
    plt.scatter(X_test, y_test, color ='b')
    plt.xlabel("X (Range : -5 to 5)")
    plt.ylabel("Y (Range : 15 to -15)")
    plt.legend(['Training Dataset','Testing Dataset'])
    plt.title('Ridge Regression Using Close Form for Alpha = {0}'.format(alpha[i]))
    plt.show()
    
    plt.plot(X,regr.predict(X), color ='k')    
    plt.scatter(X_train,y_train, color ='r')
    plt.scatter(X_test, y_test, color ='b')
    plt.xlabel("X (Range : -5 to 5)")
    plt.ylabel("Y (Range : 15 to -15)")
    plt.legend(['Training Dataset','Testing Dataset'])
    plt.title('Ridge Regression Using Sklearn for Alpha {0}'.format(alpha[i]))
    plt.show()