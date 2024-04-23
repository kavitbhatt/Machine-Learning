import numpy as np
import matplotlib.pyplot as plt
h=10**(-7)

def y(x):
    return  x * np.exp(-x**2)

def numdiff(x):
    X=np.array([x-h,x,x+h])
    Y=y(X)
    z=np.diff(Y)/np.diff(X)
    return z[0]

stepsize=[0.001,0.005,0.01,1.0]
N=len(stepsize)
table=np.zeros([4,4])

i=0
while(i<N):
    x0=-np.sqrt(1.5)
    z0=np.array(numdiff(x0))
    it=0
    while (np.abs(z0)>0.001):
        x0=x0-stepsize[i]*z0
        z0=np.array(numdiff(x0))
        it+=1
    print('After convergence minima value is at =[', x0,'] and'
          ,'function value is= ',y(x0),',no of itrations = ',it
          ,'while stepsize = ',stepsize[i],'\n')
    table[i,0]=stepsize[i]
    table[i,1]=it
    table[i,2]=x0
    table[i,3]=y(x0)
    i+=1
j=0
print('table')
while (j<N):
        print(table[j][0],'   ',table[j][1],'   ',table[j][2],'    ',
              table[j][3])
        j+=1
        
x=np.linspace(-np.sqrt(1.5),0,1000)
minima=np.mean(table[:,3])
funx=y(x)
plt.figure(figsize=(10,8))
plt.plot(x,funx)
plt.axhline(minima,color='r')
plt.xlabel("X")
plt.ylabel("Y= X * exp(-X**2)")
plt.legend(['X * exp(-X**2)', minima])
plt.grid()
plt.tight_layout()
plt.show()