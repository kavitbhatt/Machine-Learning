import numpy as np
import matplotlib.pyplot as plt

h=10**(-7)

def y(x1,x2):
    return 10*x1**2 + 5*x1*x2 + 10*(x2-3)**2


def numdiff(x1,x2):
    X1=np.array([x1-h,x1,x1+h])
    X2=np.array([x2-h,x2,x2+h])
    X11,X22=np.meshgrid(X1,X2)
    
    Y=y(X11,X22)
    z1=np.diff(Y,axis=1)/np.diff(X11,axis=1)
    z2=np.diff(Y,axis=0)/np.diff(X22,axis=0)
    return z1[0][0],z2[0][0]

def gradient_norm(z):
    return np.linalg.norm(z)

stepsize=[0.001,0.005,0.01,0.05]
N=len(stepsize)
table=np.zeros([4,5])

x0=[10,15]
i=0
while(i<N):
    x01=x0[0]
    x02=x0[1]
    z0=np.array(numdiff(x01,x02))
    it=0
    while gradient_norm(z0) > 0.001:
        x01=x01-stepsize[i]*z0[0]
        x02=x02-stepsize[i]*z0[1]
        z0=np.array(numdiff(x01,x02))
        it+=1
    print('After convergence minima value is at =[', x01,',', x02,'] and'
          ,'function value is= ',y(x01,x02),',no of itrations = ',it,
          'while stepsize = ',stepsize[i],'\n')
    table[i,0]=stepsize[i]
    table[i,1]=it
    table[i,2]=x01
    table[i,3]=x02
    table[i,4]=y(x01,x02)
    i+=1

j=0
print('table')
while (j<N):
        print(table[j][0],'   ',table[j][1],'   ',table[j][2],'    ',
              table[j][3],'   ',table[j][4])
        j+=1
        
x1=np.linspace(-10,10,1000)
x2=np.linspace(-15,15,1000)
x1,x2=np.meshgrid(x1,x2)
funx=y(x1,x2)
plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')
ax.plot_surface(x1, x2, funx,cmap='plasma')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title('Y=10*x1*2 + 5*x1*x2 + 10(x2-3)**2')
plt.grid()
plt.tight_layout()
plt.show()