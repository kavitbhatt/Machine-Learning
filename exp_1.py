import numpy as np

dim_x = int(input("Enter dimension of vector x = "))
dim_A_0 = int(input("Enter row dimension of matrix A = "))
dim_A_1 = int(input("Enter column dimension of matrix A = "))
dim_B_0 = int(input("Enter row dimension of matrix B = "))
dim_B_1 = int(input("Enter column dimension of matrix B = "))

x = np.zeros((dim_x,1))
print("\nEnter Vector x : ")
for i in range(dim_x):
    x[i] = int(input("Enter value {0} = ".format(i+1)))
    
A = np.zeros((dim_A_0,dim_A_1))
print("\nEnter Matrix A : ")
for i in range(dim_A_0):
    for j in range(dim_A_1):
        A[i][j] = int(input("Enter value [{0}] [{1}] = "
                            .format(i+1,j+1)))

B = np.zeros((dim_B_0,dim_B_1))
print("\nEnter Matrix B : ")
for i in range(dim_B_0):
    for j in range(dim_B_1):
        B[i][j] = int(input("Enter value [{0}] [{1}] = "
                            .format(i+1,j+1)))

A_T = A.T
B_T = B.T

print("\nA Transponse")
for i in range(dim_A_0):
    for j in range(dim_A_1):
        print(A_T[i][j],end = " ")
    print()    
    
print("\nB Transponse")
for i in range(dim_B_0):
    for j in range(dim_B_1):
        print(B_T[i][j],end = " " )
    print()    

res_AB = np.dot(A,B)
res_BA = np.dot(B,A)
res_BtAt = np.dot(B_T,A_T)
res_AtB = np.dot(A_T,A)
res_ABt = np.dot(A,B_T)
res_BAt= np.dot(B,A_T)
res_BtA= np.dot(B.T,A)

if np.allclose(res_AB.T,res_BtAt):
    print("\n (AB)T equals BT*AT ")
else:
    print("\n (AB)T not equals BT*AT ")
    
det_A = np.linalg.det(A)
det_B = np.linalg.det(B)

print("\n Determinant of A : {0}".format(det_A))
print("\n Determinant of B : {0}".format(det_B))

if det_A==0:
    print("\n A is singular")
else:
    print("\n A is not singular")

if det_B==0:
    print("\n B is singular")
else:
    print("\n B is not singular")

inv_A = np.linalg.inv(A)
inv_B = np.linalg.inv(B)

print("\nINVERSE OF MATRIX A ")
for i in range(dim_A_0):
    for j in range(dim_A_1):
        print(inv_A[i][j],end = " ")
    print() 
    
print("\nINVERSE OF MATRIX B ")
for i in range(dim_B_0):
    for j in range(dim_B_1):
        print(inv_B[i][j],end = " ")
    print()  
    
identity_mat = np.identity(dim_A_0)
res_A_A_inv = np.dot(A,inv_A)
res_A_inv_A = np.dot(inv_A,A)

if np.allclose(res_A_A_inv,identity_mat) == np.allclose(
        res_A_inv_A,identity_mat):
    print("\n A*A-1 = I and A-1*A = I")

print("\n Trace of A = {0}".format(np.matrix.trace(A)))
print("Trace of B = {0}".format(np.matrix.trace(B)))
print("Trace of A-1 = {0}".format(np.matrix.trace(inv_A)))
print("Trace of B-1 = {0}".format(np.matrix.trace(inv_B)))
print("Trace of A_T = {0}".format(np.matrix.trace(A_T)))
print("Trace of B_T = {0}".format(np.matrix.trace(B_T)))

trace_AB = np.matrix.trace(res_AB)
trace_BA = np.matrix.trace(res_BA)
trace_AtB = np.matrix.trace(res_AtB)
trace_ABt = np.matrix.trace(res_ABt)
trace_BAt = np.matrix.trace(res_BAt)
trace_BtA = np.matrix.trace(res_BtA)

if trace_AB==trace_BA:
    print("\n Trace (A*B) = trace (B*A)")
else : 
    print("\n Trace (A*B) != trace (B*A)")

if trace_ABt == trace_AtB == trace_BAt == trace_BtA:
    print('''\n Trace(AT*B) = trace(A*BT) = trace(B*AT) 
          = trace(BT*A)''')
else:
    print('''\n Trace(AT*B) != trace(A*BT) != trace(B*AT) 
          != trace(BT*A)''')
    
y = np.dot(A,x)
x_hat = y*inv_A
inner_product = np.dot(x.T,y)

if inner_product == 0 :
    print("\n x and y are orthogonal")
else:
    print("\n x and y are not orthogonal")
    
norm_x = np.linalg.norm(x)
normalized_x = x/norm_x
norm_y = np.linalg.norm(y)
normalized_y = x/norm_y

print("\nL2 Norm of vector x = {0}".format(norm_x))
print("Normalized vector x : {0}".format(normalized_x))
print("\\nL2 Norm of of vector y = {0}".format(norm_y))
print("Normalized vector y : {0}".format(normalized_y))

if inner_product <= (norm_x * norm_y):
    print("\n Cauchy-Schwartz inequality Verified")
else:
    print("\n Cauchy-Schwartz inequality Not Verified")
    
res_xty = np.dot(x.T,y)
res_xyt = np.dot(y.T,x)
print(res_xty)
print(res_xyt)

if res_xty == res_xyt:
    print("xty equals ytx")
else: 
    print("xty not equals ytx")

res_ytAx = np.dot(np.dot(y.T,A),x)
res_xtAty = np.dot(np.dot(x.T,A.T),y)

if np.allclose(res_ytAx,res_xtAty) :
    print("\n yT*A*x = xT*AT*y ")
else:
    print("\n yT*A*x != xT*AT*y ")
   
egval_A, egvec_A = np.linalg.eig(A)   
egval_B, egvec_B = np.linalg.eig(B)

print("\n Eigen Values of Matrix A : {0}".format(egval_A))
print("Eigen Vector of Matrix A : {0}".format(egvec_A))

print("\n Eigen Values of Matrix B : {0}".format(egval_B))
print("Eigen Vector of Matrix B : {0}".format(egvec_B))

Λ_A = np.diag(egval_A)
A_decomposed = np.dot(np.dot(egvec_A,Λ_A),np.linalg.inv(egvec_A))
print("\nEVD of Vector A : {0}".format(A_decomposed))

Λ_B = np.diag(egval_B)
B_decomposed = np.dot(np.dot(egvec_B,Λ_B),np.linalg.inv(egvec_B))
print("\nEVD of Vector B : {0}".format(B_decomposed))