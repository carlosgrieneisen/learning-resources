import numpy as np


class Solve():
    def __init__(self, A,b,x):
        self.A_o = A # matriz
        self.b_o = b # vetor
        self.x_o = x # produto de A @ b
        self.N = len(A) # dimensão de A (usada pra loops etc, atenção não necessariamente é quadrada)
    
    # escalonamento
    def elimGauss(self):
        A = np.array(self.A_o,dtype=float,copy=True)
        N = len(A)
        k = A[0][0]/A[1][0]
        for i in range(N):
            for j in range(N):
                print(A[i][j], end=" ")
            print()

    def fatoraLU(self):
        # nomenclatura apenas
        A = np.array(self.A_o,dtype=float,copy=True)
        L = np.eye(self.N) #matriz identidade L, N x N
        # coluna
        for j in range(0,self.N-1):
            # linha abaixo do elemento A[j,j]
            # se fosse até N-1, cancela por causa do j do loop   
            for i in range(j+1,self.N): 
                L[i,j]= A[i,j]/A[j,j]
                A[i,:]=A[i,:]-L[i,j]*A[j,:]
        return L,A
    
    def resolveL(self,L,b):
        N=self.N
        y=np.zeros(N)
        y[0]=b[0]
        for i in range(1,N):
            y[i]=(b[i]-L[i,:i]@y[:i])/L[i,i]
        return y
    
    def resolveU(self,U,y):
        N=self.N
        x = np.zeros(N) 
        x[N-1]=y[N-1]/U[N-1,N-1]
        for i in range(N-2,-1,-1):
            x[i]=(y[i]-U[i,i+1:]@x[i+1:])/U[i,i]
        return x

#testing values
B = np.array([[1,1,1],
              [2,1,-1],
              [2,-1,1]])
N=len(B)
b=np.array([-2,1,3])

solver = Solve(B,b,None)
L,U = solver.fatoraLU()
y=solver.resolveL(L,b)
x=solver.resolveU(U,y)

print(y,"\n")
print(L@y,"\n")
print(x,"\n")   #solution
print(L@U,"\n") #prints A
print(B@x,"\n") #prints b
