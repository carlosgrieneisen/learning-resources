import numpy as np

class FatorarPorLU():
    '''
    1. obj = FatorarPorLU(A: matrix,b: vetor_solução)
    2. L,U = obj.fatoraLU() -> retorna tupla [L,U] ondeL@U = A
    3. y = obj.resolveL(L,b) -> retorna vetor que resolve para L
    4. x = obj.resolveU(U,y) -> retorna vetor que resolve para U
    5. obj.resolve() -> executa toda rotina.
    '''
    def __init__(self, A,b):
        self.A_o = A # matriz
        self.b_o = b # vetor
        self.N = len(A) # dimensão de A (matriz quadrada N x N)

    def fatoraLU(self):
        '''
        Retorna as matrizes triangulares Superiores e Inferiores
        '''
        # nomenclatura apenas
        U = np.array(self.A_o,dtype=float,copy=True)
        L = np.eye(self.N) #matriz identidade L, N x N
        # coluna
        for j in range(0,self.N-1):
            # linha abaixo do elemento A[j,j]
            # se fosse até N-1, cancela por causa do j do loop   
            for i in range(j+1,self.N): 
                L[i,j]= U[i,j]/U[j,j]
                U[i,:]=U[i,:]-L[i,j]*U[j,:]
        return L,U
    
    def resolveL(self,L,b):
        '''
        Retorna solução para matriz triangular inferior, solução de A@x = b do obj   
        '''
        N=self.N
        y=np.zeros(N)
        y[0]=b[0]/L[0,0]
        for i in range(1,N):
            y[i]=(b[i]-L[i,:i]@y[:i])/L[i,i]
        return y
    
    def resolveU(self,U,y):
        '''
        Retorna solução para matriz triangular superior, solução de A@x = b do obj   
        '''
        N=self.N
        x = np.zeros(N) 
        x[N-1]=y[N-1]/U[N-1,N-1]
        for i in range(N-2,-1,-1):
            x[i]=(y[i]-U[i,i+1:]@x[i+1:])/U[i,i]
        return x

    def resolve(self):
        L,U = self.fatoraLU()
        y = self.resolveL(L,self.b_o)
        x = self.resolveU(U,y)
        text= str(f"L: \n{L} ,\nU: \n{U}\nx: solução sistema: \n{x}\ny: solução de L: \n{y}\n")
        return text


#A@x = b, func inicial
A = np.array([[1,1,1], 
              [2,1,-1],
              [2,-1,1]])
b=np.array([-2,1,3])

func = FatorarPorLU(A,b)
print(func.resolve())
