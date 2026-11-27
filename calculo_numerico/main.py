import numpy as np
import matplotlib.pyplot as plt
import time 

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

    def resolve(self, complete=False):
        L,U = self.fatoraLU()
        y = self.resolveL(L,self.b_o)
        x = self.resolveU(U,y)
        if complete == True:
            text= str(f"L: \n{L} ,\nU: \n{U}\nx: solução sistema: \n{x}\ny: solução de L: \n{y}\n")
        else:
            text= str(f"x: solução sistema: \n{x}")
        return text

class TDMA():
    '''
    Tridiagonal Matrix Algorithm. Forma simplificada para resolver sistemas lineares de matriz tridiagonais por eliminação gaussiana.
    '''
    def __init__(self, A,b):
        self.A=A
        self.b=b
        self.dim=len(A)
    
    def extrai_diagonais(self):
        diagonal_principal = [self.A[0, 0]]
        diagonal_inferior = []
        diagonal_superior = [self.A[0, 1]]
        for i in range(1, self.dim - 1):
            diagonal_inferior.append(self.A[i, i - 1])
            diagonal_principal.append(self.A[i, i])
            diagonal_superior.append(self.A[i, i + 1])
        if self.dim > 1:
            diagonal_inferior.append(self.A[self.dim - 1, self.dim - 2])
            diagonal_principal.append(self.A[self.dim - 1, self.dim - 1])
        return np.array(diagonal_principal, dtype=float),np.array(diagonal_inferior, dtype=float),np.array(diagonal_superior, dtype=float)
    
    def resolve(self):
        n = self.dim
        b, a_sub, c = self.extrai_diagonais()
        a = np.concatenate(([0.0], a_sub))
        d = np.array(self.b, dtype=float, copy=True)
        x = np.zeros(n)
        # calcula cl e dl sobrescrevendo-os em c e d
        c[0] = c[0] / b[0]
        for i in range(1, n - 1):
            c[i] = c[i] / (b[i] - a[i] * c[i - 1])
        d[0] = d[0] / b[0]
        for i in range(1, n):
            d[i] = (d[i] - a[i] * d[i - 1]) / (b[i] - a[i] * c[i - 1])
        # faz substituicao reversa para obter a solucao x
        x[n - 1] = d[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = d[i] - c[i] * x[i + 1]
        return x

class Condicionamento():
    '''
    Ache o número de condicionamento de uma matriz singular A
    Para que o sistema não tenha solução única, a det da matriz deve ser diferente de 0
    Para p diferente {1,2,inf} use np.linalg.norm(A,p)
    '''
    def __init__(self,A):
        self.A=A
        self.Ai=inverteMatriz(A)
        self.n=len(self.A)
       
    def valor(self, p=1):
        nAp,nAip=normaMatriz(self.A,p),normaMatriz(self.Ai,p)
        k=nAp * nAip
        return k

    def avaliar(self, p=1):
        k = self.valor(p)
        if k <= 10**3:
            text = "confiável, perde até 3 dígitos de precisão"
        elif k <= 10**5:
            text = "moderado, perde de 4 a 5 dígitos de precisão"
        elif k < 10**16:
            text = "mal condicionado, solução sensível/instável"
        else:
            text = "crítico, nenhum dígito confiável, matriz praticamente singular"
        return text + "\n"

class MetodosIterativos:
    '''
    Métodos de resolução de matrizes de forma iterativa, gauss-seidel, jacobi. Tipicamente por aproximação.
    '''
    def __init__(self,A,b,x):
        self.A=A
        self.b=b
        self.x=x
        self.N=len(A)
        self.x0=np.zeros(self.N)

    def jacobi(self, max_iter=50,tol=1e-6):
        x = np.copy(self.x0)
        xnew = np.zeros(self.N)
        historico_res = []

        for it in range(max_iter):
            for i in range(self.N):
                xnew[i] = (self.b[i] - self.A[i, :i] @ x[:i] - self.A[i, i+1:] @ x[i+1:]) / self.A[i, i]

            x = np.copy(xnew)
            res = self.b - self.A @ x
            normr = normaVetor(res,p=2)
            historico_res.append(normr)
            if normr < tol: 
                print(f"o sistema convergiu após {it + 1} iterações, com tolerância {tol}")
                break
        return x, historico_res
    
    def gauss_seidel(self, max_iter=50, tol=1e-6):
        n = len(self.b)
        x = np.copy(self.x0)
        historico_res = []

        for it in range(max_iter):
            for i in range(n):
                x[i] = (self.b[i] - self.A[i, :i] @ x[:i] - self.A[i, i+1:] @ x[i+1:]) / self.A[i, i]

            res = self.b - self.A @ x
            normr = normaVetor(res,p=2)
            historico_res.append(normr)
            if normr < tol: 
                print(f"o sistema convergiu após {it + 1} iterações, com tolerância {tol}")
                break
        return x, historico_res

    def analise_convergencia(self,max_iter=50,tol=1e-6,plotar=True):
        '''
        1. Compara resolução de ambos métodos de resolução. É possível passar números de iteração customizados, tolerância e se irá plotar ou não respectivamente.
        2. _analise_convergencia(self, max_iter: int, tol=int, plotar: bool)
        '''
        resultados = {}
        for nome, metodo in [("Jacobi", self.jacobi), ("Gauss-Seidel", self.gauss_seidel)]:
            t0 = time.perf_counter()
            x, hist = metodo(max_iter=max_iter, tol=tol)
            tempo_ms = (time.perf_counter() - t0) * 1000
            taxa = (hist[-1] / hist[0]) ** (1 / len(hist)) if len(hist) > 0 and hist[0] > 0 else None
            resultados[nome] = {
                "convergiu": bool(hist and hist[-1] < tol),
                "iteracoes": len(hist),
                "residuo_inicial": hist[0] if hist else None,
                "residuo_final": hist[-1] if hist else None,
                "taxa_media_reducao": taxa,
                "tempo_ms": tempo_ms,
                "solucao": x,
                "historico": hist,
            }
        if plotar:
            plt.figure(figsize=(8, 5))
            for nome, dados in resultados.items():
                it = range(1, dados["iteracoes"] + 1)
                plt.semilogy(it, dados["historico"], marker="o", markersize=3, label=f"{nome} ({dados['iteracoes']} it.)")
            plt.axhline(tol, color="red", linestyle="--", label=f"Tolerância ({tol})")
            plt.xlabel("Iteração")
            plt.ylabel("Norma do resíduo ||b - Ax||₂ (escala log)")
            plt.title("Convergência: Jacobi vs Gauss-Seidel")
            plt.grid(True, which="both", linestyle=":", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.show()
        return resultados
    
    def analise_convergencia_teorica(self):
        D,L,U=destrincharMatriz(self.A)
        T_J = -inverteMatriz(D) @ (L + U)
        T_GS = -inverteMatriz(D + L) @ U
        info = {}
        for nome, T in [("Jacobi", T_J), ("Gauss-Seidel", T_GS)]:
            rho = max(abs(np.linalg.eigvals(T)))
            info[nome] = {
                "raio_espectral": rho,
                "converge": rho < 1,
                "norma_1": normaMatriz(T, 1),
                "norma_inf": normaMatriz(T, np.inf),
                "T": T,
            }
        return info
        
def normaVetor(vetor, p=1):
    '''
    vetor, p em L**p
    np.linalg.norm(matriz,grau) para matriz
    '''
    if p == np.inf:
        return max(abs(i) for i in vetor)
    somatorio = 0.0
    for i in vetor:
        somatorio += abs(i) ** p
    return somatorio ** (1 / p)

def normaMatriz(matriz, p=1):
    n = len(matriz)
    maior = 0
    if p == 1:
        for j in range(n):
            soma = 0
            for i in range(n):
                soma += abs(matriz[i, j])
            if soma > maior:
                maior = soma

    elif p == 2:
        B = matriz.T @ matriz
        return np.sqrt(max(np.linalg.eigvals(B).real))
    
    elif p == np.inf:
        for i in range(n):
            soma = 0
            for j in range(n):
                soma += abs(matriz[i, j])
            if soma > maior:
                maior = soma
    else:
        raise ValueError("Use p=1 ou p=np.inf")
    return maior

def inverteMatriz(A):
    n=len(A)
    invertindo = FatorarPorLU(A,np.zeros(n))
    L,U=invertindo.fatoraLU()
    I=np.eye(n)
    Ai=np.zeros((n,n))
    for j in range(n):
        y=invertindo.resolveL(L,I[:,j])
        Ai[:, j] = invertindo.resolveU(U,y)
    return Ai

def destrincharMatriz(A):
    n = len(A)
    L = np.zeros((n, n))
    D = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i, j] = A[i, j]
            elif i == j:
                D[i, j] = A[i, j]
            else:
                U[i, j] = A[i, j]
    return D,L,U
''' 
#A@x = b, func inicial
A1 = np.array([[1,1,1], 
              [2,1,-1],
              [2,-1,1]])
b1=np.array([-2,1,3])
func = FatorarPorLU(A1,b1)
print(func.resolve())

A2 = np.array([[2,1,0,0,0], 
               [1,2,1,0,0],
               [0,1,2,1,0],
               [0,0,1,2,1],
               [0,0,0,1,2]])
b2 = np.array( [4,4,0,0,2])
func2=TDMA(A2,b2)
print(func2.resolve())

A3=np.array([[5,4,0,0,0,0],
             [1,3,1,0,0,0],
             [0,2,4,1,0,0],
             [0,0,1,2,1,0],
             [0,0,0,2,3,2],
             [0,0,0,0,1,2]])

b3=np.array([13,10,20,16,35,17])
func3=TDMA(A3,b3)
print("\n\n",func3.resolve())


A1=np.array([[71,41],
             [51,30]])

A2=np.array([[3,-5,7],
             [1,-2,4],
             [-8,1,-7]])

cond1=Condicionamento(A1)
cond2=Condicionamento(A2)

print(cond1.valor(),"\n",cond1.valor(2),"\n",cond1.valor(np.inf),"\n",cond1.avaliar())
print("\n",cond2.valor(),"\n",cond2.valor(2),"\n",cond2.valor(np.inf),"\n",cond2.avaliar())


A3=np.array([[100000, -9999.99],
             [-9999.99, 1000.1]])
b3= np.array([-10, 1])
b4= np.array([-9.999, 1.01])

func3_r=FatorarPorLU(A3,b3)
func3_a=Condicionamento(A3)
func4_r=FatorarPorLU(A3,b4)

print(b3,"\n",func3_r.resolve(),"\n","\n",func3_a.valor(),func3_a.avaliar())
print(b4,"\n",func4_r.resolve(),"\n","\n",func3_a.valor(),func3_a.avaliar())

A3=np.array([[5,4,0,0,0,0],
             [1,3,1,0,0,0],
             [0,2,4,1,0,0],
             [0,0,1,2,1,0],
             [0,0,0,2,3,2],
             [0,0,0,0,1,2]])
b3=np.array([13,10,20,16,35,17])
it_solver=MetodosIterativos(A3,b3,None)
res = it_solver.analise_convergencia(max_iter=100, tol=1e-16)
'''