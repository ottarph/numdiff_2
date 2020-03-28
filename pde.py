import numpy as np

import numpy.linalg as npl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from scipy.sparse import identity, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized


class PDE:

    def __init__(self, f, u_0, mu, T, g_0=0, g_M=0, x_0=0, x_M=1, t_0=0):
        self.f = f
        self.u_0 = u_0 #initial value
        self.mu = mu
        self.T = T


        self.g_0 = g_0
        self.g_M = g_M
        self.x_0 = x_0
        self.x_M = x_M
        self.t_0 = t_0

    def solver(self, M, N):

        #Initializing the grid and stepsizes
        self.M = M
        self.x = np.linspace(self.x_0, self.x_M, M+1)
        self.h = (self.x_M-self.x_0)/M
        self.N = N
        self.t = np.linspace(self.t_0, self.T, N+1)
        self.k = (self.T-self.t_0)/N
        self.r = self.mu*self.k/self.h**2

        #initializing the vectors for U^n and U^*
        self.U_n = self.u_0(self.x)
        self.U_s = np.zeros((M+1), dtype = float)

        #Initializing the matrix for the numerical solution on the grid
        self.U_grid = np.zeros((N+1,M+1), dtype = float)
        self.U_grid[0,] += self.U_n


        #Construction of the system to find U^*
        #A = np.tridiag(-self.r*0.5,1+self.r,-self.r*0.5, M+1)
        A = np.zeros((M+1, M+1), dtype = float)
        A += np.diag(np.full(M,-self.r*0.5),-1)
        A += np.diag(np.full(M,-self.r*0.5),1)
        A += np.diag(np.full(M+1,1+self.r))

        B = np.zeros((M+1, M+1), dtype = float)
        B += np.diag(np.full(M,self.r*0.5),-1)
        B += np.diag(np.full(M,self.r*0.5),1)
        B += np.diag(np.full(M+1,1-self.r))

        #fixing boundary conditions for U^*
        A[0,0]=1
        A[0,1]=0
        A[M,M-1]=0
        A[M,M]=1

        for t in range(1,N+1):
            b = B @ self.U_n + self.k*self.f(self.U_n)

            b[0] = self.g_0
            b[M] = self.g_M

            self.U_s = npl.solve(A,b)

            #Computing U^(n+1)
            self.U_n = self.U_s + (self.k/2)*(self.f(self.U_s)-self.f(self.U_n))
            self.U_grid[t,] += self.U_n



    def plot2D(self, title="", x_skip=1, t_skip=1):
        #def plot2D(X, Y, Z, title=""):
        # Stolen from project in TMA4215 Numerisk Matematikk and modified


        x = np.linspace(self.x_0, self.x_M, self.M+1, dtype=float)[::x_skip]
        t = np.linspace(self.t_0, self.T, self.N+1, dtype=float)[::t_skip]
        X, Y = np.meshgrid(x,t)

        Z = self.U_grid[::t_skip,::x_skip]

        # Define a new figure with given size and dpi
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z,
                            rstride=1, cstride=1, # Sampling rates for the x and y input data
                            cmap=cm.viridis)      # Use the new fancy colormap viridis

        # Set initial view angle
        ax.view_init(30, 225)

        # Set labels and show figure
        ax.set_xlabel('$x$')
        ax.set_ylabel('$t$')
        ax.set_title(title)
        plt.show()
        

def relative_error_x(pde, exact_solution, T, a, mu, Ms):

    step_num = len(Ms)

    errvec = np.zeros(step_num)
    stepvec = np.zeros_like(errvec)

    N = 1000

    for i, M in enumerate(Ms):
        print(M)
        pde.solver(M,N)
        u = exact_solution(M=M, N=N, T=T, a=a, mu=mu)
        err = np.abs(pde.U_grid - u)
        err = err.flatten()

        Ind = np.argmax(err)
        assert u.flatten()[Ind] != 0, "u[Ind] == 0"
        e = err[Ind] / np.abs(u.flatten()[Ind])
        errvec[i] = e

        stepvec[i] = pde.h

    order = np.polyfit(np.log(stepvec),np.log(errvec),1)[0]
    print(f"order: {order}")
    plt.loglog(stepvec,errvec)
    plt.show()

def relative_error_t(pde, exact_solution, T, a, mu, Ns):

    step_num = len(Ns)

    errvec = np.zeros(step_num)
    stepvec = np.zeros_like(errvec)

    M = 1000

    for i, N in enumerate(Ns):
        print(N)
        pde.solver(M,N)
        u = exact_solution(M=M, N=N, T=T, a=a, mu=mu)
        err = np.abs(pde.U_grid - u)
        err = err.flatten()

        Ind = np.argmax(err)
        assert u.flatten()[Ind] != 0, "u[Ind] == 0"
        e = err[Ind] / np.abs(u.flatten()[Ind])
        errvec[i] = e

        stepvec[i] = pde.k

    order = np.polyfit(np.log(stepvec),np.log(errvec),1)[0]
    print(f"order: {order}")
    plt.loglog(stepvec,errvec)
    plt.show()


def exact_solution(M, N, T, a, mu): #Function for computing an exact solution, found using separation of variables

    x = np.linspace(0, 1, M+1)
    U_n = np.sin(np.pi*x)
    U = np.zeros((N+1,M+1), dtype = float)
    U[0,] += U_n
    #t = np.linspace(self.t_0, self.T, N+1)
    k = (T)/N
    #print(f"k:{k}")
    mult = np.exp((a-(np.pi)**2*mu)*k)
    #print(mult)
    for n in range(1,N+1):
        U_n *= mult
        U[n,] += U_n
    return U

def convergence_test_X(pde, exact_solution, T, a, mu):
    step_num = 6 #Different stepsizes
    stepvec = np.zeros(step_num)
    errvec = np.zeros(step_num)
    M = 10
    N = 1000
    for i in range(step_num):
        pde.solver(M,N)
        err = pde.U_grid - exact_solution(M=M, N=N, T=T, a=a, mu=mu)

        stepvec[i] = pde.h
        #taking the maximum norm of all gridpoints and timesteps
        errvec[i] = npl.norm(err.flatten(), np.inf)

        M *= 2
    order = np.polyfit(np.log(stepvec),np.log(errvec),1)[0]
    print(f"order: {order}")
    plt.loglog(stepvec,errvec)
    plt.show()


def convergence_test_T(pde, exact_solution, T, a, mu):
    step_num = 7 #Different stepsizes
    stepvec = np.zeros(step_num)
    errvec = np.zeros(step_num)
    M = 1000
    N = 10
    for i in range(step_num):
        pde.solver(M,N)
        err = pde.U_grid - exact_solution(M=M, N=N, T=T, a=a, mu=mu)

        stepvec[i] = pde.k
        #taking the maximum norm of all gridpoints and timesteps
        errvec[i] = npl.norm(err.flatten(), np.inf)

        N *= 2
    order = np.polyfit(np.log(stepvec),np.log(errvec),1)[0]
    print(f"order: {order}")

    


if __name__ == '__main__':

    M = 100
    N = 100
    a=4
    mu=0.5
    T=3

    poisson = PDE(f=lambda x: a*x,u_0=lambda x: np.sin(np.pi*x), mu=mu, T=T,)

    #convergence_test_X(poisson,exact_solution, T, a, mu)

    Ms = [32, 64, 128, 256]#, 512, 670, 700, 730, 800]
    #relative_error_x(poisson, exact_solution, T, a, mu, Ms)      
    relative_error_t(poisson, exact_solution, T, a, mu, Ms)
