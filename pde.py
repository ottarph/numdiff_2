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

    def solver(self, M, N, frames=0):

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


        self.frames = frames
        # Amount of frames to keep, if frames=0, don't declare the storage
        if frames > 0:
            self.U = np.zeros((frames, M+1), dtype=float)

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

            print(f"U_n {self.U_n}")



    def plot2D(self, title=""):
        #def plot2D(X, Y, Z, title=""):
        # Stolen from project in TMA4215 Numerisk Matematikk and modified

        assert self.frames > 0, "Don't have any stored frames, try a 1D plot"

        x = np.linspace(self.x_0, self.x_M, self.M+1, dtype=float)
        t = np.linspace(self.t_0, self.T, self.frames, dtype=float)
        X, Y = np.meshgrid(x,t)

        print(X.shape)
        print(Y.shape)

        Z = self.U
        print(Z.shape)

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


if __name__ == '__main__':

    M = 20
    N = 20
    frames=3
    poisson = PDE(f=lambda x: 10*x,u_0=lambda x: np.sin(np.pi*x), mu=1, T=3,)
    #print(poisson.f)

    import numpy.random as npr
    U = npr.randn(frames, M) + 15
    #poisson.U = U
    #poisson.plot2D()

    poisson.solver(M,N,frames=frames)
    #poisson.plot2D()
    #print((poisson.U_grid).shape)
    #print(poisson.U_grid)

    plt.plot(poisson.x,(poisson.U_grid)[0,], label="$U_0$")
    #plt.plot(poisson.x,(poisson.U_grid)[1,], label="$U_1$")
    plt.plot(poisson.x,(poisson.U_grid)[5,], label="$U_5$")
    plt.plot(poisson.x,(poisson.U_grid)[10,], label="$U_{10}$")
    plt.plot(poisson.x,(poisson.U_grid)[15,], label="$U_{15}$")
    plt.plot(poisson.x,(poisson.U_grid)[poisson.N,], label="$U_N$")
    #plt.plot(self.x,self.U_s)
    plt.legend()
    plt.show()
