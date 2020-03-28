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

            #print(f"U_n {self.U_n}")



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

    M = 200
    N = 200
    frames=3
    a=4
    mu=0.5
    T=3

    poisson = PDE(f=lambda x: a*x,u_0=lambda x: np.sin(np.pi*x), mu=mu, T=T,)
    #print(poisson.f)

    import numpy.random as npr
    U = npr.randn(frames, M) + 15
    #poisson.U = U
    #poisson.plot2D()

    poisson.solver(M,N,frames=frames)
    #poisson.plot2D()
    #print((poisson.U_grid).shape)
    #print(poisson.U_grid)


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

    u = exact_solution(M=M, N=N, T=T, a=a, mu=mu)


    '''
    plt.plot(poisson.x, u[0,], label="$U_0$")
    #plt.plot(poisson.x,(poisson.U_grid)[1,], label="$U_1$")
    plt.plot(poisson.x, u[5,], label="$U_5$")
    plt.plot(poisson.x, u[10,], label="$U_{10}$")
    plt.plot(poisson.x, u[15,], label="$U_{15}$")
    plt.plot(poisson.x, u[N,], label="$U_N$")
    plt.title("Exact solution")
    plt.legend()
    plt.show()

    plt.plot(poisson.x,(poisson.U_grid)[0,], label="$U_0$")
    #plt.plot(poisson.x,(poisson.U_grid)[1,], label="$U_1$")
    plt.plot(poisson.x,(poisson.U_grid)[5,], label="$U_5$")
    plt.plot(poisson.x,(poisson.U_grid)[10,], label="$U_{10}$")
    plt.plot(poisson.x,(poisson.U_grid)[15,], label="$U_{15}$")
    plt.plot(poisson.x,(poisson.U_grid)[poisson.N,], label="$U_N$")
    plt.title("Numerical solution")
    plt.legend()
    plt.show()
    '''

    def convergence_test(pde, exact_solution, T, a, mu):
        step_num = 8 #Different stepsizes
        stepvec = np.zeros(step_num)
        errvec = np.zeros(step_num)
        M = 10
        N = 1000
        for i in range(step_num):
            pde.solver(M,N)
            err = pde.U_grid - exact_solution(M=M, N=N, T=T, a=a, mu=mu)

            stepvec[i] = pde.h
            errvec[i] = npl.norm(err.flatten(), np.inf)

            M *= 2
        order = np.polyfit(np.log(stepvec),np.log(errvec),1)[0]
        #print(f"Stepvec: {stepvec}")
        #print(f"Errvec: {errvec}")
        print(f"order: {order}")
        plt.loglog(stepvec,errvec)
        plt.show()


    def convergence_testT(pde, exact_solution, T, a, mu):
        step_num = 10 #Different stepsizes
        stepvec = np.zeros(step_num)
        errvec = np.zeros(step_num)
        M = 1000
        N = 10
        for i in range(step_num):
            pde.solver(M,N)
            err = pde.U_grid - exact_solution(M=M, N=N, T=T, a=a, mu=mu)

            #Normvec is a vector of the max norm of the spatial error for every timestep,
            #errvec contains the maxnorm of normvec, ie largest error in time
            stepvec[i] = pde.k
            normvec = np.zeros(M+1)
            for j in range(M+1):
                normvec[j] = npl.norm(err[:,j], np.inf)
            errvec[i] = npl.norm(normvec, np.inf)

            N *= 2
        order = np.polyfit(np.log(stepvec),np.log(errvec),1)[0]
        print(f"Stepvec: {stepvec}")
        print(f"Errvec: {errvec}")
        print(f"order: {order}")


    convergence_test(poisson,exact_solution, T, a, mu)

    #goob = np.array([1,4,7])
    #groog = np.arange(10)+5
    #print(groog[goob])


    '''
    error = np.abs(u-poisson.U_grid)
    errvec = np.zeros(N+1,dtype=float)
    for i in range(N+1):
        errvec[i] = npl.norm(error[i,])


    plt.plot(poisson.t,errvec)
    plt.show()
    '''
    #print(error[N,N])
