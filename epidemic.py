import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from scipy.sparse import identity, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized


class Epidemic:
    #def __init__(self, f, u_0, mu, T, g_0=0, g_M=0, x_0=0, x_M=1, t_0=0)
    def __init__(self, beta, gamma, mu_s, mu_i, s0, i0, x_0, x_M):

        self.beta = beta # interaction constant between infected and susceptible
        self.gamma = gamma # Rate of infected population healing

        self.mu_s = mu_s # Diffusion constant for susceptible population
        self.mu_i = mu_i # Diffusion constant for infected population
        self.k_i = mu_i / mu_s # mu_s * k_i = mu_i

        self.s0 = s0 # Initial distribution of susceptible population as a function of x
        self.i0 = i0 # Initial distribution of infected population as a function of x

        # Domain
        self.x_0 = x_0
        self.x_M = x_M

        self.f = lambda u: np.array([-beta/self.k_i * u[1] * u[0], # Reaction term in pde
                                beta/self.k_i * u[1] * u[0] - gamma/self.k_i * u[1]], dtype=float)

        # initial state as function of x
        self.u0 = lambda x: np.array([self.s0(x), self.i0(x)*self.k_i], dtype=float) 


    def solver(self, M, N, frames=0):

        #Initializing the grid and stepsizes
        self.M = M
        self.x = np.linspace(self.x_0, self.x_M, M+1)
        self.h = (self.x_M - self.x_0) / M
        self.N = N
        self.t = np.linspace(self.t_0, self.T, N+1)
        self.k = (self.T - self.t_0) / N
        self.r = self.mu * self.k / self.h**2

        #initializing the vectors for U^n and U^*
        self.U_n = self.u_0(self.x)
        self.U_s = np.zeros((M+1), dtype = float)

        #Initializing the matrix for the numerical solution on the grid
        self.U_grid = np.zeros((N+1,M+1), dtype = float)
        self.U_grid[0,] += self.U_n



        #Construction of the system to find U^*
        #A = np.tridiag(-self.r*0.5,1+self.r,-self.r*0.5, M+1)
        A = np.zeros((M+1, M+1), dtype = float)
        A += np.diag(np.full(M, -self.r*0.5), -1)
        A += np.diag(np.full(M, -self.r*0.5), 1)
        A += np.diag(np.full(M+1, 1 + self.r))

        B = np.zeros((M+1, M+1), dtype = float)
        B += np.diag(np.full(M, self.r*0.5), -1)
        B += np.diag(np.full(M, self.r*0.5), 1)
        B += np.diag(np.full(M+1, 1 - self.r))

        # fixing boundary conditions for U^*
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

    def isolated_development(self, s0, i0, T, N):

        def rk4(f, u0, T, N):

            h = T / N
            
            u = np.copy(u0)
            us = np.empty((N+2, *u0.shape), dtype=float)
            us[0] = u0

            t = 0
            i = 0

            while t < T:

                k1 = f(u)
                k2 = f(u + 0.5*h*k1)
                k3 = f(u + 0.5*h*k2)
                k4 = f(u + h*k3)

                u += 1/6 * h * (k1 + 2*k2 + 2*k3 + k4)
                t += h
                i += 1

                us[i] = u
            
            return us
                

        f_iso = lambda u: np.array([-self.beta * u[1] * u[0], # Reaction term in pde
                                    self.beta * u[1] * u[0] - self.gamma * u[1]], dtype=float)
        
        u0_iso = np.array([s0, i0], dtype=float)
        u = rk4(f_iso, u0_iso, T, N)
        s = u[:,0]
        i = u[:,1]
        r = s0 + i0 - s - i
        t = np.linspace(0, T, u.shape[0])
        n = s + i + r
        plt.plot(t, s, label='Susceptible')
        plt.plot(t, i, label='Infected')
        plt.plot(t, r, label='Recovered')
        plt.plot(t, n, label='Population')
        plt.axhline(y=0, color='black', linewidth=0.4)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    plague = Epidemic(beta=3, gamma=1, mu_s=0.1, mu_i=0.05, s0=lambda x: 0*x + 1, i0=lambda x: 0*x + 0.5)

    plague.isolated_development(1, 0.05, 10, 100)

