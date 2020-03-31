import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from scipy.sparse import identity, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized


class Epidemic:
    
    def __init__(self, beta, gamma, mu_s, mu_i, T, s0, i0, x_0=0, x_M=1, t_0=0, g_0=0, g_M=0):

        self.beta = beta # interaction constant between infected and susceptible
        self.gamma = gamma # Rate of infected population healing

        self.mu_s = mu_s # Diffusion constant for susceptible population
        self.mu_i = mu_i # Diffusion constant for infected population

        self.s0 = s0 # Initial distribution of susceptible population as a function of x
        self.i0 = i0 # Initial distribution of infected population as a function of x

        # Domain
        self.x_0 = x_0
        self.x_M = x_M
        self.t_0 = t_0
        self.T = T

        ## Dirichlet boundary conditions
        # Neumann boundary conditions
        self.g_0 = g_0
        self.g_M = g_M
      
        # Reaction terms
        self.f_s = lambda i, s: -self.beta * i * s
        self.f_i = lambda i, s: self.beta * i * s - self.gamma * i


    def solver(self, M, N):

        #Initializing the grid and stepsizes
        self.M = M
        self.x = np.linspace(self.x_0, self.x_M, M+1)
        self.h = (self.x_M - self.x_0) / M
        self.N = N
        self.t = np.linspace(self.t_0, self.T, N+1)
        self.k = (self.T - self.t_0) / N
        
        self.r_i = self.mu_i * self.k / self.h**2
        self.r_s = self.mu_s * self.k / self.h**2

        #initializing the vectors for i^n,s^n and i^*,s^*
        self.I_n = self.i0(self.x)
        self.I_s = np.zeros((M+1), dtype = float)
        self.S_n = self.s0(self.x)
        self.S_s = np.zeros((M+1), dtype = float)

        #Initializing the matrix for the numerical solution on the grid
        self.I_grid = np.zeros((N+1,M+1), dtype = float)
        self.I_grid[0,] += self.I_n
        self.S_grid = np.zeros((N+1,M+1), dtype = float)
        self.S_grid[0,] += self.S_n



        #Construction of the system to find U^*
        A_i = dok_matrix((M+1, M+1))
        B_i = dok_matrix((M+1, M+1))
        A_s = dok_matrix((M+1, M+1))
        B_s = dok_matrix((M+1, M+1))
        for i in range(1,M):
            A_i[i,i-1] = -self.r_i * 0.5
            A_i[i,i] = 1 + self.r_i 
            A_i[i,i+1] = -self.r_i * 0.5

            B_i[i,i-1] = self.r_i * 0.5
            B_i[i,i] = 1 - self.r_i 
            B_i[i,i+1] = self.r_i * 0.5

            A_s[i,i-1] = -self.r_s * 0.5
            A_s[i,i] = 1 + self.r_s 
            A_s[i,i+1] = -self.r_s * 0.5

            B_s[i,i-1] = self.r_s * 0.5
            B_s[i,i] = 1 - self.r_s 
            B_s[i,i+1] = self.r_s * 0.5

        B_s[0,0] = 1 - self.r_s
        B_s[0,1] = self.r_s * 0.5
        B_s[M,M] = 1 - self.r_s
        B_s[M,M-1] = self.r_s * 0.5

        B_i[0,0] = 1 - self.r_i
        B_i[0,1] = self.r_i * 0.5
        B_i[M,M] = 1 - self.r_i
        B_i[M,M-1] = self.r_i * 0.5
        

        #First order Neumann boundary conditions for U^*
        A_i[0,0]=-1
        A_i[0,1]=1
        A_i[M,M-1]=-1
        A_i[M,M]=1

        A_s[0,0]=-1
        A_s[0,1]=1
        A_s[M,M-1]=-1
        A_s[M,M]=1

        solve_i = factorized(A_i.tocsc())
        solve_s = factorized(A_s.tocsc())

        for t in range(1,N+1):

            b_i = B_i @ self.I_n + self.k*self.f_i(self.I_n, self.S_n)
            b_s = B_s @ self.S_n + self.k*self.f_s(self.I_n, self.S_n)

            b_i[0] = self.g_0
            b_i[M] = self.g_M

            b_s[0] = self.g_0
            b_s[M] = self.g_M

            self.I_s = solve_i(b_i)
            self.S_s = solve_s(b_s)

            #Computing U^(n+1)
            self.I_n = self.I_s + (self.k/2) * (self.f_i(self.I_s, self.S_s) - self.f_i(self.I_n, self.S_n))
            self.S_n = self.S_s + (self.k/2) * (self.f_s(self.I_s, self.S_s) - self.f_s(self.I_n, self.S_n))

            #self.U_grid[t,] += self.U_n
            self.I_grid[t,] += self.I_n
            self.S_grid[t,] += self.S_n


        
    def plot2D(self, title_i="Infected", title_s="Susceptible", x_skip=1, t_skip=1, show=True):
        #def plot2D(X, Y, Z, title=""):
        # Stolen from project in TMA4215 Numerisk Matematikk and modified

        
        x = np.linspace(self.x_0, self.x_M, self.M+1, dtype=float)[::x_skip]
        t = np.linspace(self.t_0, self.T, self.N+1, dtype=float)[::t_skip]
        X, Y = np.meshgrid(x,t)

        Z = self.S_grid[::t_skip,::x_skip]

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
        ax.set_title(title_s)

        Z = self.I_grid[::t_skip,::x_skip]

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
        ax.set_title(title_i)


        if show:
            plt.show()

    def curve(self, show_removed=False, hospital=False, show=True, title=""):

        t = np.linspace(self.t_0, self.T, self.N+1, dtype=float)
        S = np.sum(self.S_grid, axis=1)
        I = np.sum(self.I_grid, axis=1)
        N = S[0] + I[0]
        if show_removed:
            R = N - S - I

        plt.figure()
        plt.plot(t, S, 'k-', label='$S(t)$')
        plt.plot(t, I, 'k--', label='$I(t)$')
        if show_removed:
            plt.plot(t, R, color='black', linestyle='dotted', label='$R(t)$')
        plt.axhline(y=0, linewidth=1, color='black')
        if hospital:
            plt.axhline(y=hospital*N, linestyle='dotted', color='black', linewidth=1, label='Hospital capacity')
        plt.legend(frameon=False, fontsize=13)
        plt.xlim(self.t_0, self.T)
        plt.xlabel('$t$', fontsize=13)
        plt.title(title)
        if show:
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

    i0 = lambda x: np.sin(np.pi*x)**2 * 0.3
    s0 = lambda x: np.sin(np.pi*x)**2 * 0.7
    beta = 3
    gamma = 1
    mu = 0.1
    T = 10

    plague = Epidemic(beta=beta, gamma=gamma, mu_s=mu, mu_i=mu, T=T,
                        s0=s0, i0=i0)


    M = 64
    N = 64
    plague.solver(M, N)
    #plague.plot2D()
    plague.curve()

