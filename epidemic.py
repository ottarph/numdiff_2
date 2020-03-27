import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from scipy.sparse import identity, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized


class Epidemic:

    def __init__(self, beta, gamma, mu_s, mu_i, s0, i0):

        self.beta = beta # interaction constant between infected and susceptible
        self.gamma = gamma # Rate of infected population healing

        self.mu_s = mu_s # Diffusion constant for susceptible population
        self.mu_i = mu_i # Diffusion constant for infected population
        self.k_i = mu_i / mu_s # mu_s * k_i = mu_i

        self.s0 = s0 # Initial distribution of susceptible population as a function of x
        self.i0 = i0 # Initial distribution of infected population as a function of x

        self.f = lambda u: np.array([-beta/self.k_i * u[1] * u[0], # Reaction term in pde
                                beta/self.k_i * u[1] * u[0] - gamma/self.k_i * u[1]], dtype=float)

        self.u0 = lambda x: np.array([self.s0(x), self.i0(x)*self.k_i], dtype=float)

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
        #u[:,1] = u[:,1] / self.k_i
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

