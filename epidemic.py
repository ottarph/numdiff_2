import numpy as np

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
        


if __name__ == '__main__':
    plague = Epidemic(beta=3, gamma=1, mu_s=0.1, mu_i=0.05, s0=lambda x: 0*x + 1, i0=lambda x: 0*x + 0.5)

