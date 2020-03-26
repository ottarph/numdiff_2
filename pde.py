import numpy as np

import numpy.linalg as npl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from scipy.sparse import identity, dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized


class PDE:

    def __init__(self, f, mu, M, N, T, x_0=0, x_M=1, t_0=0, frames=0):
        self.f = f
        self.mu = mu

        self.M = M
        self.h = (x_M - x_0) / M
        self.N = N
        self.k = (T - t_0) / N

        self.x_0 = x_0
        self.x_M = x_M
        self.t_0 = t_0
        self.T = T

        self.U_n = np.zeros((M+1), dtype=float)
        self.U_s = np.zeros((M+1), dtype=float)

        self.frames = frames
        # Amount of frames to keep, if frames=0, don't declare the storage
        if frames > 0:
            self.U = np.zeros((frames, M+1), dtype=float)


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
    poisson = PDE(f=lambda x, y: x**2 - y**2, mu=1, M=M, N=N, T=3, frames=frames)
    print(poisson.f)

    import numpy.random as npr
    U = npr.randn(frames, M) + 15
    #poisson.U = U
    poisson.plot2D()
