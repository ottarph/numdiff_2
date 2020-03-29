#This file contains the code used for making the plots in the paper

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

import pde
import pde_sparse
import epidemic


#For section: A Dirichlet problem on the unit interval
M = 200
N = 200
mu = 0.5
a = 3
T=1

eq = pde_sparse.PDE(f=lambda x: a*x,u_0=lambda x: np.sin(np.pi*x), mu=mu, T=T,)
eq.solver(M,N)

#plt.plot(eq.x,eq.U_grid[15], label="U_{15}")
exact = pde_sparse.exact_solution(M, N, T, a, mu)


plt.plot(eq.x,eq.U_grid[0], label="$U_{0}$")
plt.plot(eq.x,eq.U_grid[50], label="$U_{50}$")
plt.plot(eq.x,eq.U_grid[100], label="$U_{100}$")
plt.plot(eq.x,eq.U_grid[150], label="$U_{150}$")
plt.plot(eq.x,eq.U_grid[200], label="$U_{200}$")
plt.title("Numerical solution")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.legend()
plt.show()

plt.plot(eq.x,exact[0], label="$U_{0}$")
plt.plot(eq.x,exact[50], label="$U_{500}$")
plt.plot(eq.x,exact[100], label="$U_{1000}$")
plt.plot(eq.x,exact[150], label="$U_{1500}$")
plt.plot(eq.x,exact[200], label="$U_{2000}$")
plt.title("Exact solution")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.legend()
plt.show()

'''
Ns = [32, 64, 128, 256]
pde_sparse.convergence_test_T(eq, pde_sparse.exact_solution, T, a, mu)
pde_sparse.relative_error_t(eq, pde_sparse.exact_solution, T, a, mu, Ns)
'''

'''
print(f"k = {eq.k}")
phi = eq.k*mu/2
print(f"phi = {phi}")
'''
