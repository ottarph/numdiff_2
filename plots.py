#This file contains the code used for making the plots in the paper

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

import pde
import pde_sparse
import epidemic


#For section: A Dirichlet problem on the unit interval
M = 100
N = 100
mu = 0.5
a = 3
T=1

eq = pde_sparse.PDE(f=lambda x: a*x,u_0=lambda x: np.sin(np.pi*x), mu=mu, T=T,)
eq.solver(M,N)

#plt.plot(eq.x,eq.U_grid[15], label="U_{15}")
exact = pde_sparse.exact_solution(M, N, T, a, mu)
err = np.abs(exact-eq.U_grid)

#Numerical solution
for i in [0,25,50,75,100]:
    lab = "$U^{" + str(i) + "}$"
    plt.plot(eq.x, eq.U_grid[i], label=lab)
#plt.title("Numerical solution")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlabel("x")
plt.ylabel("U")
plt.legend()
plt.show()

#absolute error
for i in [0,25,50,75,100]:
    lab = "$U^{" + str(i) + "}$"
    plt.plot(eq.x, err[i], label=lab)
#plt.title("Error")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlabel("x")
plt.ylabel("|u-U|")
plt.legend()
plt.show()

'''
#exact solution
for i in [0,25,50,75,100]:
    lab = "$U^{" + str(i) + "}$"
    plt.plot(eq.x, exact[i], label=lab)
plt.title("Exact solution")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlabel("x")
plt.ylabel("U")
plt.legend()
plt.show()
'''

'''
Ns = [32, 64, 128, 256]
pde_sparse.convergence_test_T(eq, pde_sparse.exact_solution, T, a, mu)
pde_sparse.relative_error_t(eq, pde_sparse.exact_solution, T, a, mu, Ns)
'''
