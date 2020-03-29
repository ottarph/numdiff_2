from epidemic import *

norm = lambda x: np.exp(-x**2 / 2) / np.sqrt(2*np.pi)

# cpa
plt.figure()
beta = 7
gamma = 5
mu_s = 0.1
mu_i = 0.02
T = 10
S0 = 0.99
I0 = 0.01
s0 = lambda x: 0*x + S0
i0 = lambda x: I0 * np.sin(np.pi * norm(10*(x-0.5)))**2 / 0.19135259495744814

epi = Epidemic(beta, gamma, mu_s, mu_i, T, s0, i0)
M, N = 100, 200
epi.solver(M, N)
epi.curve(show_removed=False, hospital=0.05, show=False, title="cpa")
#plt.savefig("cpa.pdf", format='pdf')

# cpb
plt.figure()
beta = 8
gamma = 5
mu_s = 0.1
mu_i = 0.02
T = 10
S0 = 0.99
I0 = 0.01
s0 = lambda x: 0*x + S0
i0 = lambda x: I0 * np.sin(np.pi * norm(10*(x-0.5)))**2 / 0.19135259495744814

epi = Epidemic(beta, gamma, mu_s, mu_i, T, s0, i0)
M, N = 100, 200
epi.solver(M, N)
epi.curve(show_removed=False, hospital=0.05, show=False, title="cpb")

plt.show()
