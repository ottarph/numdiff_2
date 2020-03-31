from epi_sparse import *

save = False

norm = lambda x: np.exp(-x**2 / 2) / np.sqrt(2*np.pi)

# cpa
#plt.figure()
beta = 7
gamma = 5
mu_s = 0.1
mu_i = 0.02
T = 10
S0 = 0.99
I0 = 0.01
M, N = 100, 200
s0 = lambda x: 0*x + S0
i0 = lambda x: I0 * np.sin(np.pi * norm(10*(x-0.5)))**2 / 0.19135259495744814
title = ("cpa", "")[save]

epi = Epidemic(beta, gamma, mu_s, mu_i, T, s0, i0)
epi.solver(M, N)
epi.curve(show_removed=False, hospital=0.05, show=False, title=title)
if save: plt.savefig("figures/cpa.pdf", format='pdf')


# cpb
#plt.figure()
beta = 8
gamma = 5
mu_s = 0.1
mu_i = 0.02
T = 10
S0 = 0.99
I0 = 0.01
s0 = lambda x: 0*x + S0
i0 = lambda x: I0 * np.sin(np.pi * norm(10*(x-0.5)))**2 / 0.19135259495744814
title = ("cpb", "")[save]

epi = Epidemic(beta, gamma, mu_s, mu_i, T, s0, i0)
M, N = 100, 200
epi.solver(M, N)
epi.curve(show_removed=False, hospital=0.05, show=False, title=title)
if save: plt.savefig("figures/cpb.pdf", format='pdf')

'''

# cpc
#plt.figure()
beta = 7
gamma = 4
mu_s = 0.1
mu_i = 0.02
T = 10
S0 = 0.99
I0 = 0.01
s0 = lambda x: 0*x + S0
i0 = lambda x: I0 * np.sin(np.pi * norm(10*(x-0.5)))**2 / 0.19135259495744814
title = ("cpc", "")[save]

epi = Epidemic(beta, gamma, mu_s, mu_i, T, s0, i0)
M, N = 100, 200
epi.solver(M, N)
epi.curve(show_removed=False, hospital=0.05, show=False, title=title)
if save: plt.savefig("figures/cpc.pdf", format='pdf')


# cpd
#plt.figure()
beta = 8
gamma = 4
mu_s = 0.1
mu_i = 0.02
T = 10
S0 = 0.99
I0 = 0.01
s0 = lambda x: 0*x + S0
i0 = lambda x: I0 * np.sin(np.pi * norm(10*(x-0.5)))**2 / 0.19135259495744814
#plt.figure()
#x = np.linspace(0, 1, M+1)
#plt.plot(x, i0(x), 'k-')
title = ("cpd", "")[save]

epi = Epidemic(beta, gamma, mu_s, mu_i, T, s0, i0)
M, N = 100, 200
epi.solver(M, N)
epi.curve(show_removed=False, hospital=0.05, show=False, title=title)
if save: plt.savefig("figures/cpd.pdf", format='pdf')


# cpe
#plt.figure()
beta = 7
gamma = 5
mu_s = 0.1
mu_i = 0.02
T = 10
S0 = 0.99
I0 = 0.01
s0 = lambda x: 0*x + S0
i0 = lambda x: I0 * np.sin(np.pi * norm(10*(x-0.5)))**2 / 0.19135259495744814
i0 = lambda x: np.heaviside(x - 0.9, 1/0.1) * np.sin(np.pi*(x-0.9)/0.1)**2 * 4 * I0
i0 = lambda x: np.sin(np.pi*4*x)**2 * I0
#plt.figure()
#x = np.linspace(0, 1, M+1)
#plt.plot(x, i0(x), 'k-')
title = ("cpe", "")[save]

epi = Epidemic(beta, gamma, mu_s, mu_i, T, s0, i0)
M, N = 100, 200
epi.solver(M, N)
#epi.plot2D(x_skip=4, t_skip=4, show=False)
epi.curve(show_removed=False, hospital=0.05, show=False, title=title)
if save: plt.savefig("figures/cpe.pdf", format='pdf')

'''

# cpf
#plt.figure()
beta = 8
gamma = 5
mu_s = 0.02
mu_i = 0.0
T = 10
S0 = 0.99
I0 = 0.01
M, N = 100, 200
s0 = lambda x: 0*x + S0
i0 = lambda x: I0 * np.sin(np.pi * norm(10*(x-0.5)))**2 / 0.19135259495744814
title = ("cpf", "")[save]

epi = Epidemic(beta, gamma, mu_s, mu_i, T, s0, i0)
epi.solver(M, N)
epi.curve(show_removed=False, hospital=0.05, show=False, title=title)
if save: plt.savefig(f"figures/cpf.pdf", format='pdf')
epi.plot2D(x_skip=1, t_skip=1, show=False, title_i="")
if save: plt.savefig(f"figures/cpf_inf.pdf", format='pdf')


# cpg
#plt.figure()
beta = 8
gamma = 5
mu_s = 0.05
mu_i = 0.025
T = 10
S0 = 0.99
I0 = 0.01
M, N = 100, 200
s0 = lambda x: 0*x + S0
i0 = lambda x: I0 * np.sin(np.pi * norm(10*(x-0.5)))**2 / 0.19135259495744814
title = ("cpg", "")[save]

epi = Epidemic(beta, gamma, mu_s, mu_i, T, s0, i0)
epi.solver(M, N)
epi.curve(show_removed=False, hospital=0.05, show=False, title=title)
if save: plt.savefig(f"figures/cpg.pdf", format='pdf')
epi.plot2D(x_skip=1, t_skip=1, show=False, title_i="")
if save: plt.savefig(f"figures/cpg_inf.pdf", format='pdf')

plt.show()
