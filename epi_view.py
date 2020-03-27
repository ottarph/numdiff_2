import numpy as np
import matplotlib.pyplot as plt

def rk4(f, u0, T, N):

    h = T / N
    
    u = np.copy(u0)
    us = np.empty((N+2, *u0.shape), dtype=float)
    print(us.shape)
    us[0] = u0

    t = 0
    i = 0

    while t < T:

        k1 = f(t, u)
        k2 = f(t + 0.5*h, u + 0.5*h*k1)
        k3 = f(t + 0.5*h, u + 0.5*h*k2)
        k4 = f(t + h, u + h*k3)

        u += 1/6 * h * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        i += 1

        us[i] = u
    
    return us

beta = 3
gamma = 1

f = lambda t, u: np.array([-beta * u[1] * u[0], beta* u[1] * u[0] - gamma * u[1]], dtype=float)

'''
x = np.linspace(0, 1, 100)
u0 = np.array([np.ones_like(x), 0.5* np.ones_like(x)])
plt.plot(x, u0[0,:], label='$t = 0$')
plt.plot(x, u0[1,:], label='$t = 0$')

T = 1
u = rk4(f, u0, 0.05, 1)[-1]
print(u.shape)
plt.plot(x, u[0,:], label=f'$t = {T}$')
plt.plot(x, u[1,:], label=f'$t = {T}$')
plt.legend()
#plt.show()
'''

plt.figure()

T = 3
u0 = np.array([1,0.5], dtype=float)
u = rk4(f, u0, T, 100)
t = np.linspace(0, T, u.shape[0])
plt.plot(t, u[:,0], label='Susceptible')
plt.plot(t, u[:,1], label='Infected')
plt.legend()
plt.show()
