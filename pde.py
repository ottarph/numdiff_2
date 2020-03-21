

class PDE:

    def __init__(self, f, mu):
        self.f = f
        self.mu = mu


if __name__ == '__main__':

    poisson = PDE(lambda x, y: x**2 - y**2, 1)
    print(poisson.f)
    