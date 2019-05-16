import numpy as np


def norm_p(x, p=2):
    assert p >= 1
    return sum(x**p) ** (1/p)


class RosenbrockProvider:
    """
    Rosenbrock function and it's gradient and hessian matrix
    """
    @staticmethod
    def f(x1, x2):
        return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

    @staticmethod
    def grad(x1, x2):
        g1 = -400 * (x2 - x1 ** 2) * x1 + 2 * (x1 - 1)
        g2 = 200 * (x2 - x1 ** 2)
        return np.array([g1, g2])

    @staticmethod
    def hessian(x1, x2):
        return np.array([
            [400 * (3 * x1 ** 2 - x2) + 2, -400 * x1],
            [-400 * x1, 200]
        ])
