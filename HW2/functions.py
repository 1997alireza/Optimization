import numpy as np


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


class LeastSquares:
    """
    Least Squares function and it's gradient and hessian matrix
    """
    def __init__(self, A, b):
        A_shape = np.shape(A)
        b_shape = np.shape(b)
        print(A_shape, b_shape)
        assert len(b_shape) == 1 and len(A_shape) == 2 and b_shape[0] == A_shape[0] == A_shape[1]
        self.A = A
        self.b = b

    def f(self, *x):
        return sum((np.dot(self.A, x) - self.b)**2)

    def grad(self, *x):
        ax_b = np.dot(self.A, x) - self.b
        return 2 * np.dot(self.A.T, ax_b)

    def hessian(self, *x):
        return 2 * np.dot(self.A.T, self.A)
