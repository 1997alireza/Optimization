import numpy as np
from numpy import matmul


class FunctionProvider:
    def __init__(self, f, g, h):
        self.f = f
        self.grad = g
        self.hessian = h

    def __add__(self, other):
        f = lambda x: self.f(x) + other.f(x)
        g = lambda x: self.grad(x) + other.grad(x)
        h = lambda x: self.hessian(x) + other.hessian(x)

        return FunctionProvider(f, g, h)

    def __mul__(self, other):
        f = lambda x: self.f(x) * other
        g = lambda x: self.grad(x) * other
        h = lambda x: self.hessian(x) * other

        return FunctionProvider(f, g, h)

    def __rmul__(self, other):
        return self.__mul__(other)


class QuadraticFunctionProvider(FunctionProvider):
    """
    1/2 x^𝑇 A x - b^𝑇 x
    """
    def __init__(self, A, b):
        b = np.reshape(b, [-1, 1])
        assert np.shape(b)[0] == np.shape(A)[0] == np.shape(A)[1]

        f = lambda x: 0.5 * matmul(matmul(x.T, A), x) - matmul(b.T, x)
        g = lambda x: matmul(A, x) - b
        h = lambda x: A

        FunctionProvider.__init__(self, f, g, h)
