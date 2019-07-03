import numpy as np
from numpy import matmul as matmul
from numpy.linalg import norm
import time


class LinearConjugateGradient:
    def __init__(self, A, b, x=None):
        A = np.array(A)
        b = np.array(b)
        b = np.reshape(b, (-1, 1))
        assert np.shape(b)[0] == np.shape(A)[0]
        if not x:
            x = np.random.rand(np.shape(A)[1], 1)
        else:
            x = np.reshape(x, (-1, 1))
            assert np.shape(x)[0] == np.shape(A)[1]

        self.A = A
        self.b = b
        self.x = x

    def residual(self):
        return np.matmul(self.A, self.x) - self.b

    _MAX_ITER, _TOLERANCE = 100000, .00000001
    def solve(self):
        r = self.residual()
        p = -r

        A = self.A
        b = self.b
        x = self.x

        k = 0

        time_start_sec = time.time()
        while k <= self._MAX_ITER and norm(r) > self._TOLERANCE:
            pAp = matmul(matmul(p.T, A), p)
            alpha = - matmul(r.T, p) / pAp
            x += alpha * p
            r = self.residual()
            beta = matmul(matmul(r.T, A), p) / pAp
            p = -r + beta * p

            k += 1

        time_end_sec = time.time()

        return x, (time_end_sec - time_start_sec)
