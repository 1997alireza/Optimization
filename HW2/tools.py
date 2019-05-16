import numpy as np


def norm_p(x, p=2):
    assert p >= 1
    return sum(x**p) ** (1/p)


def inverse_matrix_2_2(m):
    a = m[0][0]
    b = m[0][1]
    c = m[1][0]
    d = m[1][1]
    return np.array([[d, -b], [-c, a]]) / (a * d - b * c)


def rosenbrock_function(x1, x2):
    """

    :param x1:
    :param x2:
    :return: Rosenbrock function
    """
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2


def rosenbrock_grad(x1, x2):
    """

    :param x1:
    :param x2:
    :return: gradient of Rosenbrock function
    """
    g1 = -400 * (x2 - x1**2) * x1 + 2 * (x1 - 1)
    g2 = 200 * (x2 - x1**2)

    return np.array([g1, g2])


def rosenbrock_hessian(x1, x2):
    """

    :param x1:
    :param x2:
    :return: hessian of Rosenbrock function
    """
    return np.array([
        [400 * (3 * x1 ** 2 - x2) + 2, -400 * x1],
        [-400 * x1, 200]
    ])


def rosenbrock_inverse_hessian(x1, x2):
    """

    :param x1:
    :param x2:
    :return: inverse hessian of Rosenbrock function
    """
    h = rosenbrock_hessian(x1, x2)
    return inverse_matrix_2_2(h)
