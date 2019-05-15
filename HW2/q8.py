import numpy as np

TOLERANCE, MAX_STEP = .001, 1000


def f(x1, x2):
    """

    :param x1:
    :param x2:
    :return: Rosenbrock function
    """
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2


def grad_f(x1, x2):
    """

    :param x1:
    :param x2:
    :return: gradient of Rosenbrock function
    """
    g1 = -400 * (x2 - x1**2) * x1 + 2 * (x1 - 1)
    g2 = 200 * (x2 - x1**2)

    return np.array([g1, g2])


def hessian_inverse_f(x1, x2):
    """

    :param x1:
    :param x2:
    :return: hessian inverse of Rosenbrock function
    """
    h_inverse_b = np.array([
        [200, 400*x1],
        [400*x1, 400*(3*x1**2-x2)+2]
    ])
    return h_inverse_b / (80000 * (3 * x1**2 - x2) + 400 - 160000 * x1**2)


def norm_p(x, p=2):
    assert p >= 1
    return sum(x**p) ** (1/p)


def steepest_descent_direction(x):
    return -1 * grad_f(*x)


def newton_direction(x):
    h_i = hessian_inverse_f(*x)
    g = grad_f(*x)
    return -1 * np.dot(h_i, g)


last_BFGS_H, last_step_length, last_x = [None] * 3
def BFGS_direction(x):
    global last_BFGS_H, last_step_length, last_x

    if last_BFGS_H is None:
        H = hessian_inverse_f(*x)
    else:
        last_p = -1 * np.dot(last_BFGS_H, grad_f(*x))
        x_kPlus1 = last_x
        s = last_step_length * last_p
        x_k = x_kPlus1 - s
        y = grad_f(*x_kPlus1) - grad_f(*x_k)
        m = 1 / np.dot(s, y)
        s = s[:, None]  # shape: (2,) -> (2,1)
        y = y[:, None]
        s_yT = np.dot(s, y.T)
        y_sT = np.dot(y, s.T)
        s_sT = np.dot(s, s.T)
        mL = np.identity(2) - m * s_yT
        mR = np.identity(2) - m * y_sT
        rL = np.dot(np.dot(mL, last_BFGS_H), mR)
        rR = m * s_sT
        H = rL + rR

    last_BFGS_H = H
    return -1 * np.dot(H, grad_f(*x))


def first_wolf_condition(x, p, alpha, c):
    """
    sufficient decrease condition
    """
    new_x = x + alpha * p
    return f(*new_x) <= f(*x) + c * alpha * np.dot(p, grad_f(*x))


def backtracking_line_length(x, p):
    alpha, m, c = 1, .5, .5  # parameters
    while not first_wolf_condition(x, p, alpha, c):
        alpha = alpha * m
    return alpha


def line_search(x, descent_direction_algorithm=steepest_descent_direction):
    global last_BFGS_H, last_x, last_step_length
    last_BFGS_H = None
    step = 0
    while norm_p(grad_f(*x)) >= TOLERANCE and step <= MAX_STEP:
        p = descent_direction_algorithm(x)
        alpha = backtracking_line_length(x, p)
        x = x + alpha * p
        last_x, last_step_length = x, alpha
        print('{}. new x: {} using alpha={}'.format(step, x, alpha))
        step += 1
    print()
    return x


if __name__ == '__main__':
    in_x = [
        np.array([1.2, 1.2]),
        np.array([-1.2, 1])
    ]
    results = [line_search(xi, BFGS_direction) for xi in in_x]
    print('Results :', results)
