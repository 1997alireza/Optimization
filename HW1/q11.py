import cvxpy as cp
import numpy as np

# Quadratic function: 1/2 x^𝑇 P x + q^𝑇 x + r
P = np.array([
    [13, 12, -2],  # row one
    [12, 17, 6],  # row two
    [-2, 6, 12]  # row three
])
q = np.array([-22, -14.5, 13])
r = 1

x = cp.Variable(3)
objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T * x + r)
constraints = [-1 <= x, x <= 1]
prob = cp.Problem(objective, constraints)
print('Optimal objective value    :', prob.solve())
print('optimal value for variables: x=\n{}'.format(x.value))
