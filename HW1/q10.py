import cvxpy as cp

x1 = cp.Variable(1)
x2 = cp.Variable(1)

constraints = [2*x1+x2 >= 1, x1+3*x2 >= 1, x1 >= 0, x2 >= 0]

objectives = [
    cp.Minimize(x1 + x2),
    cp.Minimize(-x1 - x2),
    cp.Minimize(x1),
    cp.Minimize(cp.max_elemwise(x1, x2)),
    cp.Minimize(cp.power(x1, 2) + 9*cp.power(x2, 2))
]

for o_id, objective in enumerate(objectives):
    prob = cp.Problem(objective, constraints)
    print('< Objective', o_id, '>')
    print('Optimal objective value    :', prob.solve())
    print('optimal value for variables: x1={}, x2={}'.format(x1.value, x2.value))
    print('-------------------------------------------------------------------------\n')
