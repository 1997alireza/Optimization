# Optimization Course Assignments

Under supervision of [Dr. Maryam Amirmazlaghani](http://ceit.aut.ac.ir/autcms/people/verticalPagesAjax/professorHomePage.htm?url=mazlaghani&depurl=computer-engineering&lang=en&cid=11875)


### \< Homework one >
* Solve optimization problem using [`cvxpy`](https://www.cvxpy.org/) library
* Minimize [`Quadratic Functions`](https://latex.codecogs.com/svg.latex?\Large&space;\frac{1}{2}x^TPx+q^T+r;%20P%20\in%20S^n%20{\color{Red}%20\textup{%20convex}%20\textup{%20if%20}%20P%20\succeq%200})
<!--: <img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{1}{2}x^TPx+q^T+r;%20P%20\in%20S^n%20{\color{Red}%20\textup{%20convex}%20\textup{%20if%20}%20P%20\succeq%200}" title="\frac{1}{2}x^TPx+q^T+r;%20P%20\in%20S^n%20{\color{Red}%20\textup{%20convex}%20\textup{%20if%20}%20P%20\succeq%200}" />-->

### \< Homework two >
Unconstrained optimization algorithms 
* Line Search methods
  * Find a [`decent direction`](https://en.wikipedia.org/wiki/Descent_direction)
    * Steepest Decent
    * Newton
    * BFGS (a Quasi-Newton method)
  * Find a step length that satisfies [`wolfe conditions`](https://en.wikipedia.org/wiki/Wolfe_conditions)
    * Backtracking Line Search
    
* Trust Region methods
  * Construct a model function as   <code>ƒ<sub>k</sub> + p<sup>T</sup>∇ƒ<sub>k</sub> + 0.5 p<sup>T</sup>B<sub>k</sub>p</code>    by choosing a matrix as B<sub>k</sub>
    * Hessian Matrix
  * Solve the constrained subproblem and find a step
    * Cauchy Point
    * Dogleg
