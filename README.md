# Optimization Course Assignments

Under supervision of [Dr. Maryam Amirmazlaghani](http://ceit.aut.ac.ir/autcms/people/verticalPagesAjax/professorHomePage.htm?url=mazlaghani&depurl=computer-engineering&lang=en&cid=11875)


### \< Homework one >
* Solve optimization problem using [`cvxpy`](https://www.cvxpy.org/) library
* Minimize [`Quadratic Functions`](https://latex.codecogs.com/svg.latex?\Large&space;\frac{1}{2}x^TPx+q^T+r;%20P%20\in%20S^n%20{\color{Red}%20\textup{%20convex}%20\textup{%20if%20}%20P%20\succeq%200})
<!--: <img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{1}{2}x^TPx+q^T+r;%20P%20\in%20S^n%20{\color{Red}%20\textup{%20convex}%20\textup{%20if%20}%20P%20\succeq%200}" title="\frac{1}{2}x^TPx+q^T+r;%20P%20\in%20S^n%20{\color{Red}%20\textup{%20convex}%20\textup{%20if%20}%20P%20\succeq%200}" />-->

### \< Homework two >
**Unconstrained optimization** algorithms 
* [Line Search methods](https://people.maths.ox.ac.uk/hauser/hauser_lecture2.pdf)
  * Find a [`decent direction`](https://en.wikipedia.org/wiki/Descent_direction)
    * Steepest Decent
    * Newton
    * BFGS (a Quasi-Newton method)
  * Find a step length that satisfies [`wolfe conditions`](https://en.wikipedia.org/wiki/Wolfe_conditions)
    * Backtracking Line Search

* [Trust Region methods](https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods)
  * Construct a model function as   <code>ƒ<sub>k</sub> + p<sup>T</sup>∇ƒ<sub>k</sub> + 0.5 p<sup>T</sup>B<sub>k</sub>p</code>    by choosing a matrix as B<sub>k</sub>
    * Hessian Matrix
  * Solve the constrained subproblem and find a step
    * Cauchy Point
    * Dogleg
    
### \< Homework three >
**Unconstrained optimization** algorithms
<br/><i>Solving large linear systems of equations: <code>Ax = b</code> or <code><b>min<sub>x</sub></b> 0.5 x<sup>T</sup>Ax - b<sup>T</sup>x</code></i>
* [Linear Conjugate Gradient](https://en.wikipedia.org/wiki/Conjugate_gradient_method)

**Constrained optimization** algorithms
<br/><i>Solving inequality constrained minimization: <code><b>min<sub>x</sub></b> 0.5 x<sup>T</sup>Ax - b<sup>T</sup>x <b>s.t.</b> Px ≼ q</code></i>
* [Log Barrier](http://www.stat.cmu.edu/~ryantibs/convexopt-F15/lectures/15-barr-method.pdf)
  * Solving an equality constrained minimization using [Newton method](http://www.math.uwaterloo.ca/~hwolkowi//henry/reports/talks.d/t06talks.d/06msribirs.d/summercoursemsri07.d/equalconstrminbookpgs.pdf) on each iteration
* [Primal Dual Interior Point](http://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/primal-dual.pdf)
