# Optimal control for networked moments

This code is a collection of tools to design control inputs subject to constraints on the moments of the state vector.

`state-selection.jl` contains a sample script and `optimizer.jl` contains necessary functions. Other `.jl` scripts replicate figures from:

>Solimine, P. and Meyer-Baese, A., 2022, December. Input design for the optimal control of networked moments. In 2022 IEEE 61st Conference on Decision and Control (CDC) (pp. 5894-5901). IEEE.


Functions:

inverse_gramian(A,B,a=0.,b=1.)
This function computes the inverse of the controllability gramian matrix given:

A = nxn matrix, describing autonomous behavior of linear dynamical system
B = nxm matrix, describing a control input schematic
and optionally:
a = starting time
b = ending time
Result is a symmetric and positive definite matrix provided the system is controllable under B


energy(x,x0,M,tlim=1.)
This function returns the minimum energy required to steer system to a given state from a given starting point. Inputs are:
x = final/desired state for control
x0 = starting state
M = inverse of controllability gramian (see above)
and optionally:
tlim = time horizon (deadline)

median_projector(x,eta)
A sample projector which projects state x onto the constrained space $C = {x \in R^n | median(x) >= eta }$. Inputs are:
x = state to project
eta = median threshold

gradient(func,x,h=1e-8)
Computes the gradient of a function by fixed differences (in the forward direction). Inputs:
func = function to differentiate
x = values at which to differentiate
h = step size for fixed differences

pgd_optimizer(objective,projector,state0,max_step_size=1e-1,crit=1e-5,maxit=100000)
Computes the optimal state for control by projected gradient descent. Inputs:
objective(x) = a function to minimize
projector(x) = a function that projects x onto the constrained subspace
state0 = initial state
max_step_size = first step size to try. will be halved if objective(x1) < objective(x0)
crit = critical threshold for convergence criterion
maxit = maximum number of iterations of projected gradient descent

pgm_max_sync(A,B0,nD;tol=1e-5,initstep=0.001,t0=0.,t1=1.,return_its=false)
Finds the best (most efficient in terms of input signal norm) input schematic to maximally synchronize the network subject to a constrained number of controllers.
A = System dynamics matrix
B0 = Initial guess of control input schematic
nD = Normalization for B, so that trace(B'B) = nD + ϵ
