# state-selection

This code finds the optimal state for controllability of a network subject to a [set of] state constraints

state-selection.jl contains a sample script and optimizer.jl contains necessary functions.

Functions:

inverse_gramian(A,B,a=0.,b=1.)
This function computes the inverse of the controllability gramian matrix given:\n
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
The core function of the library. Computes the optimal state for control by projected gradient descent. Inputs:
objective(x) = a function to minimize
projector(x) = a function that projects x onto the constrained subspace
state0 = initial state
max_step_size = first step size to try. will be halved if objective(x1) < objective(x0)
crit = critical threshold for convergence criterion
maxit = maximum number of iterations of projected gradient descent
