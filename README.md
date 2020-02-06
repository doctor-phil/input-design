# state-selection

This code finds the optimal state for controllability of a network subject to a [set of] state constraints

state-selection.jl contains a sample script and optimizer.jl contains necessary functions.

Functions:

inverse_gramian(A,B,a=0.,b=1.)
This function computes the inverse of the controllability gramian matrix given:
A = nxn matrix, describing autonomous behavior of linear dynamical system
B = nxm matrix, describing a control input schematic
Result is symmetric and positive definite if rank([ B AB A^2B ... A^(n-1)B ])=n

