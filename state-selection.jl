include("./optimizer.jl")

#sample script for controllable state selection

A = [ 0. 0 1 0 0 ; 1 0 0 0 0 ; 0 0.5 0 0 0.5 ; 0 0 1 0 0 ; 0 0 0 1 0 ]

B = [ 1. 0 0 0 0 ]'

x0 = [ -1. -1 0 1 1 ]'

eta = 0.5

M = inverse_gramian(A,B)

projector(x) = median_projector(x,eta)

xf = exp(A)*x0

W = inv(factorize(M))

gamma = (sum(xf) - length(xf)*eta) / sum(W)

xstar = xf - W * gamma * ones(length(xf),1)

@time state = pgd_optimizer(x -> energy(x,x0,M),projector,xstar)

@show state
