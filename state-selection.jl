include("./optimizer.jl")

#sample script for controllable state selection

A = [ 0. 0 1 0 0 ; 1 0 0 0 0 ; 0 0.5 0 0 0.5 ; 0 0 1 0 0 ; 0 0 0 1 0 ]

B = [ 1. 0 0 0 0 ; 0 1 0 0 0 ]'

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

@time B1,nits = pgm(A,B,x0,eta,2.,return_its=true)

@show min_energy(B,A,x0,eta)
@show min_energy(B1,A,x0,eta)

@time B3, nits2 = pgm(A,B,x0,eta,2.,t0=0.,t1=100.,return_its=true)

@time B2, numits = nested_pgm(A,B,x0,eta,2.)   #this takes about an hour eek
# returns B2 =
