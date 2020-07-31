include("./optimizer.jl")

#sample script for controllable state selection

A = [ 0. 0 1 0 0 ; 1 0 0 0 0 ; 0 0.5 0 0 0.5 ; 0 0 1 0 0 ; 0 0 0 1 0 ]
#A = [ 0. 0 1 0 0 ; 1 0 0 0 0 ; 0 1 0 0 1 ; 0 0 1 0 0 ; 0 0 0 1 0 ]
#A = [ 0. 0 0.343 0 0 ; 0.685 0 0 0 0 ; 0 0.235 0 0 0.789 ; 0 0 0.239 0 0 ; 0 0 0 0.325 0 ]

B = [ 1. 0 0 0 0 ; 0 1 0 0 0 ]'

x0 = [ -1. -1 0 1 1 ]'

eta = 0.5

a,v = eigen(flow_matrix(A))
@show testb = [ v[:,4] v[:,5] ]
testb2 = [ v[:,1] v[:,1] ]
@time testb3 = pgm2(A,testb,2)

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
@show min_energy(testb2,A,x0,eta)

@time B3, nits2 = pgm(A,B,x0,eta,2.,t0=0.,t1=100.,return_its=true)

#@time B2, numits = nested_pgm(A,B,x0,eta,2.)   #this takes about an hour eek
# returns B2 =

A = [-0.5 0 0 0 ; -0.5 0 0 0 ; 0.3 0.5 0 0 ; 0 0 1 0. ];
B0 = [ 1. ; 0.02 ; .5 ; 0.7];
x0 = [ 1.0; 0.5; 0; 0 ];

a,v = eigen(flow_matrix(A))
@show testb = [ v[:,1] v[:,1] ]
testb2 = [ v[:,1] v[:,2] ]
testb4 = [ v[:,1] v[:,4] ]
testb5 = [ v[:,3] v[:,4] ]
@time testb3 = pgm2(A,testb,2)
@time B1,nits = pgm(A,B0,x0,eta,2.,return_its=true)

@show min_energy(B0,A,x0,eta)
@show min_energy(B1,A,x0,eta)
@show min_energy(testb,A,x0,eta)
@show min_energy(testb2,A,x0,eta)

obj(x) = gtilde1(A,x,x0)
@time b1, ob, nits = general_objective_pgm(obj,A,B0,1.;return_its=true)
@show b1 = sphere_projection(b1,1+1e-8)
obj2(x) = gtilde2(A,x,x0)
@time b2, ob, nits = general_objective_pgm(obj2,A,B0,1.;return_its=true)
@show b2 = sphere_projection(b2,1+1e-8)
obj3(x) = gtilde3(A,x,x0)
@time b3, ob, nits = general_objective_pgm(obj3,A,B0,1.;return_its=true)
@show b3 = sphere_projection(b3,1+1e-8)  #gives the same result... coincidence?
xfin = optimal_mean_state(testb,A,eta,x0)
using Plots

plot(t -> u(t,A,testb,x0,xf = xfin)[1],0.,1.)
plot!(t -> u(t,A,testb,x0,xf = xfin)[2],0.,1.)

C = controllability_matrix(A,b1)
CTC = C' * C
@show eigen(CTC)
W,err = gramian(A,b1)
@show eigen(W)
@show rank(W)

plt = plot()
for j=1:4
	plot!(plt, i -> trajectory(A,zeros(4,2),i,x0,xfi=xfin)[j], 0, 1.)
end
plot!(plt, i -> sum(trajectory(A,testb,i,x0,xfi=xfin))/4, 0, 1.)
plot!()

@time b2 = pgm_max_sync(A,b1,1.)

@show b2

@show energy(xf,x0,pinv_gramian(A,b1))
@show energy(xf,x0,pinv_gramian(A,b2))

plot(t -> u(t,A,b1,x0)[1],0.,1.)
plot!(t -> u(t,A,b2,x0)[1],0.,1.)

plot(t -> quadgk(x -> norm(u(x,A,b1,x0)),0,t)[1],0.,1.)
plot!(t -> quadgk(x -> norm(u(x,A,b2,x0)),0,t)[1],0.,1.)

plot(t -> norm_input(u,xf,x0,t,A,b1),0.,1.)
plot!(t -> norm_input(u,xf,x0,t,A,b2),0.,1.)

plot()
for i=1:4
	plot!(t-> trajectory(A,b2,t,x0)[i],0,1)
end
plot!()
