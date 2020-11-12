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
B0 = rand(4);
x0 = [ 1.0; 0.5; 0.25; -1 ];

a,v = eigen(flow_matrix(A,a=0,b=15.))
@show testb = [ v[:,4] v[:,4] ]
testb2 = [ v[:,1] v[:,2] ]
testb4 = [ v[:,1] v[:,4] ]
testb5 = [ v[:,3] v[:,4] ]
@time testb3 = pgm2(A,testb,2)
@time B1,nits = pgm(A,B0,x0,eta,1.,return_its=true)

@show min_energy(B0,A,x0,eta)
@show min_energy(B1,A,x0,eta)
@show min_energy(testb,A,x0,eta)
@show min_energy(testb2,A,x0,eta)

obj(x) = gtilde1(A,x,x0)
#@time b1, ob, nits = general_objective_pgm(obj,A,B0,1.;return_its=true)
@show b1 = sphere_projection(b1,1+1e-8)
obj2(x) = gtilde2(A,x,x0)
@time b2, ob, nits = general_objective_pgm(obj2,A,B0,1.;return_its=true)
@show b2 = sphere_projection(b2,1+1e-8)
obj3(x) = gtilde3(A,x,x0)
@time b3, ob, nits = general_objective_pgm(obj3,A,B0,1.;return_its=true)
@show b3 = sphere_projection(b3,1+1e-8)  #gives the same result... coincidence?

C = controllability_matrix(A,b1)
CTC = C' * C
@show eigen(CTC)
W,err = gramian(A,b1)
@show eigen(W)
@show rank(W)


using Plots, PlotThemes

eta = 1.
xfin = optimal_mean_state(testb,A,eta,x0,t1=15.)
M = pinv(gramian(A,testb,0.,15.))
plot(t -> u(t,A,testb,x0,M,tf=15.,xf = xfin)[1],0.,15.)
plot!(t -> u(t,A,testb,x0,M,tf=15.,xf = xfin)[2],0.,15.)

plt = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14)
for j=1:4
	plot!(plt, i -> trajectory(A,testb,i,x0,M,xfi=xfin)[j], 0, 1.,label="")
end
plot!(plt, i -> sum(trajectory(A,testb,i,x0,M,xfi=xfin))/4, 0, 1.,label="",linecolor=:black, linestyle=:dash, linewidth=2.5)
plot!(plt, i -> 1, label = "", linecolor=:black, linewidth=2, linestyle=:solid)
plot!()

xfin2 = optimal_mean_state(testb,A,eta,x0,t1=15.)
plt2 = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14)
for j=1:4
	plot!(plt2, i -> trajectory(A,testb,i,x0,M,t1=15.,xfi=xfin2)[j], 0, 15.,label="")
end
plot!(plt2, i -> sum(trajectory(A,testb,i,x0,M,t1=15.,xfi=xfin2))/4, 0, 15.,label="",linecolor=:black, linestyle=:dashdot, linewidth=4)
plot!(plt2, i -> 1, label = "", linecolor=:black, linestyle=:solid, linewidth=4)
#plot!(plt2, [15,15],[-6,7.5],label = "",linecolor=:black, linestyle = :dash)
savefig(plt2,"controlled_avg.pdf")

plt3 = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14,legend=:bottomleft)
for j=1:4
	plot!(plt3, i -> trajectory(A,zeros(4,2),i,x0,M,xfi=xfin2)[j], 0, 15.,label="")
end
plot!(plt3, i -> sum(trajectory(A,zeros(4,2),i,x0,M,xfi=xfin2))/4, 0, 15.,label="",linecolor=:black, linestyle=:dashdot, linewidth=4)
plot!(plt3, i -> 1, label = "Desired threshold", linecolor=:black, linewidth=4, linestyle=:solid)
plot!(plt3, [0,15],[1,1], label = "Average state", linecolor=:black, linewidth=2, linestyle=:dashdot)
plot!()
savefig(plt3,"autonomous_avg.pdf")

using Random
Random.seed!(123)
randb = sphere_projection(rand(4,2),2)
xfin3 = optimal_mean_state(randb,A,eta,x0,t1=15.)
M2 = pinv(gramian(A,randb,0.,15.))
plt3 = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14,legend=:bottomleft)
for j=1:4
	plot!(plt3, i -> trajectory(A,randb,i,x0,M2,t1=15.,xfi=xfin3)[j], 0, 15.,label="")
end
plot!(plt3, i -> sum(trajectory(A,randb,i,x0,M2,t1=15.,xfi=xfin3))/4, 0, 15.,label="", linewidth=4, linecolor=:black, linestyle=:dashdot)
plot!(plt3, i -> 1, label = "", linecolor=:black, linewidth=4, linestyle=:solid)
#plot!(plt3, [15,15],[-4,7.5],label = "",linecolor=:black, linestyle = :solid)
savefig(plt3,"randb_avg.pdf")

plt4 = plot(t -> u(t,A,randb,x0,M,tf=15.,xf = xfin3)[1],0.,15.,label="RAM 1",xlabel="t",ylabel="Input Signal",legendfontsize=14,tickfontsize=14,guidefontsize=14,linewidth=2,legend=:topleft,linecolor=:black,linestyle=:dash)
plot!(plt4,t -> u(t,A,randb,x0,M,tf=15.,xf = xfin3)[2],0.,15.,label="RAM 2",linestyle=:dashdot,linewidth=2,linecolor=:black)
plot!(plt4,t -> u(t,A,testb,x0,M,tf=15.,xf = xfin2)[1],0.,15.,label="Flux 1 & 2",linestyle=:solid,linewidth=2,linecolor=:black)
savefig(plt4,"inputs_avg.pdf")

ninp_rand(t) = (u(t,A,randb,x0,M,tf=15.,xf = xfin3)[1])^2 + (u(t,A,randb,x0,M,tf=15.,xf = xfin3)[2])^2
plt5 = plot(t -> quadgk(a -> ninp_rand(a),0.,t)[1],0.,15.,label="RAM",linestyle=:dash,legend=:topleft,linecolor=:black,linewidth=2,ylabel="Cumulative Input Energy",xlabel="t",legendfontsize=14,tickfontsize=14,guidefontsize=14)
ninp_opt(t) = (u(t,A,testb,x0,M,tf=15.,xf = xfin2)[1])^2 + (u(t,A,testb,x0,M,tf=15.,xf = xfin2)[2])^2
plot!(plt5,t -> quadgk(a -> ninp_opt(a),0.,t)[1],0.1,15.,label="Flux",linestyle=:solid,linecolor=:black,linewidth=2)
savefig(plt5,"energies_avg.pdf")
