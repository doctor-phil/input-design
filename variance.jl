include("./optimizer.jl")

using Random, Plots

Random.seed!(12345)
init = sphere_projection(rand(4,1).*2 .- 1,1)

A = [-0.5 0 0 0 ; -0.5 0 0 0 ; 0.3 0.5 0 0 ; 0 0 1 0. ];

proj(B) = sphere_projection(B,2)
objective(B) = -lambdamax(A,B,t0=0.,t1=15.)

@time testb,val = general_objective_pgm(objective,A,init,1,initstep=2.)
#ITS THE SAME WTF!!

eta = 4.
M = pinv(gramian(A,testb,0,15))
xfin = optimal_var_state(testb,A,eta,x0,t1=15.)
plt2 = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14)
for j=1:4
	plot!(plt2, i -> trajectory(A,testb,i,x0,M,t1=15.,xfi=xfin)[j], 0, 15.,label="")
end
plot!(plt2, i -> sum((trajectory(A,testb,i,x0,M,t1=15.,xfi=xfin)-xf).^2)/4, 0, 15.,label="",linecolor=:black, linestyle=:dashdot, linewidth=4)
plot!(plt2, i -> eta/4, label = "", linecolor=:black, linestyle=:solid, linewidth=4)
#plot!(plt2, [15,15],[-6,7.5],label = "",linecolor=:black, linestyle = :dash)
savefig(plt2,"controlled_var.pdf")

energy(xfin,x0,M)

eta = 4.
M = pinv(gramian(A,testb,0,15))
xfin = optimal_var_state(testb,A,eta,x0,t1=15.)
plt3 = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14)
for j=1:4
	plot!(plt3, i -> trajectory(A,zeros(4),i,x0,M,t1=15.,xfi=xfin)[j], 0, 15.,label="")
end
plot!(plt3, i -> sum((trajectory(A,zeros(4),i,x0,M,t1=15.,xfi=xfin)-xf).^2)/4, 0, 15.,label="",linecolor=:black, linestyle=:dashdot, linewidth=4)
plot!(plt3, i -> eta/4, label = "", linecolor=:black, linestyle=:solid, linewidth=4)
#plot!(plt2, [15,15],[-6,7.5],label = "",linecolor=:black, linestyle = :dash)
savefig(plt3,"autonomous_var.pdf")

eta = 4.
M2 = pinv(gramian(A,init,0,15))
xfin3 = optimal_var_state(init,A,eta,x0,t1=15.)
plt3 = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14)
for j=1:4
	plot!(plt3, i -> trajectory(A,init,i,x0,M2,t1=15.,xfi=xfin3)[j], 0, 15.,label="")
end
plot!(plt3, i -> sum((trajectory(A,init,i,x0,M2,t1=15.,xfi=xfin3)-xf).^2)/4, 0, 15.,label="",linecolor=:black, linestyle=:dashdot, linewidth=4)
plot!(plt3, i -> eta/4, label = "", linecolor=:black, linestyle=:solid, linewidth=4)
#plot!(plt2, [15,15],[-6,7.5],label = "",linecolor=:black, linestyle = :dash)
savefig(plt3,"randb_var.pdf")

plt4 = plot(t -> u(t,A,init,x0,M2,tf=15.,xf = xfin3)[1],0.,15.,label="RAM",xlabel="t",ylabel="Input Signal",legendfontsize=14,tickfontsize=14,guidefontsize=14,linewidth=2,legend=:topleft,linecolor=:black,linestyle=:dash)
plot!(plt4,t -> u(t,A,testb,x0,M,tf=15.,xf = xfin)[1],0.,15.,label="Flux",linestyle=:solid,linewidth=2,linecolor=:black)
savefig(plt4,"inputs_var.pdf")

ninp_rand(t) = (u(t,A,init,x0,M2,tf=15.,xf = xfin3)[1])^2
plt5 = plot(t -> quadgk(a -> ninp_rand(a),0.,t)[1],0.,15.,label="RAM",linestyle=:dash,legend=:topleft,linecolor=:black,linewidth=2,ylabel="Cumulative Input Energy",xlabel="t",legendfontsize=14,tickfontsize=14,guidefontsize=14)
ninp_opt(t) = (u(t,A,testb,x0,M,tf=15.,xf = xfin)[1])^2
plot!(plt5,t -> quadgk(a -> ninp_opt(a),0.,t)[1],0.1,15.,label="Flux",linestyle=:solid,linecolor=:black,linewidth=2)
savefig(plt5,"energies_var.pdf")