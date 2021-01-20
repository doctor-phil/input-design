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



using CSV
eta = 34
A = -laplacian(Float64.(Matrix(CSV.read("karate.csv";header=false))))
Random.seed!(12345)
randb = rand(length(A[1,:]),5)
x0 = rand(length(A[1,:]))
M = inv(gramian(A,randb,0,3))
xfin_cel = optimal_var_state(randb,A,eta,x0,t0=0,t1=3)
plt4 = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14)
for j=1:length(A[1,:])
	plot!(plt4, i -> trajectory(A,randb,i,x0,M,t1=3,xfi=xfin_cel)[j], 0, 3, label="")
end

plot!()

eta = 34
A = -laplacian(Float64.(Matrix(CSV.read("karate.csv";header=false))))
Random.seed!(12345)
randb = rand(length(A[1,:]),1)
x0 = rand(length(A[1,:]))
M = pinv(gramian(A,randb,0,1.5))
xfin_cel = optimal_var_state(randb,A,eta,x0,t0=0,t1=1.5)
plt5 = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14)
for j=1:length(A[1,:])
	plot!(plt5, i -> trajectory(A,zeros(34),i,x0,M,t1=1.5,xfi=xfin_cel)[j], 0, 1,label="")
end

plot!()

eta = 500
A = -laplacian(Float64.(Matrix(CSV.read("karate.csv";header=false))))
Random.seed!(12345)
testb = max_eigvec(flow_matrix(A,a=0.,b=1))
x0 = rand(length(A[1,:]))
M = pinv(gramian(A,testb,0,1.5))
xfin_cel = optimal_var_state(testb,A,eta,x0,t0=0,t1=1.5)
plt3 = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14)
for j=1:length(A[1,:])
	plot!(plt3, i -> trajectory(A,testb,i,x0,M,t1=1.5,xfi=xfin_cel)[j], 0, 1.5,label="")
end

plot!()


#for real this time
Random.seed!(1234)
init = sphere_projection(rand(4,1).*2 .- 1,1)

A = [-0.5 0 0 0 ; -0.5 0 0 0 ; 0.3 0.5 0 0 ; 0 0 1 0. ];
x0 = rand(length(A[1,:]))
xf = exp(A)*x0
eta = 5
M1 = pinv(gramian(A,init,0,1))
l,xfin_cel = var_solver(M1,xf,eta)
plt3 = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14)
for j=1:length(A[1,:])
	plot!(plt3, i -> trajectory(A,init,i,x0,M1,t1=1,xfi=xfin_cel)[j], 0, 1,label="")
end
plot!()

Random.seed!(1234)
init = sphere_projection(rand(4,1).*2 .- 1,1)
x0 = rand(length(A[1,:]))
xf = exp(A)*x0
M1 = pinv(gramian(A,init,0,1))
l,xfin_cel1 = var_solver(M1,xf,eta)
A = [-0.5 0 0 0 ; -0.5 0 0 0 ; 0.3 0.5 0 0 ; 0 0 1 0. ];
eta = 5
testb4, obj = general_objective_pgm(x -> var_energy_vec(x,A,x0,eta),A,init,1,verbose=true)
M2 = pinv(gramian(A,testb4,0,1))
l,xfin_cel2 = var_solver(M2,xf,eta)
plt3 = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14)
for j=1:length(A[1,:])
	plot!(plt3, i -> trajectory(A,testb4,i,x0,M2,t1=1,xfi=xfin_cel2)[j], 0, 1,label="")
end

plot!()

plt4 = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14,yaxis=[-5,5])
plot!(plt4, i -> u(i,A,init,x0,M1,xf=xfin_cel1)[1],0,1,label="RAM",linestyle=:dash,linecolor=:black,linewidth=2)
plot!(plt4, i -> u(i,A,testb4,x0,M2,xf=xfin_cel2)[1],0,1,label="NPGM",linestyle=:solid,linecolor=:black,linewidth=2)
savefig(plt4,"variance_energies.pdf")

ninp_rand(t) = (u(t,A,init,x0,M1,tf=1.,xf = xfin_cel1)[1])^2
plt5 = plot(t -> quadgk(a -> ninp_rand(a),0.,t)[1],0.,1,label="RAM",linestyle=:dash,legend=:topleft,linecolor=:black,linewidth=2,ylabel="Cumulative Input Energy",xlabel="t",legendfontsize=14,tickfontsize=14,guidefontsize=14)
ninp_opt(t) = (u(t,A,testb4,x0,M2,tf=1.,xf = xfin_cel2)[1])^2
plot!(plt5,t -> quadgk(a -> ninp_opt(a),0.,t)[1],0,1,label="NPGM",linestyle=:solid,linecolor=:black,linewidth=2)
savefig(plt5,"cumulative_var_energy.pdf")

Random.seed!(1)
init2 = sphere_projection(rand(4,1),1)
testb, obj = general_objective_pgm(x -> var_energy_vec(x,A,x0,eta),A,init2,1,verbose=true)
M3 = pinv(gramian(A,testb,0,1))
l,xfin_cel2 = var_solver(M2,xf,eta)

plt4 = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14)
plot!(plt4, i -> u(i,A,init,x0,M1,xf=xfin_cel)[1],0,1)
plot!(plt4, i -> u(i,A,testb,x0,M,xf=xfin_cel2)[1],0,1)

#variance control for zachary's karate club, laplacian dynamics (DeGroot)
eta = 1
A = -laplacian(Float64.(Matrix(CSV.read("karate.csv";header=false))))
Random.seed!(12345)
randb = rand(length(A[1,:]),10)*2 .- 1
x0 = rand(length(A[1,:]))
M = inv(gramian(A,randb,0,3))
xf = exp(3*A)*x0
l,xfin_cel = var_solver(M,xf,eta)
plt4 = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14)
for j=1:length(A[1,:])	#plot the autonomous dynamics
	@time plot!(plt4, i -> trajectory(A,zeros(34,10),i,x0,M,t1=3,xfi=xfin_cel)[j], 0, 3, label="")
end

plot!()
savefig(plt4,"karate_autonomous.pdf")

eta = 1
A = -laplacian(Float64.(Matrix(CSV.read("karate.csv";header=false))))
Random.seed!(12345)
randb = sphere_projection(rand(length(A[1,:]),10)*2 .- 1,10.0000001)
x0 = rand(length(A[1,:]))
M = inv(gramian(A,randb,0,3))
xf = exp(3*A)*x0
l,xfin_cel = var_solver(M,xf,eta)
plt5 = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14)
for j=1:length(A[1,:])	#plot RAM dynamics
	@time plot!(plt5, i -> trajectory(A,randb,i,x0,M,t1=3,xfi=xfin_cel)[j], 0, 3, label="")
end

plot!()
savefig(plt5,"karate_ram_variance.pdf")

eta = 1
A = -laplacian(Float64.(Matrix(CSV.read("karate.csv";header=false))))
Random.seed!(12345)
randb = sphere_projection(rand(length(A[1,:]),10)*2 .- 1,10.0000001)
x0 = rand(length(A[1,:]))
xf = exp(3*A)*x0
testb4, obj = general_objective_pgm(x -> var_energy_vec(x,A,x0,eta),A,reshape(randb,340,1),1,verbose=true)
testb = reshape(testb4,34,10)
M = inv(gramian(A,testb,0,3))
l,xfin_cel = var_solver(M,xf,eta)
plt5 = plot(xlabel="t", ylabel="State",legendfontsize=14,tickfontsize=14,guidefontsize=14)
for j=1:length(A[1,:])	#plot RAM dynamics
	@time plot!(plt5, i -> trajectory(A,testb,i,x0,M,t1=3,xfi=xfin_cel)[j], 0, 3, label="")
end

plot!()
savefig(plt5,"karate_pgm_variance.pdf")


M1 = inv(gramian(A,randb,0,3))
xf = exp(3*A)*x0
l,xfin_cel1 = var_solver(M,xf,eta)

M2 = inv(gramian(A,testb,0,3))
xf = exp(3*A)*x0
l,xfin_cel2 = var_solver(M2,xf,eta)	#they're the same wtf

ninp_rand(t) = norm(u(t,A,randb,x0,M1,tf=3.,xf = xfin_cel1)[:])^2
plt5 = plot(t -> quadgk(a -> ninp_rand(a),0.,t)[1],0.,3,label="RAM",linestyle=:dash,legend=:topleft,linecolor=:black,linewidth=2,ylabel="Cumulative Input Energy",xlabel="t",legendfontsize=14,tickfontsize=14,guidefontsize=14)
ninp_opt(t) = norm(u(t,A,testb,x0,M2,tf=3.,xf = xfin_cel2)[:])^2
plot!(plt5,t -> quadgk(a -> ninp_opt(a),0.,t)[1],0,3,label="NPGM",linestyle=:solid,linecolor=:black,linewidth=2)
savefig(plt5,"cumulative_var_energy.pdf")
