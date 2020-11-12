include("./optimizer.jl")

using CSV, Plots, DataFrames, DelimitedFiles

A = Float64.(Matrix(CSV.read("powernet.csv";header=false))) - I

#@time Phi = flow_matrix(A)

#@time lambda, v = eigen(Phi)

#fluxcentrality = real(v[:,1])

#writedlm("fluxcent.csv",fluxcentrality,',')

A = Float64.(Matrix(CSV.read("alzheimers_structural_brain_network.csv";header=false)))

@time Phi = flow_matrix(A)

@time lambda, v = eigen(Phi)

flux = -real(v[:,1])

writedlm("fluxcent_alzheimers.csv",flux,',')

histogram(flux)

A = Float64.(Matrix(CSV.read("karate.csv";header=false)))

@time Phi = flow_matrix(A)

@time lambda, v = eigen(Phi)
flux = zeros(34,3)
flux[:,2] = real(v[:,1])

Phi = flow_matrix(A,b=0.015)
lambda, v = eigen(Phi)
flux[:,1] = -real(v[:,1])

Phi = flow_matrix(A,b=0.15)
lambda, v = eigen(Phi)
flux[:,2] = -real(v[:,1])

Phi = flow_matrix(A,b=1.5)
lambda, v = eigen(Phi)
flux[:,3] = -real(v[:,1])

writedlm("fluxcent_karate3.csv",flux,',')

A = Float64.(Matrix(CSV.read("celegans.csv";header=false)))

@time Phi = flow_matrix(A)

@time lambda, v = eigen(Phi)

flux = real(v[:,1])

writedlm("fluxcent_celegans.csv",flux,',')

histogram(flux)
