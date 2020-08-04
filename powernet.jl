include("./optimizer.jl")

using CSV, Plots, DataFrames, DelimitedFiles

A = Float64.(Matrix(CSV.read("powernet.csv";header=false))) - I

@time Phi = flow_matrix(A)
#takes about an hour... not bad?

@time lambda, v = eigen(Phi)

fluxcentrality = real(v[:,1])

writedlm("fluxcent.csv",fluxcentrality,',')
