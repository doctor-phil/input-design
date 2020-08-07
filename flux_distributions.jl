include("./optimizer.jl")

using CSV, Plots, DataFrames, DelimitedFiles

A = Float64.(Matrix(CSV.read("powernet.csv";header=false))) - I

#@time Phi = flow_matrix(A)
#takes about an hour... not bad?

#@time lambda, v = eigen(Phi)

#fluxcentrality = real(v[:,1])

#writedlm("fluxcent.csv",fluxcentrality,',')

phi = Float64.(Matrix(CSV.read("fluxcent.csv";header=false)))

on = ones(5300)

L = zeros(5300,5300)

L[diagind(L)] = A' *on

L = L - A

lam, v = eigen(A)
