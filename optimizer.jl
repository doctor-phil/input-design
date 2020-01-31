using ForwardDiff, DiffResults, Statistics, QuadGK, LinearAlgebra

tstar = 1.

x0 = [ -1. -1 0 1 1 ]'

A = [ 0 0 1 0 0 ; 1 0 0 0 0 ; 0 1 0 0 1 ; 0 0 1 0 0 ; 0. 0 0 1 0 ]

B = [ 1. 0 0 0 0 ]'

BBT = B * (B')

fin = exp(A*tstar)*x0

function gram(t)
	At = exp(A*t)
	g = At*BBT*(At')
	return(g)
end

W,err = quadgk(x -> gram(x), 0, tstar)		#compute reachability gramian

M = inv(W)					#invert W

function energy(x)				#objective function for minimization
	o = (fin - x)' * M * (fin - x)
	return(o)
end

function project(x,eta)				#projection onto C = {x in [0,1]^n | median(x) >= eta}
	med = median(x)
	if med >= eta
		xnew = x			#no need to project
	else
		medians = 0
		xnew = x
		indices = Int64[]
		for i in 1:size(x,1)		#count number of elts equal to median
			if x[i] == med
				medians++
				append!(indices,i)
			elseif x[i] > med && x[i] < eta
				xnew[i] = eta
			end
		end
		
		j = 1
		while median(xnew) < eta
			xnew[indices[j]] = eta
			j++
		end
	end
	return(xnew)
end

@show project(x0,0.5)
