using ForwardDiff, Statistics, QuadGK, LinearAlgebra, Printf

function inverse_gramian(A,B,a=0.,b=1.)
	W,err = quadgk(x -> exp(A*x)*(B*(B'))*(exp(A*x)'),a,b)		#compute reachability gramian
	inv(factorize(W))
end

function gramian(A,B,a=0.,b=1.)
	W,err = quadgk(x -> exp(A*x)*(B*(B'))*(exp(A*x)'),a,b)	#invert W
end

function energy(x,x0,M,tlim=1.)			#objective function for minimization
	fin = exp(A*tlim)*x0
	((fin - x)' * M * (fin - x))[1]
end

function median_projector(x,eta)		#projection onto C = {x in R^n | median(x) >= eta}
	med = median(x)
	xnew = copy(x)
	n = size(x,1)
	if med < eta				#otherwise no need to project
		medians = 0
		xnew = copy(x)
		indices = Int64[]		#stores the indices of elements which are equal to median
		for i in 1:n			#count number of elts equal to median
			if x[i] == copy(med)
				medians+=1
				append!(indices,i)
			elseif x[i] > med && x[i] < eta
				xnew[i] = copy(eta)
			end
		end

		j = 1
		while median(xnew) < eta && j <= n
			xnew[indices[j]] = copy(eta)
			j+=1
		end
	end
	return(xnew)
end

function gradient(func,x,h=1e-8)		# fixed differences
	it = 1
	n = size(x,1)
	grad = Float64[]
	while it <= n
		xh = copy(x)
		xh[it] = copy(x[it]) + h
		append!(grad,(func(xh) - func(x)))
		it += 1
	end
	return grad .* (1/h)
end

function pgd_optimizer(objective, projector, state0; max_step_size = 1e-5, crit = 1e-5, maxit = 100000)
	diff = crit + 1.			#performs pgd optimization
	it = 0
	state1 = copy(state0)
	while diff > crit && it < maxit
		state0 = copy(state1)
		grad = ForwardDiff.gradient(objective,state1)
		step_size = copy(max_step_size)
		step = step_size*grad
		state1 = projector(state0 - step)
		halvings = 0
		while objective(state1) - objective(state0) > 0. && halvings < 100000
			step_size /= 2
			step = step_size*grad
			state1 = projector(state0 - step)
			halvings += 1
		end
		diff = norm(state1 - state0)
		it +=1
	end
	if it == maxit
		@printf "Maximum iterations reached"
	end
	gradient = ForwardDiff.gradient(objective,state1)
	return(state1)
end


function min_energy(B,A,x0,eta;t0=0.,t1=1.)
	n = Int(sqrt(length(A)))
	if length(B) > n
		B0 = reshape(B,n,Int(length(B)/n))
	else
		B0 = copy(B)
	end
	W,err = gramian(A,B0)
	me = (sum(exp(A*(t1-t0))*x0) - n*eta)^2 / sum(W)
	return me
end

function derivative_N(B,ND)
	dndb = 4*(tr(B' * B) - ND).*B
	return dndb
end

function tangent_projection(Matrix,B,ND)
	v = reshape(Matrix,length(Matrix),1)
	S = reshape(derivative_N(B,ND),length(Matrix),1)
	S2 = reshape(pinv(S),1,length(Matrix))
	P = (I - S*S2) * v
	return P
end

function sphere_projection(B,Mpe)
	G = sqrt(Mpe / tr(B'*B)) *B
	return G
end

function pgm(A,B0,x0,eta,nD;tol=1e-20,initstep=0.01,t0=0.,t1=1.,return_its=false)
	M = nD + 1e-10
	B1 = sphere_projection(B0,M)
	n = Int(sqrt(length(A)))
	m = Int(length(B0)/n)
	objective(x) = min_energy(x,A,x0,eta,t0=t0,t1=t1)
	costheta = 0.
	numits = 0
	while 1-costheta > tol
		step = copy(initstep)
		B0 = copy(B1)
		B0V = reshape(B0,length(B0),1)
		grad = ForwardDiff.gradient(objective,B0V)
		proj_grad = tangent_projection(grad,B0,M)
		inter = B0V - step*proj_grad
		B1 = sphere_projection(reshape(inter,n,m),M)
		B1V = reshape(B1,length(B1),1)
		costheta = dot(B1V,B0V) / (norm(B1V)*norm(B0V))
		numits+=1
	end
	if return_its
		return B1,numits
	else
		return B1
	end
end

function nested_pgm(A,B0,x0,eta,nD;tol=1e-5,initstep=0.01)
	M = nD + 1e-10
	B1 = sphere_projection(B0,M)
	n = Int(sqrt(length(A)))
	m = Int(length(B)/n)
	objective(x) = energy(pgd_optimizer(y -> energy(y,x0,inverse_gramian(A,reshape(x,n,m))),projector,xstar),x0,inverse_gramian(A,reshape(x,n,m)))
	costheta = 0.
	numits = 0
	while 1-costheta > tol
		step = copy(initstep)
		B0 = copy(B1)
		B0V = reshape(B0,length(B0),1)
		grad = ForwardDiff.gradient(objective,B0V)
		proj_grad = tangent_projection(grad,B0,M)
		inter = B0V - step*proj_grad
		B1 = sphere_projection(reshape(inter,n,m),M)
		B1V = reshape(B1,length(B1),1)
		costheta = dot(B1V,B0V) / (norm(B1V)*norm(B0V))
		numits+=1
	end
	return B1,numits
end

function controllability_matrix(A,B)
	n = length(A[:,1])
	C = copy(B)
	for i = 1:n-1
		C = [ C (A^i)*B ]
	end
	return C
end

function project(a,b)	#project vec a onto vec b
	v = (dot(a,b)/dot(b,b))*b
	return v
end

function orthonormalize(C) #gram schmidt procedure
	C = Matrix(C)
	n = length(C[1,:])
	m = length(C[:,1])
	r = fakerank(C)
	D = copy(C[:,1])
	E = copy(D) ./ norm(D)
	i=1
	j=1
	while fakerank(D) < r
		i+=1
		if fakerank([ D C[:,i] ]) > fakerank(D)
			d = copy(C[:,i])
			for k=1:j
				d -= project(C[:,i],D[:,k])
			end
			D = [ D d ]
			E = [ E (d./norm(d))]
			j+=1
		end
	end
	return E
end

function fakerank(A)	#because julia can't take the rank of a vector annoying
	if length(A[1,:]) > 1
		r = rank(A)
	else
		r = 1
	end
	return r
end

function proj_into_space(v,C) #project v into span(C)
	D = orthonormalize(C)
	m = length(D[1,:])
	u = Float64.(zeros(length(v)))
	for i = 1:m
		u += project(v,D[:,i])
	end
	return u
end

function delta(A,B,x0)
	del = exp(A)*x0
	del .-= proj_into_space(exp(A)*x0,controllability_matrix(A,B))
	return del
end

function num_reachable(A,BV,x0)
	n = length(A[1,:])
	m = Int(length(BV)/n)
	del = delta(A,reshape(BV,n,m),x0)
	count = 0
	for i = 1:length(x0)
		if del[i] == 0
			count += 1
		end
	end
	return count
end

function general_objective_pgm(obj,A,B0,x0,nD;tol=1e-20,initstep=0.01,t0=0.,t1=1.,return_its=false)
	objective(x) = obj(x)
	M = nD + 1e-10
	B1 = sphere_projection(B0,M)
	B1V = reshape(B1,length(B0),1)
	n = Int(sqrt(length(A)))
	m = Int(length(B0)/n)
	costheta = 0.
	numits = 0
	while 1-costheta > tol
		step = copy(initstep)
		B0 = copy(B1)
		B0V = reshape(B0,length(B0),1)
		grad = gradient(objective,B0V)		#fixed differences
		proj_grad = tangent_projection(grad,B0,M)
		inter = B0V .- step*proj_grad
		B1 = sphere_projection(reshape(inter,n,m),M)
		B1V = reshape(B1,length(B1),1)
		costheta = dot(B1V,B0V) / (norm(B1V)*norm(B0V))
		numits+=1
	end
	if return_its
		return round.(B1,digits=8),objective(round.(B1V,digits=8)),numits
	else
		return round.(B1,digits=8),objective(round.(B1V,digits=8))
	end
end


#TEST SCRIPT

function u(t,A,B,x0;M=inverse_gramian(A,B),xf=proj_into_space(exp(A)*x0,controllability_matrix(A,B)))
   u = -B' * exp(A*(1. - t))' * M*xf
   return u
end

A = [1 0 0 0 ; 1 0 0 0 ; 1 1 0 0 ; 0 0 1 0. ];
B0 = [ 1; 0; 0; 0. ];
x0 = [ 1; 0.5; 0; 0 ];
obj(x) = -num_reachable(A,x,x0)
@time b1, ob, nits = general_objective_pgm(obj,A,B0,x0,1.,return_its=true)
@show b1 = sphere_projection(b1,1+1e-8)

using Plots

plot(t -> u(t,A,b1,x0)[1],0.,1.)
