using ForwardDiff, Statistics, QuadGK, LinearAlgebra, Printf

function inverse_gramian(A,B,a=0.,b=1.)
	W,err = quadgk(x -> exp(A*x)*(B*(B'))*(exp(A*x)'),a,b)		#compute reachability gramian
	inv(factorize(W))
end

function gramian(A,B,a=0.,b=1.)
	W,err = quadgk(x -> exp(A*x)*(B*(B'))*(exp(A*x)'),a,b)
	return W
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

function gradient(func,x;h=1e-12) #cfd
    e = zeros(length(x))
    grad = zeros(length(x))
    for i=1:length(x)
        e[i] = h
        grad[i] = (func(x .+ e) - func(x .- e)) / (2*h)
        e[i] = 0.
    end
    return grad
end

function pgd_optimizer(objective, projector, state0; max_step_size = 1e-5, crit = 1e-5, maxit = 100000)
	diff = crit + 1.			#performs pgd optimization (minimization)
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
	W = gramian(A,B0)
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

function orthonormalize(C) #gram schmidt
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
	return round.(u,digits=5)
end

function delta(A,B,x0)
	del = exp(A)*x0
	del .-= proj_into_space(exp(A)*x0,controllability_matrix(A,B))
	return round.(del,digits=5)
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

function control_pinv(x0,C)
	return pinv(C) * x0
end

function general_objective_pgm(obj,A,B0,nD;tol=1e-20,initstep=0.01,t0=0.,t1=1.,verbose=false)
	M = nD + 1e-10
	B1 = sphere_projection(B0,M)
	B1V = reshape(B1,length(B0),1)
	n = Int(sqrt(length(A)))
	m = Int(length(B0)/n)
	costheta = 0.
	numits = 0
	step = copy(initstep)
	cuttings = 0
	while 1-costheta > tol && cuttings < 100
		step = copy(initstep)
		cuttings = 0
		B0 = copy(B1)
		B0V = reshape(B0,length(B0),1)
		grad = gradient(obj,B0V)		#fixed differences
		proj_grad = tangent_projection(grad,B0,M)
		inter = B0V .- step*proj_grad
		B1 = sphere_projection(reshape(inter,n,m),M)
		B1V = reshape(B1,length(B1),1)

		while obj(B1V) > obj(B0V) || obj(B1V) < 0
			step /= 1.68
			cuttings += 1
			inter = B0V .- step*proj_grad
			B1 = sphere_projection(reshape(inter,n,m),M)
			B1V = reshape(B1,length(B1),1)
		end
		if verbose
			@show obj(B1V)
		end
		costheta = dot(B1V,B0V) / (norm(B1V)*norm(B0V))
		numits+=1
	end
	if verbose
		return B1,obj(round.(B1V,digits=8)),numits,cuttings,step
	else
		return B1,obj(round.(B1V,digits=8))
	end
end

function u(t,A,B,x0,M;tf=1.,xf=Float64.(zeros(length(x0))))
	if t <= tf
		u = B' * exp(A*(tf - t))' * M*(xf .- exp(A*tf)*x0)
	else
		u = zeros(length(B[1,:]),1)
	end
	return u
end

function trajectory(A,B,t,x0,M;xfi=Float64.(zeros(length(x0))),t1= 1.)
	x,err = quadgk(z -> (exp((t-z)*A)*B*u(z,A,B,x0,M,tf=t1,xf=xfi)),0,t,rtol=1e-7)
	return exp(t*A)*x0 .+ x
end

function norm_input(u,xf,x0,tf,A,B)
	q,err = quadgk(t -> norm(u(t,A,B,x0,tf=tf,xf=xf)),0.,tf)
	return q
end

function pinv_gramian(A,B;a=0.,b=1.)
	W,err = quadgk(x -> exp(A*x)*(B*(B'))*(exp(A*x)'),a,b)		#compute reachability gramian
	return pinv(W)
end

function grad_EB(A,B,x0;a=0.,b=1.)
	WBinv = pinv_gramian(A,B;a=a,b=b)
	Xf = exp(A*(b-a)) * x0 * x0' * exp(A*(b-a))'
	CTC(t) = exp(A*t)' * WBinv * Xf * WBinv * exp(A*t)
	q,err = quadgk(t -> CTC(t),a,b)
	return q*B
end

function pgm_max_sync(A,B0,nD;tol=1e-5,initstep=0.001,t0=0.,t1=1.,return_its=false)
	M = nD + 1e-10
	B1 = sphere_projection(B0,M)
	n = Int(sqrt(length(A)))
	m = Int(length(B0)/n)
	costheta = 0.
	numits = 0
	while 1-costheta > tol
		step = copy(initstep)
		B0 = copy(B1)
		B0V = reshape(B0,length(B0),1)
		grad = reshape(grad_EB(A,B0,x0),length(B0),1)
		proj_grad = tangent_projection(grad,B0,M)
		inter = B0V - step*proj_grad
		B1 = sphere_projection(reshape(inter,n,m),M)
		B1V = reshape(B1,length(B1),1)
		if num_reachable(A,B1V,x0) < num_reachable(A,B0V,x0)
			B1 = copy(B0)
			B1V = copy(B0V)
		end
		costheta = dot(B1V,B0V) / (norm(B1V)*norm(B0V))
		numits+=1
	end
	if return_its
		return B1,numits
	else
		return B1
	end
end

function gtilde1(A,B,x0)
	n = length(x0)
	B = reshape(B,n,Int(length(B)/n))
	gt = 0.
	pj = control_pinv(x0,controllability_matrix(A,B))
	for i=1:n
		if x0[i] != 0
			gt += ((x0[i] - pj[i]) / x0[i])^2
		end
	end
	return gt
end

function gtilde2(A,B,x0)
	n = length(x0)
	B = reshape(B,n,Int(length(B)/n))
	gt = 0.
	pj = proj_into_space(x0,controllability_matrix(A,B))
	for i=1:n
		if x0[i] != 0
			gt += ((x0[i] - pj[i]) / x0[i])^2
		end
	end
	return gt
end

function gtilde3(A,B,x0)
	n = length(x0)
	B = reshape(B,n,Int(length(B)/n))
	gt = 0.
	pj = proj_into_space(x0,controllability_matrix(A,B))
	for i=1:n
		gt += ((x0[i] - pj[i])^2)
	end
	return gt
end

function flow_matrix(A;a=0.,b=1.,tolerance=1e-5)
	n = length(A[1,:])
	on = ones(n,1)
	M,err = quadgk(x -> (exp(A*x)' *(on*on')*(exp(A*x))),a,b,rtol=tolerance)
	return M
end

function gram_sum(A,B)
	W = gramian(A,B)
	return sum(W)
end

function gram_sum_vec(A,B)
	n = Int(sqrt(length(A)))
	if length(B) > n
		m = Int(length(B)/n)
		B0 = reshape(B,n,m)
	else
		m = 1
		B0 = copy(B)
	end
	W = gramian(A,B0)
	s = sum(W)
	return s
end

function flow_ev(A;a=0.,b=1.)
	n = length(A[1,:])
	on = ones(n,1)
	a,err = quadgk(x -> (exp(A*x)' *on),a,b)
	return a
end

function pgm2(A,B0,nD;tol=1e-20,initstep=0.01,t0=0.,t1=1.,return_its=false)
	M = nD + 1e-10
	B1 = sphere_projection(B0,M)
	n = Int(sqrt(length(A)))
	m = Int(length(B0)/n)
	objective(x) = -gram_sum_vec(A,x)
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

function optimal_mean_state(B,A,eta,x0;t0=0.,t1=1.)
	v = exp(A*(t1-t0))*x0
	n = length(x0)
	on = ones(n,1)
	denom = gram_sum(A,B)
	alpha = sum(v) - n*eta
	xa = v .- (alpha/denom)*gramian(A,B)*on
	return xa
end

function opt_sort(x,W)
	n = length(x)
	u,v = eigen(W)
	lambda,ind = findmax(u)
	vmax = v[:,ind]
	oldi = Int32.(zeros(n))
	newi = Int32.(zeros(n))
	xnew = copy(x)

	for i=1:n
		for j=1:n
			if x[j] <= x[i]
				newi[i]+=1
			end
		end
	end

	for i = 1:n
		for j = 1:n
			if i!=j
				if newi[i] == newi[j]
					if vmax[i] > vmax[j]
						newi[j]-=1
					elseif vmax[i] == vmax[j]
						if i > j
							newi[j]-=1
						else
							newi[i]-=1
						end
					else
						newi[i]-=1
					end
				end
			end
		end
	end

	for i=1:n
		xnew[newi[i]] = x[i]
		oldi[newi[i]] = i
	end

	return xnew, oldi
end

function proj_median_state(x,W,eta;t0=0.,t1=1.)
	n = length(x)
	sorted, oldi = opt_sort(x,W)
	term = 0
	xnew = copy(x)
	i = Int32(floor(n/2))

	while term==0
		i+=1
		if i <= n && sorted[i] < eta
			xnew[oldi[i]] = eta
		else
			term = 1
		end
	end

	return xnew
end

function med_nested_pgm(A,B0,x0,eta,nD;tol=1e-5,initstep=0.01)
	M = nD + 1e-10
	B1 = sphere_projection(B0,M)
	n = Int(sqrt(length(A)))
	m = Int(length(B)/n)
	objective(x) = energy(pgd_optimizer(y -> energy(y,x0,inverse_gramian(A,reshape(x,n,m))),x -> proj_median_state(x,gramian(A,reshape(x,n,m))),xstar),x0,inverse_gramian(A,reshape(x,n,m)))
	costheta = 0.
	numits = 0
	while costheta > 1-tol
		step = copy(initstep)
		B0 = copy(B1)
		B0V = reshape(B0,length(B0),1)
		grad = ForwardDiff.gradient(objective,B0V)
		proj_grad = tangent_projection(grad,B0,M)
		B1 = sphere_projection(reshape(B0V - step*proj_grad,n,m),M)
		B1V = reshape(B1,length(B1),1)
		costheta = dot(B1V,B0V) / (norm(B1V)*norm(B0V))
		numits+=1
	end
	return B1,numits
end

function lambdamax(A,B;t0=0.,t1=1.)
	mid = reshape(B,length(A[:,1]),Int64(length(B)/length(A[:,1])))
	u,v = eigen(gramian(A,mid,t0,t1))
	l = maximum(u)
	return l
end

function optimal_var_state(B,A,eta,x0;t0=0.,t1=1.)
	gram = gramian(A,B,t0,t1)
	u,v = eigen(gram)
	l = maximum(real.(u))
	index = 0
	for i=1:length(u)
		if u[i] == l
			index = i
		end
	end

	omega = real.(v[:,index])

	x = exp(A*(t1-t0))*x0
	x += eta.^(1/2) *omega
	return x
end

function laplacian(A)
	L = zeros(size(A))
	L[diagind(L)] = sum(A,dims=1)
	L -= A
	return L
end

function max_eigvec(A)
	u,v = eigen(A)
	l = maximum(real.(u))
	index = 0
	for i=1:length(u)
		if u[i] == l
			index = i
		end
	end

	omega = real.(v[:,index])

	return omega
end

function max_eigvec(A,B)
	u,v = eigen(A,B)
	l = maximum(real.(u))
	index = 0
	for i=1:length(u)
		if u[i] == l
			index = i
		end
	end

	omega = real.(v[:,index])

	return omega
end

function len(l,M,xf)
	n = length(M[1,:])
	D = I - (1/n)*ones(n,n)
	H = (M - l*(D^2))
	if rank(H) == n
		x = H\(M*xf)
	else
		x = ones(n).*Inf
	end
	return norm(D*x)^2
end

function var_state(lambda,M,xf)
	n = length(M[1,:])
	D = I - (1/n)*ones(n,n)
	if fakerank(M-lambda*(D^2)) == n
		xs = (M - lambda*(D^2)) \ (M*xf)
	else
		xs = zeros(4)
	end
	return xs
end

function var_solver(M,xf,eta)
	n = length(M[1,:])
	D = (I - (1/n)*ones(n,n))^2
	u,v = eigen(M,D)
	u = ifelse.(u .< 0, Inf, u)
	mu = minimum(u)
	f(x) = len(x,M,xf)-eta
	interval = 1e-20
	while isnan(f(mu-interval))
		interval *=10
	end
	l = find_zero(f,(0,mu-interval))
	x = var_state(l,M,xf)
	return l,x
end

function var_energy_vec(b,A,x0,eta;t0=0.,t1=1.)
	n = length(A[1,:])
	B = reshape(b,n,Int64(length(b)/n))
	M = inverse_gramian(A,B,t0,t1)
	xf = exp(A*(t1-t0))*x0
	l,x = var_solver(M,xf,eta)
	e = energy(x,x0,M)
	return e
end

function var_eig(b,A,eta;t0=0.,t1=1.)
	n = length(A[1,:])
	B = reshape(b,n,Int64(length(b)/n))
	M = inverse_gramian(A,B,t0,t1)
	D = (I - (1/n)*ones(n,n))^2
	u,v = eigen(M,D)
	u = ifelse.(u .< 0, Inf, u)
	mu = minimum(u)
	return mu
end
