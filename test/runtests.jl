using ModelCalibration
using Tests

nx = 2
ntheta = 2
nobs = 5

x = hcat(
    collect(range(0.0,pi,length=nobs)),
    collect(range(10.0,50.0,length=nobs))
)
theta = hcat(
    collect(range(0.1,4.0,length=50)),
    collect(range(0.1,4.0,length=50))
)

design = Array{Float64}(undef,size(x)[1]*size(theta)[1],
    (size(x)[2]+size(theta)[2]))

function true_fn(x::Vector{Float64},theta::Vector{Float64})
    y = theta[1]*sin(theta[2]*x[1]) + x[2]
    return y
end

simobs = Vector{Float64}(undef,size(design)[1])

for i in eachindex(simobs)
    simobs[i] = true_fn(x[i,:],theta[i,:])
end

rho = repeat([0.95],size(x)[2])
sig2 = 0.7
disc_corr = correlation_construct(rho,x,nx,nobs)
disc_covar = sig2 .* disc_corr
disc_mean = rand(MvNormal(repeat([0.0],nobs),disc_covar),1)
delta = rand(disc_mean,disc_covar)

tau2 = 0.3
error = rand(MvNormal(repeat([0.0],nobs),
    tau2 .* Matrix(1.0I,nobs,nobs)),1)

expobs = Array{Float64}(undef,nobs,nx+1)
expobs[:,1:nx] .= x

theta_true = [0.4,3.3]
for i in axes(expobs)[1]
    expobs[i,end] = true_fn(expobs[i,1:nx],theta_true)
end

expobs[:,end] .= expobs[:,end] .+ delta .+ error

nmcmc = 10000
nburn = 5000
ntrialburn = 100
ntrial = 100

alpha_tau2 = 1e-9
alpha_sig2 = 1e-9
beta_tau2 = 1e-9
beta_sig2 = 1e-9
a_rho = 1.0
b_rho = 1.0


@testset "setup" begin
    try
        nobs,nrep,nloc,scales,data,priors = setup(design,
        simobs,expobs,nx,ntheta,alpha_tau2,beta_tau2,
        alpha_sig2,beta_sig2,a_rho,b_rho)
    catch
        @test false
    end

    @test nobs == size(expobs)[1]
    @test nloc == size(unique(expobs[:,1:nx],dims=1))[1]
    @test nrep == round(Int,size(expobs)[1]/nobs)
    for i in 1:nx
        @test scales.x.min[i] == minimum(expobs[:,i])
        @test scales.x.max[i] == maximum(expobs[:,i])
    end
    for i in 1:ntheta
        @test scales.theta.min[i] == minimum(design[:,nx+i])
        @test scales.theta.max[i] == maximum(design[:,nx+i])
    end
    @test scales.y.min == minimum(simobs)
    @test scales.y.max == maximum(simobs)

    @test data.sim.x == (design[:,1:nx] .- scales.x.min) ./
        (scales.x.max .- scales.x.min)
    @test data.sim.theta == (design[:,(nx+1):(nx+ntheta)] .-
     scales.x.min) ./ (scales.theta.max .- scales.theta.min)
    @test data.sim.y == (simobs .- scales.y.min) ./
        (scales.y.max .- scales.y.min)
    @test data.exp.x == (expobs[:,1:nx] .- scales.x.min) ./
        (scales.x.max .- scales.x.min)
    for i in 1:nrep
        start = (i-1)*nloc + 1
        stop = i*nloc
        @test data.exp.y[:,i] == (expobs[start:stop,end] .- 
            scales.y.min) ./ (scales.y.max .- scales.y.min)
    end

    @test priors.sig2.par1 == alpha_sig2
    @test priors.sig2.par2 == beta_sig2
    @test priors.tau2.par1 == alpha_tau2
    @test priors.tau2.par2 == beta_tau2
    @test priors.rho.par1 == a_rho
    @test priors.rho.par2 == b_rho
    for i in 1:ntheta
        @test priors.theta.par1[i] == scales.theta.min[i]
        @test priors.theta.par2[i] == scales.theta.max[i]
    end
end

@testset "surrogate model" begin
    model = surrogate_model(data.sim.x,data.sim.theta,
        data.sim.y,nx,ntheta)
    test_idx = rand(axes(data.sim.theta)[1],20)
    test_set = data.sim.theta[test_idx,:]
    for i in axes(test_set)[1]
        theta = test_set[i,(nx+1):(nx+ntheta)]
        eta = predict_y_all(theta,model)
        train = data.sim.y[test_idx[i],:]
        @test isapprox(eta,train)
    end
end

#TODO edit likelihood function to not need full vars and data inputs
#TODO add multiple dispatches of the likelihood function
@testset "maximum likelihood calculation" begin
    mle = get_mle(data,nx,ntheta)
    max_lik = loglik(mle[1],mle[2])
    adjust = 1e-4 .* (scales.theta.max .- scales.theta.min)
    @test max_lik >= loglik(mle[1] .+ adjust,mle[2])
    @test max_lik >= loglik(mle[1] .- adjust,mle[2])
    adjust = (1e-6 .* (scales.y.max .- scales.y.min)).^2
    @test max_lik >= loglik(mle[1],mle[2] .+ adjustment)
    @test max_lik >= loglik(mle[1],mle[2] .- adjustment)
end

###############
# griddy gibbs method
@testset "theta bounds" begin
    delta = 1e-3
    bounds = find_lik_asymptote(data,theta_mle,covar_mle,delta)
    for i in 1:ntheta
        thetas = bounds[1]
        lik_1 = lik(thetas,mle[2])
        thetas[i] -= delta
        diff = lik(thetas,mle[2]) - lik_1
        @test abs(diff) <= 1e-7
        thetas = bounds[2]
        lik_1 = lik(thetas,mle[2])
        thetas[i] += delta
        diff = lik(thetas,mle[2])
        @test abs(diff) <= 1e-7
    end
end

#TODO add grid responses to output of this function
@testset "Grid generation" begin
    doe = generate_sample_grid(30,data,nx,theta)
    for i in 1:ntheta
        @test prod(doe[:,i] .< bounds[1][i])
        @test prod(doe[:,i] .> bounds[2][i])
    end
end
@testset "generate grid info" begin
    try    
        grid_info = GridData(
            sig_star2 = GridPar(
                density=40,
                bounds = ScalePar(
                    min = 1e-2,
                    max = 10000.0
                )
            ),
            rho = GridPar(
                density=10,
                bounds = ScalePar(
                    min=0.0001,
                    max = 0.9999
                )
            )
        )
    catch
        @test false
    end
end

@testset "griddy gibbs preallocation" begin
    try
        c_sse,log_det_sig,sig_design = preallocate(data,
            grid_info)
    catch
        @test false
    end
    @test size(c_sse)[1] == size(doe)[1]
    @test size(c_sse)[3] == grid_info.sig_star2.density
    @test size(c_sse)[2] == grid_info.rho.density

    @test size(c_sse)[3] == size(log_det_sig)[1]
    @test size(c_see)[2] == size(log_det_sig)[2]

    @test size(sig_design)[1] == grid_info.rho.density
    @test size(sig_design)[1] == grid_info.sig_star2.density
end

@testset "griddy gibbs precomputation" begin
    precompute!(data,c_sse,log_det_sig,sig_design)

    test_idx = [rand(1:size(c_sse)[i],min(5,0.01*size(c_sse)[i]))
        for i in eachindex(size(c_sse))]
    for k in test_idx[3]
        for j in test_idx[2]
            for i in test_idx[1]
                rho_vec = repeat([sig_design[j,k,1]],nx)
                corr = correlation_construct(rho_vec,
                    data.exp.x,nx,nloc)
                sig = Matrix(1.0I,nloc,nloc) +
                    sig_design[j,k,2] .* corr
                y_hat = grid_resp[i,:]
                @test log_det_sig[j,k] == det(sig)
                y_bar = mean(data.exp.y,dims=2)
                val = scaled_sse(y_hat,inv(sig),y_bar,nobs)
                @test val == c_sse[i,j,k]
            end
        end
    end
end

@testset "initialize variables" begin
    try
        sampling_vars = init_vars(data,nmcmc)
    catch
        @test false
    end
    @test length(sampling_vars.theta) == nmcmc
    @test length(sampling_vars.tau2) == nmcmc
    @test length(sampling_vars.rho) == nmcmc
    @test length(sampling_vars.sig_star2) == nmcmc

    @test typeof(sampling_vars.theta[1]) == Int
    @test typeof(sampling_vars.rho[1]) == Int
    @test typeof(sampling_vars.sig_star2[1]) == Int
    @test typeof(sampling_vars.tau2[1]) == Float64
end

#TODO add tests for sampling scheme

#TODO add tests for continuous method
# initializing variables at theta optimum
# step size algorithm
# sampling scheme

#TODO add tests for post processing
