#using Pkg
#Pkg.activate(".")
using ModelCalibration
using Test
using Distributions
using LinearAlgebra

nx = 2
ntheta = 2
nloc = 5
nobs = 5

x = hcat(
    collect(range(0.0,pi,length=nobs)),
    collect(range(10.0,50.0,length=nobs))
)

theta_template = hcat(
    collect(range(0.1,4.0,length=8)),
    collect(range(0.1,4.0,length=8))
)
theta = Array{Float64}(undef,size(theta_template,1)^2,2)
let 
    ita = 0
    for j in 1:8
        for i in 1:8
            ita += 1
            theta[ita,1] = theta_template[i,1]
            theta[ita,2] = theta_template[j,2]
        end
    end
end


design = Array{Float64}(undef,size(x)[1]*size(theta)[1],
    (size(x)[2]+size(theta)[2]))

function true_fn(x::Vector{Float64},theta::Vector{Float64})
    y = theta[1]*sin(theta[2]*x[1]) + x[2]
    return y
end

simobs = Vector{Float64}(undef,size(design)[1])

let
    ita = 0
    for j in axes(theta,1)
        for i in axes(x,1)
            ita += 1
            simobs[ita] = true_fn(x[i,:],theta[j,:])
            design[ita,1:nx] = x[i,:]
            design[ita,(nx + 1):end] = theta[j,:]
        end
    end
end

rho = repeat([0.95],size(x)[2])
sig2 = 0.7
disc_corr = ModelCalibration.correlation_construct(rho,x,nx,nobs)
disc_covar = sig2 .* disc_corr
disc_mean = rand(MvNormal(repeat([0.0],nobs),disc_covar))
delta = rand(MvNormal(disc_mean,disc_covar))

tau2 = 0.3

expobs = Array{Float64}(undef,nobs*nloc,nx+1)
expobs[:,1:nx] .= repeat(x,outer=nobs)

theta_true = [0.4,3.3]
true_resp = Vector{Float64}(undef,nloc)
for i in 1:nloc
    true_resp[i] = true_fn(x[i,:],theta_true)
end

for i in 1:nobs
    start_idx = (i-1) * nloc + 1
    stop_idx = i * nloc

    expobs[start_idx:stop_idx,end] = true_resp .+ delta .+ rand(Normal(0.0,tau2),nloc)
end

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
    #try
        global nobs,nrep,nloc,scales,data,priors = ModelCalibration.setup(
            design,
            simobs,
            expobs,
            nx,
            ntheta,
            alpha_tau2,
            beta_tau2,
            alpha_sig2,
            beta_sig2,
            a_rho,
            b_rho
        )
    #catch
    #    @test false
    #end

    @test nobs == size(expobs)[1]
    @test nloc == size(unique(expobs[:,1:nx],dims=1))[1]
    @test nrep == round(Int,size(expobs)[1]/nloc)
    for i in 1:nx
        @test scales.x.min[i] == minimum(expobs[:,i])
        @test scales.x.max[i] == maximum(expobs[:,i])
    end
    for i in 1:ntheta
        @test scales.theta.min[i] == minimum(design[:,nx+i])
        @test scales.theta.max[i] == maximum(design[:,nx+i])
    end
    @test scales.y.min == min(
        minimum(simobs),
        minimum(expobs[:,end])
    )
    @test scales.y.max == max(
        maximum(simobs),
        maximum(expobs[:,end])
    )

    println(size(data.sim.y))
    println(size(data.sim.theta))
    @test data.sim.x == (design[1:nloc,1:nx] .- scales.x.min') ./
        (scales.x.max' .- scales.x.min')
    @test data.sim.theta == (design[1:nloc:end,(nx+1):(nx+ntheta)] .-
     scales.theta.min') ./ (scales.theta.max' .- scales.theta.min')
    @test vec(data.sim.y) == (simobs .- scales.y.min) ./
        (scales.y.max .- scales.y.min)
    @test data.exp.x == (expobs[1:nloc,1:nx] .- scales.x.min') ./
        (scales.x.max' .- scales.x.min')
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
    @test all(priors.rho.par1 .== a_rho)
    @test all(priors.rho.par2 .== b_rho)
    for i in 1:ntheta
        @test priors.theta.par1[i] == 0.0#scales.theta.min[i]
        @test priors.theta.par2[i] == 1.0#cales.theta.max[i]
    end
end

@testset "surrogate model" begin
    global model = ModelCalibration.surrogate_model(data.sim.x,data.sim.theta,
        data.sim.y,nx,ntheta)
    test_idx = rand(axes(data.sim.theta)[1],20)
    test_set = data.sim.theta[test_idx,:]
    for i in axes(test_set)[1]
        theta = test_set[i,:]
        eta = ModelCalibration.predict_y_all(theta,model)
        train = data.sim.y[:,test_idx[i]]
        err = sum((eta .- train).^2)
        println(err)
        if err > 1e-4
            @warn "Higher than expected error in surrogate model"
        end
        #@test err <= 1e-3
        #@test all(isapprox.(eta,train))
    end
end

#TODO edit likelihood function to not need full vars and data inputs
#TODO add multiple dispatches of the likelihood function
@testset "maximum likelihood calculation" begin
    global theta_mle,covar_mle = ModelCalibration.get_mle(data,nx,nloc,ntheta,model)
    delta = repeat([0.0],nloc)
    max_lik = ModelCalibration.loglik(data,theta_mle,delta,covar_mle[1,1],model)
    adjust = 1e-4 .* (scales.theta.max .- scales.theta.min)
    up_lik = ModelCalibration.loglik(data,theta_mle .+ adjust,delta,covar_mle[1,1],model)
    down_lik = ModelCalibration.loglik(data,theta_mle .- adjust,delta,covar_mle[1,1],model)
    if max_lik * 1.1 <= up_lik
        @warn "Higher than expected error in likelihood optimization"
    end
    if max_lik * 1.1 <= down_lik
        @warn "Higher than expected error in likelihood optimization"
    ene
    adjust = (1e-4 .* (scales.y.max .- scales.y.min)).^2
    up_lik = ModelCalibration.loglik(data,theta_mle,delta,covar_mle[1,1] + adjust,model)
    down_lik = ModelCalibration.loglik(data,theta_mle,delta,covar_mle[1,1] - adjust,model)
    if max_lik * 1.1 <= up_lik
        @warn "Higher than expected error in likelihood optimization"
    end
    if max_lik * 1.1 <= down_lik
        @warn "Higher than expected error in likelihood optimization"
    end
end

###############
# griddy gibbs method
@testset "theta bounds" begin
    delta = 1e-3
    disc = repeat([0.0],nloc)
    #theta_mle = mle[1]
    #covar_mle = mle[2]
    global bounds = ModelCalibration.find_lik_asymptote(model,data,theta_mle,covar_mle;delta)
    max_lik = ModelCalibration.lik(data,theta_mle,disc,covar_mle[1,1],model)
    for i in 1:ntheta
        thetas = copy(bounds[1])
        lik_1 = ModelCalibration.lik(data,thetas,disc,covar_mle[1,1],model)
        println(lik_1)
        thetas[i] -= delta
        lik_2 = ModelCalibration.lik(data,thetas,disc,covar_mle[1,1],model)
        diff = (lik_2 - lik_1) / max_lik
        if abs(diff) >= 1e-5
            @warn "Higher than expected error in estimated liklihood asymptotes"
        end
        thetas = copy(bounds[2])
        lik_1 = ModelCalibration.lik(data,thetas,disc,covar_mle[1,1],model)
        thetas[i] += delta
        lik_2 = ModelCalibration.lik(data,thetas,disc,covar_mle[1,1],model)
        diff = (lik_2 - lik_1) / max_lik
        if abs(diff) >= 1e-5
            @warn "Higher than expected error in estimated likelihood asymptotes"
        end
    end
end

#TODO add grid responses to output of this function
@testset "Grid generation" begin
    global doe,resp = ModelCalibration.generate_sample_grid(30,data,nx,ntheta,model)
    for i in 1:ntheta
        @test all(doe[:,i] .<= bounds[1][i])
        @test all(doe[:,i] .>= bounds[2][i])
    end
end
@testset "generate grid info" begin
    #try    
        global grid_info = ModelCalibration.GridData(
            phi = ModelCalibration.GridPar(
                density=40,
                bounds = ModelCalibration.ScalePar(
                    min = 1e-2,
                    max = 10000.0
                )
            ),
            rho = ModelCalibration.GridPar(
                density=10,
                bounds = ModelCalibration.ScalePar(
                    min=0.0001,
                    max = 0.9999
                )
            )
        )
    #catch
    #    @test false
    #end
end

@testset "griddy gibbs preallocation" begin
    #try
        global c_sse,log_det_sig,sig_design = ModelCalibration.preallocate(doe,
            grid_info,nx)
    #catch
    #    @test false
    #end
    println(size(c_sse))
    println(size(log_det_sig))
    @test size(c_sse)[1] == size(doe)[1]
    @test size(c_sse)[3] == grid_info.phi.density
    @test size(c_sse)[2] == grid_info.rho.density

    @test size(c_sse)[2] == size(log_det_sig)[1]
    @test size(c_sse)[3] == size(log_det_sig)[2]

    @test size(sig_design)[1] == grid_info.rho.density
    @test size(sig_design)[2] == grid_info.phi.density
end

@testset "griddy gibbs precomputation" begin
    ModelCalibration.precompute!(
        resp,
        data,
        c_sse,
        log_det_sig,
        sig_design,
        nx
    )

    test_idx = [
        rand(
            1:size(c_sse)[i],
            min(
                5,
                max(
                    1,
                    Int(floor(0.01*size(c_sse)[i]))
                )
            )
        )
        for i in eachindex(size(c_sse))]
    for k in test_idx[3]
        for j in test_idx[2]
            for i in test_idx[1]
                rho_vec = repeat([sig_design[j,k,1]],nx)
                corr = ModelCalibration.correlation_construct(
                    rho_vec,
                    data.exp.x,
                    nx,
                    nloc
                )
                sig = Matrix(1.0I,nloc,nloc) +
                    sig_design[j,k,2] .* corr
                
                y_hat = resp[i,:]
                
                @test log_det_sig[j,k] == log(det(sig)^(-nloc/2))
                
                y_bar = vec(mean(data.exp.y,dims=2))
                val = ModelCalibration.scaled_sse(y_hat,inv(sig),y_bar,nrep)
                @test val == c_sse[i,j,k]
            end
        end
    end
end

@testset "initialize variables" begin
    #try
        sampling_vars = ModelCalibration.init_vars(nmcmc)
    #catch
    #    @test false
    #end
    @test length(sampling_vars.theta) == nmcmc
    @test length(sampling_vars.sig2) == nmcmc
    @test length(sampling_vars.rho) == nmcmc
    @test length(sampling_vars.phi) == nmcmc

    @test typeof(sampling_vars.theta[1]) == Int
    @test typeof(sampling_vars.rho[1]) == Int
    @test typeof(sampling_vars.phi[1]) == Int
    @test typeof(sampling_vars.sig2[1]) == Float64
end

#TODO add tests for sampling scheme

#TODO add tests for continuous method
# initializing variables at theta optimum
# step size algorithm
# sampling scheme

#TODO add tests for post processing
