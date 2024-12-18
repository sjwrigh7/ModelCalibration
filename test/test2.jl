
nxs = [0,2]
nthetas = [1,2]
nreps = [1,5]
#nobs = 5

for nx in nxs
    for ntheta in nthetas
        for nrep in nreps
            nobs = 6
            println("case: nx = $nx, ntheta = $ntheta, nrep = $nrep")
            function format_exp(expobs,gauss)
                if length(size(gauss)) > 1
                    out = repeat(expobs,size(gauss)[2])
                    gauss = Vector(vec(gauss))
                    out[:,end] .= out[:,end] .+ gauss
                else
                    out = copy(expobs)
                    out[:,end] .= out[:,end] .+ gauss
                end
                return out
            end
            
            function true_fn(theta::Vector{Float64})
                if length(theta) == 1
                    y = exp.(theta)[1]
                elseif length(theta) == 2
                    y = exp(theta[1]) + theta[2]
                end
                return y
            end
            
            function true_fn(x::Vector{Float64},theta::Vector{Float64})
                y = sum(theta .* (x))
                return y
            end
            if nx > 0
                x = [0 0;
                    pi/2 0;
                    0 -pi/2;
                    pi 0;
                    0 -pi;
                    pi -pi]
                bounds = ([4.0 for i in 1:ntheta],[0.1 for i in 1:ntheta])
                theta = ModelCalibration.get_full_fact(8,ntheta,bounds)

                design = Array{Float64}(undef,size(x)[1]*size(theta)[1],size(x)[2]+size(theta)[2])

                for i in axes(theta)[1]
                    start = size(x)[1]*(i-1) + 1
                    stop = size(x)[1]*i
                    design[start:stop,1:nx] .= x
                    design[start:stop,(nx+1):(nx+ntheta)] .= repeat(theta[i,:]',size(x)[1])
                end

                simobs = Vector{Float64}(undef,size(design)[1])
                for i in eachindex(simobs)
                    simobs[i] = true_fn(design[i,1:nx],design[i,(nx+1):end])
                end

                rho = repeat([0.2],size(x)[2])
                tau2 = 0.7
                disc_corr = ModelCalibration.correlation_construct(rho,x,nx,nobs)
                disc_covar = tau2 .* disc_corr
                disc_mean = repeat([0.0],nobs)#rand(MvNormal(repeat([0.0],nobs),disc_covar),1)
                delta = vec(rand(MvNormal(vec(disc_mean),disc_covar),1))

                sig2 = 0.3
                error = rand(MvNormal(repeat([0.0],nobs),
                    sig2 .* Matrix(1.0I,nobs,nobs)),nrep)

                expobs = Array{Float64}(undef,nobs,nx+1)
                expobs[:,1:nx] .= x

                theta_true = [0.4,3.3]
                for i in axes(expobs)[1]
                    expobs[i,end] = true_fn(expobs[i,1:nx],theta_true)
                end

                expobs[:,end] .= expobs[:,end] .+ delta
                expobs = format_exp(expobs, error)
            else
                bounds = ([4.0 for i in 1:ntheta],[0.1 for i in 1:ntheta])
                theta = ModelCalibration.get_full_fact(8,ntheta,bounds)

                design = theta

                simobs = Vector{Float64}(undef,size(design)[1])
                for i in eachindex(simobs)
                    simobs[i] = true_fn(design[i,:])
                end

                expobs = Vector{Float64}(undef,nobs)

                theta_true = [0.4,3.3]
                
                expobs = true_fn(theta_true)

                sig2 = 0.3
                error = rand(Normal(0,sqrt(sig2)),nrep)
                expobs = expobs .+ error
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

            nobs,nrep,nloc,scales,data,priors = ModelCalibration.setup(design,
                simobs,expobs,nx,ntheta,alpha_tau2,beta_tau2,
                alpha_sig2,beta_sig2,a_rho,b_rho)
            
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
            @test scales.y.min == minimum(vcat(simobs,expobs[:,end]))
            @test scales.y.max == maximum(vcat(simobs,expobs[:,end]))
            
            @test data.sim.x == (unique(design[:,1:nx],dims=1) .- scales.x.min') ./
                (scales.x.max' .- scales.x.min')
            @test data.sim.theta == (unique(design[:,(nx+1):(nx+ntheta)],dims=1) .-
                scales.theta.min') ./ (scales.theta.max' .- scales.theta.min')

            nsim = length(unique(design[:,nx+1]))
            for i in 1:nsim
                start = (i-1)*nloc + 1
                stop = i*nloc
                @test data.sim.y[:,i] == (simobs[start:stop] .- scales.y.min) ./
                (scales.y.max .- scales.y.min)
            end
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
            @test priors.rho.par1 == repeat([a_rho],nx)
            @test priors.rho.par2 == repeat([b_rho],nx)
            for i in 1:ntheta
                @test priors.theta.par1[i] == 0.0
                @test priors.theta.par2[i] == 1.0
            end

            model = ModelCalibration.surrogate_model(data.sim.x,data.sim.theta,
                data.sim.y,nx,ntheta)
            println(size(data.sim.theta))
            test_idx = rand(axes(data.sim.theta)[1],20)
            test_set = data.sim.theta[test_idx,:]
            println(test_set)
            for i in axes(test_set)[1]
                theta = test_set[i,:]
                eta = ModelCalibration.predict_y_all(theta,model)
                train = data.sim.y[:,test_idx[i]]
                error = sum(sqrt.((eta .- train).^2))/length(eta)
                #@test error <= 5e-2
                if error > 2.5e-2
                    println("data: $train")
                    println("prediction: $eta")
                    println("RMSE: $error")
                    println("scale: $(error * (scales.y.max - scales.y.min))")
                end
            end

            mle = ModelCalibration.get_mle(data,nx,nloc,ntheta,model)
            
            opt_step_size = ModelCalibration.find_stepsize(model,data,ntrial,ntrialburn,priors,
            nx,ntheta,nloc;theta_init=mle[1],scale=2.0,offset=1.5,shape=10.0,
            save_plots=false)

            
            sampling_vars = ModelCalibration.init_vars(data,nmcmc,nx,ntheta,mle[1])

            @test size(sampling_vars.theta)[1] == nmcmc
            @test size(sampling_vars.theta)[2] == ntheta
            @test length(sampling_vars.sig2) == nmcmc
            @test size(sampling_vars.rho)[1] == nmcmc
            @test size(sampling_vars.rho)[2] == nx
            @test length(sampling_vars.tau2) == nmcmc
            @test size(sampling_vars.delta)[1] == nmcmc
            @test size(sampling_vars.delta)[2] == nloc
            @test size(sampling_vars.delta) == size(sampling_vars.eta)
            @test size(sampling_vars.accept)[1] == nmcmc
            @test size(sampling_vars.accept)[2] == (nx+ntheta)
            @test size(sampling_vars.accept) == size(sampling_vars.ratio)

            @test typeof(sampling_vars.theta[1,1]) == Float64
            @test typeof(sampling_vars.rho[1,1]) == Float64
            @test typeof(sampling_vars.tau2[1]) == Float64
            @test typeof(sampling_vars.sig2[1]) == Float64
            @test typeof(sampling_vars.delta[1,1]) == Float64
            @test typeof(sampling_vars.eta[1,1]) == Float64
            @test typeof(sampling_vars.accept[1,1]) == Bool
            @test typeof(sampling_vars.ratio[1,1]) == Float64

            ModelCalibration.mcmc!(model,data,priors,sampling_vars,2,nmcmc,opt_step_size,nx,ntheta,nloc)

            @test prod(mean(sampling_vars.accept,dims=1) .< 0.38)
            @test prod(mean(sampling_vars.accept,dims=1) .> 0.12)

            ModelCalibration.post_process(sampling_vars,data,nx,ntheta,nburn,scales;mdl_apnd="test",nthin=1)
        end
    end
end