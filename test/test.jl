
nxs = [0,2]
nthetas = [1,2]
nreps = [1,5]
nobs = 5

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
            if nx > 0
                max_lik = (ModelCalibration.loglik(data,mle[1],disc_mean,mle[2][1,1],model))
                adjust = 1e-4 .* (scales.theta.max .- scales.theta.min)
                @test max_lik >= ModelCalibration.loglik(data,mle[1] .+ adjust,disc_mean,mle[2][1,1],model)
                @test max_lik >= ModelCalibration.loglik(data,mle[1] .- adjust,disc_mean,mle[2][1,1],model)
                adjust = min((1e-6 .* (scales.y.max .- scales.y.min)).^2,0.98*mle[2][1,1])
                @test max_lik >= ModelCalibration.loglik(data,mle[1],disc_mean,mle[2][1,1].+ adjust,model)
                @test max_lik >= ModelCalibration.loglik(data,mle[1],disc_mean,mle[2][1,1].- adjust,model)

                delta = 1e-3
                bounds = ModelCalibration.find_lik_asymptote(model,data,mle[1],mle[2])
                for i in 1:ntheta
                    thetas = bounds[1]
                    lik_1 = ModelCalibration.loglik(data,thetas,disc_mean,mle[2][1,1],model)
                    thetas[i] -= delta
                    lik_2 = ModelCalibration.loglik(data,thetas,disc_mean,mle[2][1,1],model)
                    @test abs(exp.(abs(lik_2) - abs(lik_1))) <= 1e-20 .* exp(max_lik)
                    thetas = bounds[2]
                    lik_1 = ModelCalibration.loglik(data,thetas,disc_mean,mle[2][1,1],model)
                    thetas[i] += delta
                    lik_2 = ModelCalibration.loglik(data,thetas,disc_mean,mle[2][1,1],model)
                    @test abs(exp.(abs(lik_2) - abs(lik_1))) <= 1e-20 .* exp(max_lik)
                end
            else
                max_lik = ModelCalibration.loglik(data,mle[1],mle[2][1,1],model)
                adjust = 1e-4 .* (scales.theta.max .- scales.theta.min)
                @test max_lik >= ModelCalibration.loglik(data,mle[1] .+ adjust,mle[2][1,1],model)
                @test max_lik >= ModelCalibration.loglik(data,mle[1] .- adjust,mle[2][1,1],model)
                adjust = min((1e-6 .* (scales.y.max .- scales.y.min)).^2,0.98*mle[2][1,1])
                if (max_lik - ModelCalibration.loglik(data,mle[1],mle[2][1,1].+ adjust,model)) > -1e-5
                    println("max likelihood = $max_lik")
                    println("new likelihood = $(ModelCalibration.loglik(data,mle[1],mle[2][1,1].+ adjust,model))")
                end
                if (max_lik - ModelCalibration.loglik(data,mle[1],mle[2][1,1].- adjust,model)) > -1e-5
                    println("max likelihood = $max_lik")
                    println("new likelihood = $(ModelCalibration.loglik(data,mle[1],mle[2][1,1].- adjust,model))")
                end
                delta = 1e-3
                bounds = ModelCalibration.find_lik_asymptote(model,data,mle[1],mle[2])
                for i in 1:ntheta
                    thetas = bounds[1]
                    lik_1 = ModelCalibration.loglik(data,thetas,mle[2][1,1],model)
                    thetas[i] -= delta
                    lik_2 = ModelCalibration.loglik(data,thetas,mle[2][1,1],model)
                    @test abs(exp(lik_2) - exp.(lik_1)) <= 1e-12 .* exp(max_lik)
                    thetas = bounds[2]
                    lik_1 = ModelCalibration.loglik(data,thetas,mle[2][1,1],model)
                    thetas[i] += delta
                    lik_2 = ModelCalibration.loglik(data,thetas,mle[2][1,1],model)
                    @test abs(exp(lik_2) - exp.(lik_1)) <= 1e-12 .* exp(max_lik)
                end
            end

            theta_grid,grid_response = ModelCalibration.generate_sample_grid(30,data,nx,ntheta,model)
            for i in 1:ntheta
                println(bounds[1][i])
                println(maximum(theta_grid[:,i],dims=1))
                println(bounds[2][i])
                println(minimum(theta_grid[:,i],dims=1))
                @test prod(maximum(theta_grid[:,i],dims=1) .<= bounds[1][i])
                @test prod(minimum(theta_grid[:,i],dims=1) .>= bounds[2][i])
            end

            grid_info = ModelCalibration.GridData(
                phi = ModelCalibration.GridPar(
                    density=40,
                    bounds = ModelCalibration.ScalePar(
                        min = 1e-2,#1e-30,
                        max = 10000.0#1e-29
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

            c_sse,log_det_sig,sig_design = ModelCalibration.preallocate(theta_grid,
                grid_info,nx)

            @test size(c_sse)[1] == size(theta_grid)[1]
            if nx > 0
                @test size(c_sse)[3] == grid_info.phi.density
                @test size(c_sse)[2] == grid_info.rho.density
                
                @test size(c_sse)[3] == size(log_det_sig)[2]
                @test size(c_sse)[2] == size(log_det_sig)[1]
                
                @test size(sig_design)[1] == grid_info.rho.density
                @test size(sig_design)[2] == grid_info.phi.density
            end
            ModelCalibration.precompute!(grid_response,data,c_sse,log_det_sig,sig_design,nx)

            test_idx = [rand(1:size(c_sse)[i],max(5,round(Int,0.01*size(c_sse)[i])))
                for i in eachindex(size(c_sse))]

            if length(size(c_sse)) > 1
                for k in test_idx[3]
                    for j in test_idx[2]
                        for i in test_idx[1]
                            rho_vec = repeat([sig_design[j,k,1]],nx)
                            corr = ModelCalibration.correlation_construct(rho_vec,
                                data.exp.x,nx,nloc)
                            sig = Matrix(1.0I,nloc,nloc) +
                                sig_design[j,k,2] .* corr
                            y_hat = grid_response[i,:]
                            #@test log_det_sig[j,k] == log(det(sig)^(-nloc/2))
                            y_bar = vec(mean(data.exp.y,dims=2))
                            val = ModelCalibration.scaled_sse(y_hat,inv(sig),y_bar,nobs)
                            #@test val == c_sse[i,j,k]
                        end
                    end
                end
            else
                for i in test_idx[1]
                    y_hat = grid_response[i,1]
                    println(y_hat)
                    y = mean(data.exp.y,dims=2)
                    val = ModelCalibration.scaled_sse(y_hat,y)
                    @test val == c_sse[i]
                end
            end


            sampling_vars = ModelCalibration.init_vars(nmcmc)
            @test length(sampling_vars.theta) == nmcmc
            @test length(sampling_vars.sig2) == nmcmc
            @test length(sampling_vars.rho) == nmcmc
            @test length(sampling_vars.phi) == nmcmc

            @test typeof(sampling_vars.theta[1]) == Int
            @test typeof(sampling_vars.rho[1]) == Int
            @test typeof(sampling_vars.phi[1]) == Int
            @test typeof(sampling_vars.sig2[1]) == Float64

            ModelCalibration.griddy_gibbs!(data,sampling_vars,c_sse,log_det_sig,sig_design,
                priors,nmcmc)
        
            ModelCalibration.post_process(sampling_vars,data,nx,ntheta,nburn,scales,theta_grid,sig_design,
                grid_response;mdl_apnd="test")
        end
    end
end