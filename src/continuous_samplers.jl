#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########################### Define Posterior Functions ############################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

"""
    mcmc!(model,data::DataStr,prior_data::PriorData,sample_vals::BulkVarsStruct,start::Int,
    stop::Int,stepsize::StepSize,nx::Int,ntheta::Int,nloc::Int)
Function to perform the Markov chain Monte Carlo simulation to draw from the posterior distributions of the parameters.

---
Positional arguments
* `model` Surrogate model of computer simulator.
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `prior_data::PriorData` Struct containing the prior distribution data for the model parameters.
* `sample_vals::BulkVarsStruct` The struct to store the sampling results.
* `start::Int` The index in `sample_vals` at which to start the sampling.
* `stop::Int` The index in `sample_vals` at which to stop the sampling.
* `stepsize::StepSize` Struct containing the stepsizes for θ and ρ to use for the Metropolis-Hastings algorithm.
* `nx::Int` The number of x dimensions.
* `ntheta::Int` The number of θ dimensions.
* `nloc::Int` The number of unique settings of x in the experimental observations.

---
Details
This function performs MCMC simulation to draw posterior samples for the parameters.
First, the function determines whether to use a univariate data model or not.
Then, samples are drawn from the posterior distributions using the following:
* Metropolis-Hastings updates for θ and ρ
* Gibbs updates for τ^2, σ^2, δ
"""
function mcmc!(model,data::DataStr,prior_data::PriorData,
            sample_vals::BulkVarsStruct,start::Int,stop::Int,
            stepsize::StepSize,nx::Int,ntheta::Int,nloc::Int)

    #initialize constant integer values for posterior calculations
    nrep = size(data.exp.y)[2]
    nloc = size(data.exp.y)[1]
    lik_power = prod(size(data.exp.y))

    # initialize vectors and floats for passing to posterior functions
    delta = copy(sample_vals.delta[start-1,:])
    theta = copy(sample_vals.theta[start-1,:])
    rho = copy(sample_vals.rho[start-1,:])
    sig2 = sample_vals.sig2[start-1]
    tau2 = sample_vals.tau2[start-1]
    eta = sample_vals.eta[start-1,:]
    corr = Array{Float64}(undef,nloc,nloc)

    #check if data model is univariate or multivariate
    if size(data.exp.x)[2] == 0
        #univariate case
        delta .= 0 #set discrepancy to 0
        @inbounds @showprogress 100 "Computing..." for i in start:stop
            #metropolis update for θ
            @inbounds for j in 1:ntheta
                #draw sample
                theta_step = metropolis_theta(model,prior_data,data,
                    theta,sig2,j,stepsize.theta[j])
                #store values
                theta[j] = theta_step.new_value
                sample_vals.theta[i,j] = theta[j]
                sample_vals.accept[i,j+nx] = theta_step.accept
                sample_vals.ratio[i,j+nx] = theta_step.ratio
            end

            #gibbs update for τ^2
            sig2 = gibbs_sig2(prior_data,data,eta,delta,lik_power,nrep)
            sample_vals.sig2[i] = sig2
        end

    else
        #multivariate case
        @inbounds @showprogress for i in start:stop
            #metropolis update for θ
            @inbounds for j in 1:ntheta
                #draw sample
                theta_step = metropolis_theta(model,prior_data,data,
                    theta,delta,sig2,j,stepsize.theta[j],nloc)
                #store values
                eta .= theta_step[2]
                theta[j] = theta_step[1].new_value
                sample_vals.theta[i,j] = theta[j]
                sample_vals.accept[i,j+nx] = theta_step[1].accept
                sample_vals.ratio[i,j+nx] = theta_step[1].ratio
            end
            #store calculated surrogate model prediction
            sample_vals.eta[i,:] .= eta

            #metropolis update for ρ
            @inbounds for j in 1:nx
                #draw sample
                rho_step = metropolis_rho(prior_data,data,
                rho,delta,tau2,j,stepsize.rho[j],nx,nloc)
                #store values
                corr .= rho_step[2]
                rho[j] = rho_step[1].new_value
                sample_vals.rho[i,j] = rho[j]
                sample_vals.accept[i,j] = rho_step[1].accept
                sample_vals.ratio[i,j] = rho_step[1].ratio
            end
            #gibbs update for σ^2
            sig2 = gibbs_sig2(prior_data,data,eta,delta,lik_power,nrep)
            sample_vals.sig2[i] = sig2

            #gibbs update for τ^2
            tau2 = gibbs_tau2(prior_data,delta,corr,nloc)
            sample_vals.tau2[i] = tau2

            #gibbs update for δ
            delta = gibbs_delta(data,sig2,tau2,eta,corr,nloc,nrep)
            sample_vals.delta[i,:] .= delta
        end
    end
end
