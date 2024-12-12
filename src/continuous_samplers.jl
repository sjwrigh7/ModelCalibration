#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########################### Define Posterior Functions ############################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
"""
    metropolis_update(proprosal::Function,posterior:Function,)
"""
function metropolis_update(proposal::Function,posterior::Function,var::Float64)
    new_value = proposal(var)
    current_posterior = posterior(var)
    proposal_posterior = posterior(var)

end
"""
    @metropolis_sample param xarg
Macro to generate generalized code for a single metropolis sample.

---
Inputs
* `param` The variable for which to sample. The value should be `theta` or `rho`
* `xarg` An extra argument to use an augmented metropolis sample function call. The only currently supported argument is `nox`.

---
Details
This macro generates code to perform the sample of the parameter.
The data from the macro is then stored appropriately in the proper structs of the `mcmc` function.
"""
macro metropolis_sample(param,xarg=nothing)
    if xarg == :nox
        app = "_nox"
    else
        app = ""
    end
    if param == :theta
        idx = "j+nx"
    elseif param == :rho
        idx = "j"
    end
    func_call = "$(param)_step = metropolis_$(param)$(app)(model,prior_data,data,
        step_vars,j,stepsize.$param[j])"
    var_store_step = "step_vars.$param[j] = $(param)_step.new_value"
    var_store_bulk = "bulk_vars.$param[i,j] = step_vars.$param[j]"
    accept_store = "bulk_vars.accept[i,$idx] = $(param)_step.accept"
    ratio_store = "bulk_vars.ratio[i,$idx] = $(param)_step.ratio"
    
    println(Meta.parse(func_call))
    println(var_store_step)
    println(var_store_bulk)
    println(accept_store)
    println(ratio_store)

    func_call = Meta.parse(func_call)
    var_store_step = Meta.parse(var_store_step)
    var_store_bulk = Meta.parse(var_store_bulk)
    accept_store = Meta.parse(accept_store)
    ratio_store = Meta.parse(ratio_store)
    
    return quote
        $(esc(func_call))
        $(esc(var_store_step))
        $(esc(var_store_bulk))
        $(esc(accept_store))
        $(esc(ratio_store))
    end
end

"""
    mcmc(data::DataStr,prior_data::PriorData,bulk_vars::BulkVarsStruct,start::Int,stop::Int,stepsize::StepSize,nx::Int,ntheta::Int,nobs::Int)
Function to perform the Markov chain Monte Carlo simulation to draw from the posterior distributions of the parameters.

---
Keyword arguments
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `prior_data::PriorData` Struct containing the prior distribution data for the model parameters.
* `bulk_vars::BulkVarsStruct` The struct to store the sampling results.
* `start::Int` The index in `bulk_vars` at which to start the sampling.
* `stop::Int` The index in `bulk_vars` at which to stop the sampling.
* `stepsize::StepSize` Struct containing the stepsizes for θ and ρ to use for the Metropolis-Hastings algorithm.
* `nx::Int` The number of x dimensions.
* `ntheta::Int` The number of θ dimensions.
* `nobs::Int` The number of unique settings of x in the experimental observations.

---
Details
This function performs MCMC simulation to draw posterior samples for the parameters.
First, the function determines whether to use a univariate data model or not.
Then, samples are drawn from the posterior distributions using the following:
* Metropolis-Hastings updates for θ and ρ
* Gibbs updates for τ^2, σ^2, δ
"""
function mcmc!(model,data::DataStr,prior_data::PriorData,
    bulk_vars::BulkVarsStruct,start::Int,stop::Int,
    stepsize::StepSize,nx::Int,ntheta::Int,nloc::Int)

    step_vars = UpdatedVars(theta=bulk_vars.theta[start-1,:],
        delta=bulk_vars.delta[start-1,:],tau2=bulk_vars.tau2[start-1],
        sig2=bulk_vars.sig2[start-1],rho=bulk_vars.rho[start-1,:])

    if size(data.exp.x)[2] == 0
        #precompute for univariate likelihood
        sum_y_2 = sum(data.exp.y).^2
        sum_y_1 = sum(data.exp.y)

        @inbounds @showprogress 100 "Computing..." for i in start:stop
            #metropolis update for θ
            @inbounds for j in 1:ntheta
                theta_step = metropolis_theta_nox(model,prior_data,data,
                    step_vars.theta,step_vars.delta,step_vars.tau2,
                    j,stepsize.theta[j])
                step_vars.theta[j] = theta_step.new_value
                bulk_vars.theta[i,j] = step_vars.theta[j]
                bulk_vars.accept[i,j+nx] = theta_step.accept
                bulk_vars.ratio[i,j+nx] = theta_step.ratio
            end

            #gibbs update for τ^2
            step_vars.tau2 = gibbs_tau2(prior_data,data,step_vars)
            bulk_vars.tau2[i] = step_vars.tau2
        end

    else
        @inbounds @showprogress for i in start:stop
            #metropolis update for θ
            @inbounds for j in 1:ntheta
                theta_step = metropolis_theta(model,prior_data,data,
                    step_vars.theta,step_vars.delta,step_vars.tau2,
                    j,stepsize.theta[j])
                step_vars.theta[j] = theta_step.new_value
                bulk_vars.theta[i,j] = step_vars.theta[j]
                bulk_vars.accept[i,j+nx] = theta_step.accept
                bulk_vars.ratio[i,j+nx] = theta_step.ratio
            end
            #metropolis update for ρ
            @inbounds for j in 1:nx
                rho_step = metropolis_rho(prior_data,data,step_vars,  
                j,stepsize.rho[j],nx,nloc)
                step_vars.rho[j] = rho_step.new_value
                bulk_vars.rho[i,j] = step_vars.rho[j]
                bulk_vars.accept[i,j] = rho_step.accept
                bulk_vars.ratio[i,j] = rho_step.ratio
            end
            #gibbs update for τ^2
            step_vars.tau2 = gibbs_tau2(model,prior_data,data,step_vars)
            bulk_vars.tau2[i] = step_vars.tau2
            #gibbs update for σ^2
            step_vars.sig2 = gibbs_sig2(prior_data,data,step_vars,nx,nloc)
            bulk_vars.sig2[i] = step_vars.sig2
            #gibbs update for δ
            step_vars.delta = gibbs_delta(model,data,step_vars,nloc)
            bulk_vars.delta[i,:] = step_vars.delta
        end
    end
    #return bulk_vars
end