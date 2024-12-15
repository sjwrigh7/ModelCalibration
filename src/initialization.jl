#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
######################## Define initialization Functions ##########################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
"""
    init_vars(data::DataStr,nmcmc::Int,nx::Int,ntheta::Int,theta_init::Vector{Float64})
Function to initialize structure to store samples for continuous domain MCMC sampling.

---
Keyword arguments:
* `data::DataStr` Strcture containing computer simulator and experimental data.
* `nmcmc::Int` Integer specifying the number of MCMC samples.
* `nx::Int` Integer specifying the number of independent control variables (x).
* `ntheta::Int` Integer specifying the number of unknown computer model variabels (θ).
* `theta_init::Vector{Float64}` Vector of length `ntheta` specifying the values at which to initialize the θ variables for MCMC.

---
Returns
* `::BulkVarsStruct` Data structure containing initialized Arrays to store the sampled values for each variable in the model.
"""
function init_vars(data::DataStr,nmcmc::Int,
    nx::Int,ntheta::Int,theta_init::Vector{Float64})
    nobs = size(data.exp.x)[1]

    theta = repeat(theta_init',nmcmc)
    tau2 = repeat([0.5],nmcmc)
    sig2 = repeat([0.5],nmcmc)
    delta = repeat([0.0],nmcmc,nobs)
    eta = repeat([0.0],nmcmc,nobs)
    rho = repeat([0.001],nmcmc,nx)
    accept = repeat([0.0],nmcmc,nx+ntheta)
    ratio = repeat([0.0],nmcmc,nx+ntheta)

    return BulkVarsStruct(theta=theta,tau2=tau2,sig2=sig2,
        delta=delta,eta=eta,rho=rho,accept=accept,
        ratio=ratio)
end

"""
    init_vars(data::DataStr,nmcmc::Int)
Function to initialize structure to store samples for discrete domain griddy Gibbs MCMC sampling.

---
Keyword arguments:
* `data::DataStr` Strcture containing computer simulator and experimental data.
* `nmcmc::Int` Integer specifying the number of MCMC samples.
* `ntheta::Int` Integer specifying the number of unknown computer model variabels (θ).

---
Returns
* `::GriddyVarsStruct` Data structure containing initialized Arrays to store the sampled values for each variable in the model.
"""
function init_vars(data::DataStr,nmcmc::Int)

    theta = repeat([1],nmcmc)
    tau2 = repeat([0.5],nmcmc)
    sig_star2 = repeat([1],nmcmc)
    rho = repeat([1],nmcmc)
    
    return GriddyVarsStruct(theta,tau2,sig_star2,rho)
end

