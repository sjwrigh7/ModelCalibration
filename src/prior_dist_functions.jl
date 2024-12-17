#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
############################ Define Prior Functions ###############################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

"""
    prior_theta(prior_data::PriorData,vars::UpdatedVars)
This function evaluates the probability density function of the prior distribution on θ.
Given that θ is modeled to have a uniform prior distribution, this function returns the same value for any value of θ.
Generally, this function is not required for evaluation in a posterior sampling scheme.

---
Positional arguments
* `prior_data::PriorData` Data structure containing the prior distribution hyperparameters for all variables
* `vars::UpdatedVars` Data structure containing the most recently sampled values of all variables in the MCMC sampling scheme.

---
Returns
* `pdf_val` A scalar Float giving the value of the θ prior distribution.

---
Details
Given that θ are all modeled to have uniform prior distributions, this function evaluates to 1/(B-A) where B and A are the upper and lower bounds for the uniform distribution, respectively.
"""
function prior_theta(prior_data::PriorData,vars::UpdatedVars)
    A = prior_data.theta.par1 #min val
    B = prior_data.theta.par2 #max val
    pdf_val = 1/(B-A)         #pdf val
    return pdf_val
end

"""
    log_prior_delta(data::DataStr,delta::Vector{Float64},rho::Vector{Float64},
    tau2::Float64nx::Int64,nloc::Int64)
Function to evaluate the probability density function of the discrepancy term (δ) prior distribution.
This implementation is for a case where the correlation matrix is not yet calculated.

---
Positional arguments
* `data::DataStr` Struct containing the experimental and computer simulator data.
* `delta::Vector{Float64}` The value of delta at which to evaluate the distribution.
* `rho::Vector{Float64}` A Vector of length p containing the ρ values for evaluating the prior distribution density function.
* `tau2::Float64` The variance of the discrepancy function's covariance matrix.
* `nx::Int64` The number of x variables in the model.
* `nloc::Int64` The length of the data model (multivariate normal distribution) for a single independent observation.
Note that `rho` are supplied separately from the other variables contained in `vars` to allow evaluation of the likelihood at different values of ρ without requiring altering the `vars` struct during Matropolis-Hastings updates.

---
Returns
* `log_pdf_val` The log evaluation of the discrepancy term (δ) prior distribution density function.

---
Details
The discrepancy term (δ) assumes a zero mean prior distribution with a covariance structure defined by Σ=τ^2*C.
C is the correlation structure. It is a size n square matrix with each element calculated by the following:
C(x,x') = ∏(ρ^[4*(x-x')])
The δ prior distribution density is evaluated by MVN(0,Σ)
"""
function log_prior_delta(data::DataStr,delta::Vector{Float64},
    rho::Vector{Float64},tau2::Float64,nx::Int64,nloc::Int64)

    mean = repeat([0],length(delta))  #mean of delta prior

    covar = tau2*correlation_construct(rho,data.exp.x,nx,nloc) #calc covar matrix

    log_pdf_val = logpdf(MvNormal(mean,covar),delta)      #calculate pdf val
    return log_pdf_val
end

"""
    log_prior_delta(delta::Vector{Float64},corr::Array{Float64,2},tau2::Float64)
Function to evaluate the probability density function of the discrepancy term (δ) prior distribution.
This implementation is for a case where the correlation matrix is already calculated.

---
Positional arguments
* `delta::Vector{Float64}` The value of delta at which to evaluate the distribution.
* `corr::Array{Float64,2}` Discrepancy function correlation matrix.
* `tau2::Float64` The variance of the discrepancy function's covariance matrix.

---
Returns
* `log_pdf_val` The log evaluation of the discrepancy term (δ) prior distribution density function.

---
Details
The discrepancy term (δ) assumes a zero mean prior distribution with a covariance structure defined by Σ=τ^2*C.
C is the correlation structure. It is a size n square matrix with each element calculated by the following:
C(x,x') = ∏(ρ^[4*(x-x')])
The δ prior distribution density is evaluated by MVN(0,Σ)
"""
function log_prior_delta(delta::Vector{Float64},
    corr::Array{Float64,2},tau2::Float64)

    mean = repeat([0],length(delta))  #mean of delta prior

    covar = tau2*corr #calc covar matrix

    log_pdf_val = logpdf(MvNormal(mean,covar),delta)      #calculate pdf val
    return log_pdf_val
end

"""
    prior_rho(rho::Float64,prior_data::PriorData,k::Int64)
Function to evaluate the prior distribution density of the discrepancy term correlation parameters (ρ).

---
Positional arguments
* `rho::Float64` The value of the correlation parameter to be evaluated in the prior distribution density funciton.
* `prior_data::PriorData` Data structure containing prior distribution hyperparameters for all variables.
* `k::Int64` An integer to specify which x variable's correlation parameter is being evaluated.

---
Returns
* `pdf_val` A scalar Float containing the evaluation of the prior distribution density function for the `k`th ρ.

---
Details
This function uses a beta distribution prior for ρ, using the parameters, a and b, already specified in `prior_data`.
This function evaluates the pdf of Beta(a,b) at ρ
"""
#### rho prior pdf function
# inputs: rhos, prior data, MCMC iteration vals
# returns: #pdf val of rho
function prior_rho(rho::Float64,prior_data::PriorData,k::Int64)
    a = prior_data.rho.par1[k] #pull prior hyperparams
    b = prior_data.rho.par2[k]
#    rho = vars.rho         #pull rho
    pdf_val = pdf(Beta(a,b),rho)   #calc pdf val
    return pdf_val
end
