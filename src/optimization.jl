#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
####################### Define Optimization Functions ########################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

"""
    theta_opt(theta:Vector{Float64})
Function to get the maximum likelihood estimate of θ.

---
Keyword arguments
* `theta::Vector{Float64}` Vector of theta values to optimize.


---
Returns
* `sse::Float64` A floating point describing the sum of squared error between the surrogate model and the experimental data.

---
Details
This function returns the SSE between the surrogate model and experimental data, for finding the maximum likelihood values of θ. Because the mean and covariance parameters are separable, minimizing the SSE will find the MLE for θ.
"""
function theta_opt(theta::Vector{Float64})
    response = predict_y_all(theta)
    sse = sum((response .- data.exp.y).^2)
    return sse
end

"""
    max_lik_theta(epochs::Int=7000,ntheta::Int)
Function to find the MLE of θ.

---
Keyword arguments
* `epochs::Int` The number of optimization epochs. Default value of 7000.
* `ntheta::Int` The number of dimensions of θ.

---
Returns
* `theta_mle::Vector{Float64}` A Vector containing the MLE for θ.

---
Details
This function calls the `BlackBoxOptim.jl` differential evolutionary optimizer on `theta_opt`, and returns the best candidate from the optimization.
"""
function max_lik_theta(epochs::Int=7000,ntheta)
    theta_vals = bboptimize(theta_opt; SearchRange = [(0.0,1.0) for i in 1:ntheta],NumDimensions=ntheta,MaxSteps=epochs)
    theta_mle = best_candidate(theta_vals)
    return theta_mle
end

"""
    var_opt(sig::Vector{Float64})
Function for optimizing the variance parameter of the data model, assuming an Identity-based covariance matrix.

---
Keyword arguments
* `sig::Vector{Float64}` The standard deviation to be used in the data model for likelihood calculations.

---
Returns
* `neg_lik::Float64` The negative log-likelihood of the data model, at the optimized values for θ, given the specified input for σ.

---
Details
This function utilizes the `Distributions.jl` framework for calculating the log likelihood. The API call used in this function uses the standard deviation for the MVN distribution when a scalar value is supplied for the covariance matrix.
"""
function var_opt(sig::Float64)
    neg_lik = -logpdf(MvNormal(opt_response,sig[1]),data.exp.y)[1]
    return neg_lik
end

"""
    max_lik_sigma(epochs::Int=7000)
Function to get the maximum likelihood value for σ.

---
Keyword arguments
* `epochs::Int` Number of optimization epochs.

---
Returns
* `sig_mle::Float64` The maximum likelihood estimate the standard deviation, whose square when multiplied by the identity matrix, gives the covariance matrix for the data model.
"""
function max_lik_sigma(epochs::Int=7000)
    sig_values = bboptimize(var_opt; SearchRange=[(1e-10,100)], NumDimensions=1, MaxSteps=epochs)
    sig_mle = best_candidate(sig_values)
    return sig_mle
end

"""
    make_covar(tau2::Float64,nobs::Int)
Simple function to generate a covariance matrix.
This dispatch (passed only a variance parameter and size of the matrix) will use the τ^2I form.

---
Keyword arguments
* `tau2::Float64` Variance parameter.
* `nobs::Int` Size of the square matrix.

---
Returns
* `covar::Array{Float64}` The covariance matrix.
"""
function make_covar(tau2::Float64,nobs::Int)
    covar = tau2 .* Matrix(1.0I,nobs,nobs)
    return covar
end

"""
    make_covar(params::CovarPars,nobs::Int)
Simple function to generate a covariance matrix.
This dispatch (passed `CovarPars` struct) will use a form with an integrated discrepancy.

---
Keyword arguments
* `params::CovarPars` Struct containing covariance matrix parameters for integrated discrepancy.
* `nobs::Int` Size of the square matrix.

---
Returns
* `covar::Array{Float64}` The covariance matrix.
"""
function make_covar(params::CovarPars,nobs::Int)
    rho_vec = repeat([params.rho],nx)
    corr_mat = correlation_construct(rho_vec,data.exp.x,nx,nobs)
    covar = params.tau2 .* Matrix(1.0I,nobs,nobs) .+ params.sig2*corr_mat
    return covar
end

"""
    covar_opt(covar:pars::Vector{Float64})
Function to optimize the hyperparameters for a data model whose that has the discrepancy term (δ) integrated out, into the covariance matrix.

---
Keyword arguments
* `covar_pars` A Vector of Floats containing the hyperparameters for the data model covariance matrix.

---
Returns
* `neg_lik::Float64` The negative of the log-likelihood of the data model.
"""
function covar_opt(covar_pars::Vector{Float64})
    tau2 = covar_pars[1]
    sig2 = covar_pars[2]
    rho = covar_pars[3]
    params = CovarPars(tau2=tau2,rho=rho,sig2=sig2)
    covar = make_covar(params,nobs)
    neg_lik = -logpdf(MvNormal(opt_response,covar),data.exp.y)
    return neg_lik
end

"""
    max_lik_covar(epochs::Int=7000)
Function to get the maximum likelihood values of the hyperparameters for the data model covariance matrix. For a data model case where the discrepancy term (δ) is integrated out.

---
Keyword arguments
* `epochs::Int` The number of optimization epochs.

---
Returns
* `covar_par_mle::Vector{Float64}` The MLE values of the covariance matrix hyperparameters.
"""
function max_lik_covar(epochs::Int=7000)
    covar_par_vals = bboptimize(covar_opt; SearchRange=[(1e-10,100),(1e-10,100),(0.001,0.999)], NumDimensions=3, MaxSteps=epochs)
    covar_par_mle = best_candidate(covar_par_vals)
    return covar_par_mle
end

"""
    get_mle(data::DataStr,nx::Int,ntheta::Int)
Function to find the MLE of θ and τ^2.

---
Keyword arguments
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `nx::Int` The number of control independent variables.
* `ntheta::Int` The number of unknown independent variables.

---
Returns
* `theta_mle::Vector{Float64}` MLE for θ.
* `var_mle::Float64` MLE of the variance parameter for an identity-based covariance matrix.
"""
function get_mle(data::DataStr,nx::Int,ntheta::Int)
    theta_mle = max_lik_theta(ntheta=ntheta)
    opt_response = predict_y_all(theta_mle)
    var_mle = max_lik_sig()^2
    covar = make_covar(var_mle,nobs)
    return theta_mle,covar
end


