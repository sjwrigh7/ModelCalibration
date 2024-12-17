#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
####################### Define Optimization Functions ########################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

"""
    max_lik_theta(ntheta::Int,model,data::DataStr,epochs::Int=7000)
Function to find the MLE of θ.

---
Positional arguments
* `ntheta::Int` The number of dimensions of θ.
* `model` Surrogate model.
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `epochs::Int` The number of optimization epochs. Default value of 7000.

---
Returns
* `theta_mle::Vector{Float64}` A Vector containing the MLE for θ.

---
Details
This function calls the `BlackBoxOptim.jl` differential evolutionary optimizer on `theta_opt`, and returns the best candidate from the optimization.
"""
function max_lik_theta(ntheta::Int,model,data::DataStr,epochs::Int=7000)
    #define objective function
    function theta_opt(theta::Vector{Float64})
        response = predict_y_all(theta,model)
        sse = sum((response .- data.exp.y).^2)
        return sse
    end
    #implement optimizer
    theta_vals = bboptimize(theta_opt; SearchRange = [(0.0,1.0) for i in 1:ntheta],
        NumDimensions=ntheta,MaxSteps=epochs)
    theta_mle = best_candidate(theta_vals)
    return theta_mle
end

"""
    max_lik_sigma(opt_response::Vector{Float64},epochs::Int=7000)
Function to get the maximum likelihood value for the data model error variance, σ.

---
Positional arguments
* `opt_response::Vector{Float64}` Vector containing the surrogate model prediction at the MLE of θ.
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `epochs::Int` Number of optimization epochs.

---
Returns
* `sig_mle::Float64` The maximum likelihood estimate the standard deviation, whose square when multiplied by the identity matrix, gives the covariance matrix for the data model.
"""
function max_lik_sigma(opt_response::Vector{Float64},data::DataStr,epochs::Int=7000)
    #define objective function
    function var_opt(sig::Vector{Float64})
        neg_lik = -logpdf(MvNormal(opt_response,sig[1]),data.exp.y)[1]
        return neg_lik
    end
    #implement optimizer
    sig_values = bboptimize(var_opt; SearchRange=(1e-10,100), NumDimensions=1, MaxSteps=epochs)
    sig_mle = best_candidate(sig_values)[1]
    return sig_mle
end

"""
    make_covar(sig2::Float64,nloc::Int)
Simple function to generate a covariance matrix.
This dispatch (passed only a variance parameter and size of the matrix) will use the σ^2I form.

---
Positional arguments
* `sig2::Float64` Variance parameter.
* `nloc::Int` Size of the square matrix.

---
Returns
* `covar::Array{Float64,2}` The covariance matrix.
"""
function make_covar(sig2::Float64,nloc::Int)
    covar = sig2 .* Matrix(1.0I,nloc,nloc)
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
* `covar::Array{Float64,2}` The covariance matrix.
"""
function make_covar(params::CovarPars,nloc::Int,nx::Int)
    rho_vec = repeat([params.rho],nx)
    corr_mat = correlation_construct(rho_vec,data.exp.x,nx,nloc)
    covar = params.sig2 .* Matrix(1.0I,nloc,nloc) .+ params.tau2*corr_mat
    return covar
end

"""
    max_lik_covar(nx::Int,nloc::Int,data::DataStr,epochs::Int=7000)
Function to get the maximum likelihood values of the hyperparameters for the data model covariance matrix. For a data model case where the discrepancy term (δ) is integrated out.

---
Positional arguments
* `nx::Int` The number of x variables.
* `nloc::Int` The number of locations at which the MVN data model is observed.
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `epochs::Int` The number of optimization epochs.

---
Returns
* `covar_par_mle::Vector{Float64}` The MLE values of the covariance matrix hyperparameters.
"""
function max_lik_covar(nx::Int,nloc::Int,data::DataStr,epochs::Int=7000)
    #define objective function
    function covar_opt(covar_pars::Vector{Float64})
        sig2 = covar_pars[1]
        tau2 = covar_pars[2]
        rho = covar_pars[3]
        params = CovarPars(sig2=sig2,rho=rho,tau2=tau2)
        covar = make_covar(params,nloc,nx)
        neg_lik = -logpdf(MvNormal(opt_response,covar),data.exp.y)
        return neg_lik
    end
    #implement optimizer
    covar_par_vals = bboptimize(covar_opt; SearchRange=[(1e-10,100),
        (1e-10,100),(0.001,0.999)], NumDimensions=3, MaxSteps=epochs)
    covar_par_mle = best_candidate(covar_par_vals)
    return covar_par_mle
end

"""
    get_mle(data::DataStr,nx::Int,nloc::Int,ntheta::Int,model;epochs::Int=7000)
Wrapper function to find the MLE of θ and σ^2.

---
Positional arguments
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `nx::Int` The number of control independent variables.
* `nloc::Int` The number of locations at which the MVN data model is observed.
* `ntheta::Int` The number of unknown independent variables.
* `model` Surrogate model.

Keyword arguments
* `epochs::Int` The number of epochs to run for the opimization.
  * default value of 7000.

---
Returns
* `theta_mle::Vector{Float64}` MLE for θ.
* `covar::Array{Float64,2}` MLE of the variance parameter for an identity-based covariance matrix.
"""
function get_mle(data::DataStr,nx::Int,nloc::Int,ntheta::Int,model;epochs::Int=7000)
    theta_mle = max_lik_theta(ntheta,model,data,epochs)
    opt_response = predict_y_all(theta_mle,model)
    var_mle = max_lik_sigma(opt_response,data,epochs)^2
    covar = make_covar(var_mle,nloc)
    return theta_mle,covar
end


