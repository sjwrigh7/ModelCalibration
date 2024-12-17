#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
######################### Define Likelihood Functions #############################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#


"""
    lik(data::DataStr,vars::UpdatedVars,thetas::Vector{Float64})
Function to evaluate the data model's likelihood function given the experimental data, surrogate model, and values for each variable in the data model.

---
Keyword arguments
* `data::DataStr` A data structure containing the computer simulator and experimental data.
* `vars::UpdatedVars` A data structure containing the most recently sampled values for each variable in the model.

---
Returns
* `likelihood` A scalar Float64 value of the likelihood function given the data and variable values.

---
Details
The likelihood function is evaluated as the product of probability density function of the data model over each independent observation.
This likelihood calculation assumes a multivariate normal distribution (having length n) with m independent observations.
Let η be the surrogate model's prediction of the response variabeles, y, for the given θ values specified in `theta`.
The likelihood is calculated following:
L(θ,τ^2,δ|y)=∏^m [MVN(η+δ,τ^2*I)]
"""
function lik(data::DataStr,vars::UpdatedVars,model)

    eta = predict_y_all(vars.theta,model) #surogate modle est

    delta = vars.delta                                #pull disc fun
    mean = eta + delta                                #model est
    response = data.exp.y                             #repsonse data
    sz = size(eta)[1]

    sig2 = vars.sig2[1,1]                             #pull tau^2
    covariance = sig2*Matrix(1.0I,sz,sz)              #calculate covar matrix
    covariance = covariance + Matrix(sqrt(eps(Float64))I,sz,sz)
    covariance = 0.5*(covariance' + covariance)    #ensure symmetry for stability

    likelihood = prod(pdf(MvNormal(mean,covariance),response)) #likelihood

    return likelihood
end

"""
    loglik(data::DataStr,vars::UpdatedVars)
Function to calculate the log of the likelihood function.

---
Keyword arguments
* `data::DataStr` Struct containing experimental and computer simulator data.
* `vars::UpdatedVars` Struct containing the values of the variables to use for the likelihood calculation.

---
Returns
* `log_likelihood::Float64` The log value of the likelihood function evaluated at the values specified in `vars`.

---
Details
Similar functionality to the non-log likelihood function.
This function uses the log of the pdf of the data model.
L(θ,τ^2,δ|y)=Σ^m [log(MVN(η+δ,τ^2*I))]
"""
function loglik(data::DataStr,theta::Vector{Float64},
    delta::Vector{Float64},sig2::Float64,model)

    eta = predict_y_all(theta,model) #surogate modle est

    mean = eta + delta                                #model est
    response = data.exp.y                             #repsonse data
    sz = size(eta)[1]

    covariance = sig2*Matrix(1.0I,sz,sz)              #calculate covar matrix
    covariance = covariance + Matrix(sqrt(eps(Float64))I,sz,sz)
    covariance = 0.5*(covariance' + covariance)    #ensure symmetry for stability

    log_likelihood = sum(logpdf(MvNormal(mean,covariance),response)) #likelihood

    return log_likelihood
end

function loglik(data::DataStr,delta::Vector{Float64},sig2::Float64,
    eta::Vector{Float64},nloc::Int)

    mean = eta + delta                                #model est
    response = data.exp.y                             #repsonse data

    ident = Matrix(1.0I,nloc,nloc)
    covariance = sig2 .* ident              #calculate covar matrix
    covariance = covariance + sqrt(eps(Float64)) .* ident
    covariance = 0.5*(covariance' + covariance)    #ensure symmetry for stability

    log_likelihood = sum(logpdf(MvNormal(mean,covariance),response)) #likelihood

    return log_likelihood
end

"""
    lik_nox(data::DataStr,vars::UpdatedVars,thetas::Vector{Float64})
A special implementation of the likelihood function for a case where there are no control variables (x) in the model i.e. the data model is a univariate normal distribution.
The univariate normal distribution data model allows for improved computational efficiency in this function compared to the general likelihood function implementation.

---
Keyword arguments
* `data::DataStr` A data structure containing the computer simulator and experimental data.
* `vars::UpdatedVars` A data structure containing the most recently sampled values for each variable in the model.


---
Returns
* `likelihood` A scalar Float64 value of the likelihood function given the data and variable values

---
Details
The likelihood function is evaluated as the product of probability density function of the data model over each independent observation.
This likelihood calculation assumes a univariate normal distribution with m independent observations.
Let η be the surrogate model's prediction of the response variabeles, y, for the given θ values specified in `theta`.
The likelihood is calculated following:
L(θ,τ^2|y)=∏^m [N(η,τ)]
"""
function lik_nox(data::DataStr,theta::Vector{Float64},sig2::Float64,model)
    eta = predict_y_all(theta,model)

    mean = eta

    response = data.exp.y

    likelihood = prod(pdf.(Normal(mean,sqrt(sig2)),response))

    return likelihood
end

function lik_nox(data::DataStr,eta::Vector{Float64},sig2::Float64)
    mean = eta

    response = data.exp.y

    likelihood = prod(pdf.(Normal(mean,sqrt(sig2)),response))

    return likelihood
end

"""
    loglik_nox(data::DataStr,vars::UpdatedVars,thetas::Vector{Float64})
A special implementation of the likelihood function for a case where there are no control variables (x) in the model i.e. the data model is a univariate normal distribution.
The univariate normal distribution data model allows for improved computational efficiency in this function compared to the general likelihood function implementation.

---
---
Keyword arguments
* `data::DataStr` A data structure containing the computer simulator and experimental data.
* `vars::UpdatedVars` A data structure containing the most recently sampled values for each variable in the model.

---
Returns
* `log_likelihood` A scalar Float64 value of the likelihood function given the data and variable values

---
Details
The likelihood function is evaluated as the product of probability density function of the data model over each independent observation.
This likelihood calculation assumes a univariate normal distribution with m independent observations.
Let η be the surrogate model's prediction of the response variabeles, y, for the given θ values specified in `theta`.
The likelihood is calculated following:
L(θ,τ^2|y)=Σ^m [log(N(η,τ))]
"""
function loglik_nox(data::DataStr,theta::Vector{Float64},sig2::Float64,model)
    eta = predict_y_all(theta,model)[1]

    mean = eta

    response = data.exp.y

    likelihood = sum(logpdf.(Normal(mean,sqrt(sig2)),response))

    return likelihood
end

function loglik_nox(data::DataStr,eta::Float64,sig2::Float64)
    mean = eta

    response = data.exp.y

    likelihood = sum(logpdf.(Normal(mean,sqrt(sig2)),response))

    return likelihood
end
