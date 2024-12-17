#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
######################### Define Likelihood Functions #############################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#


"""
    lik(data::DataStr,theta::Vector{Float64},delta::Vector{Float64},sig2::Float64,model)
Function to evaluate the data model's likelihood function.
Implementation for a multivariate normal distribution data model where the surrogate model's estimate is not already known.

---
Positional arguments
* `data::DataStr` A data structure containing the computer simulator and experimental data.
* `theta::Vector{Float64}` Vector of theta sample values.
* `delta::Vector{Float64}` Vector of discrepancy function sample.
* `sig2::Float64` Data model error variance sample.
* `model` Surrogate model.

---
Returns
* `likelihood` A scalar Float64 value of the likelihood function.

---
Details
The likelihood function is evaluated as the product of probability density function of the data model over each independent observation.
This likelihood calculation assumes a multivariate normal distribution (having length n) with m independent observations.
Let η be the surrogate model's prediction of the response variabeles, y, for the given θ values specified in `theta`.
The likelihood is calculated following:
L(θ,σ^2,δ|y)=∏^m [MVN(η+δ,σ^2*I)]
In this implementation, the surrogate model prediction is calculated in this function.
"""
function lik(data::DataStr,theta::Vector{Float64},
            delta::Vector{Float64},sig2::Float64,model)

    eta = predict_y_all(theta,model) #surogate modle estimate

    mean = eta + delta                                #data model mean estimate
    response = data.exp.y                             #repsonse data

    likelihood = prod(pdf(MvNormal(mean,sqrt(sig2)),response)) #likelihood

    return likelihood
end

"""
    lik(data::DataStr,delta::Vector{Float64},sig2::Float64,eta::Vector{Float64})
Function to evaluate the data model's likelihood function.
Implementation for a multivariate normal distribution data model where the surrogate model's estimate is already known.

---
Positional arguments
* `data::DataStr` A data structure containing the computer simulator and experimental data.
* `delta::Vector{Float64}` Vector of discrepancy function sample.
* `sig2::Float64` Data model error variance sample.
* `eta::Vector{Float64}` Surrogate model prediction

---
Returns
* `likelihood` A scalar Float64 value of the likelihood function.

---
Details
The likelihood function is evaluated as the product of probability density function of the data model over each independent observation.
This likelihood calculation assumes a multivariate normal distribution (having length n) with m independent observations.
Let η be the surrogate model's prediction of the response variabeles, y, for the given θ values specified in `theta`.
The likelihood is calculated following:
L(θ,σ^2,δ|y)=∏^m [MVN(η+δ,σ^2*I)]
"""
function lik(data::DataStr,delta::Vector{Float64},sig2::Float64,
            eta::Vector{Float64})

    mean = eta + delta                                #data model mean estimate
    response = data.exp.y                             #repsonse data

    likelihood = prod(pdf(MvNormal(mean,sqrt(sig2)),response)) #likelihood

    return likelihood
end

"""
    loglik(data::DataStr,theta::Vector{Float64},delta::Vector{Float64},sig2::Float64,model)
Function to evaluate the data model's log likelihood function.
Implementation for a multivariate normal distribution data model where the surrogate model's estimate is not already known.

---
Positional arguments
* `data::DataStr` A data structure containing the computer simulator and experimental data.
* `theta::Vector{Float64}` Vector of theta sample values.
* `delta::Vector{Float64}` Vector of discrepancy function sample.
* `sig2::Float64` Data model error variance sample.
* `model` Surrogate model.

---
Returns
* `log_likelihood` A scalar Float64 value of the likelihood function.

---
Details
The likelihood function is evaluated as the product of probability density function of the data model over each independent observation.
This likelihood calculation assumes a multivariate normal distribution (having length n) with m independent observations.
Let η be the surrogate model's prediction of the response variabeles, y, for the given θ values specified in `theta`.
The log likelihood is calculated following:
l(θ,σ^2,δ|y)=Σ^m [log(MVN(η+δ,τ^2*I))]
In this implementation, the surrogate model prediction is calculated in this function.
"""
function loglik(data::DataStr,theta::Vector{Float64},
            delta::Vector{Float64},sig2::Float64,model)

    eta = predict_y_all(theta,model) #surogate modle est

    mean = eta + delta                                #model est
    response = data.exp.y                             #repsonse data

    log_likelihood = sum(logpdf(MvNormal(mean,sqrt(sig2)),response)) #likelihood

    return log_likelihood
end

"""
    log_lik(data::DataStr,delta::Vector{Float64},sig2::Float64,eta::Vector{Float64})
Function to evaluate the data model's log likelihood function.
Implementation for a multivariate normal distribution data model where the surrogate model's estimate is already known.

---
Positional arguments
* `data::DataStr` A data structure containing the computer simulator and experimental data.
* `delta::Vector{Float64}` Vector of discrepancy function sample.
* `sig2::Float64` Data model error variance sample.
* `eta::Vector{Float64}` Surrogate model estimate.


---
Returns
* `log_likelihood` A scalar Float64 value of the likelihood function.

---
Details
The likelihood function is evaluated as the product of probability density function of the data model over each independent observation.
This likelihood calculation assumes a multivariate normal distribution (having length n) with m independent observations.
Let η be the surrogate model's prediction of the response variabeles, y, for the given θ values specified in `theta`.
The likelihood is calculated following:
l(θ,σ^2,δ|y)=Σ^m [log(MVN(η+δ,τ^2*I))]
"""
function loglik(data::DataStr,delta::Vector{Float64},sig2::Float64,
            eta::Vector{Float64})

    mean = eta + delta                                #model est
    response = data.exp.y                             #repsonse data

    log_likelihood = sum(logpdf(MvNormal(mean,sqrt(sig2)),response)) #likelihood

    return log_likelihood
end

"""
    lik(data::DataStr,theta::Vector{Float64},sig2::Float64,model)
Function to evaluate the data model's likelihood function.
Implementation for a univariate normal distribution data model where the surrogate model's estimate is not already known.

---
Positional arguments
* `data::DataStr` A data structure containing the computer simulator and experimental data.
* `theta::Vector{Float64}` Vector of theta sample values.
* `sig2::Float64` Data model error variance sample.
* `model` Surrogate model.

---
Returns
* `likelihood` A scalar Float64 value of the likelihood function.

---
Details
The likelihood function is evaluated as the product of probability density function of the data model over each independent observation.
This likelihood calculation assumes a univariate normal distribution with m independent observations.
Let η be the surrogate model's prediction of the response variabeles, y, for the given θ values specified in `theta`.
The likelihood is calculated following:
L(θ,σ^2|y)=∏^m [N(η,σ)]
In this implementation, the surrogate model prediction is calculated in this function.
"""
function lik(data::DataStr,theta::Vector{Float64},sig2::Float64,model)

    eta = predict_y_all(theta,model)

    mean = eta

    response = data.exp.y

    likelihood = prod(pdf.(Normal(mean,sqrt(sig2)),response))

    return likelihood
end

"""
    lik(data::DataStr,eta::Vector{Float64},sig2::Float64)
Function to evaluate the data model's likelihood function.
Implementation for a univariate normal distribution data model where the surrogate model's estimate is already known.

---
Positional arguments
* `data::DataStr` A data structure containing the computer simulator and experimental data.
* `eta::Vector{Float64}` Vector of theta sample values.
* `sig2::Float64` Data model error variance sample.

---
Returns
* `likelihood` A scalar Float64 value of the likelihood function.

---
Details
The likelihood function is evaluated as the product of probability density function of the data model over each independent observation.
This likelihood calculation assumes a univariate normal distribution with m independent observations.
Let η be the surrogate model's prediction of the response variabeles, y, for the given θ values specified in `theta`.
The likelihood is calculated following:
L(θ,σ^2|y)=∏^m [N(η,σ)]
"""
function lik(data::DataStr,eta::Vector{Float64},sig2::Float64)
    mean = eta

    response = data.exp.y

    likelihood = prod(pdf.(Normal(mean,sqrt(sig2)),response))

    return likelihood
end

"""
    loglik(data::DataStr,theta::Vector{Float64},sig2::Float64,model)
Function to evaluate the data model's log likelihood function.
Implementation for a univariate normal distribution data model where the surrogate model's estimate is not already known.

---
Positional arguments
* `data::DataStr` A data structure containing the computer simulator and experimental data.
* `theta::Vector{Float64}` Vector of theta sample values.
* `sig2::Float64` Data model error variance sample.
* `model` Surrogate model.

---
Returns
* `log_likelihood` A scalar Float64 value of the likelihood function.

---
Details
The likelihood function is evaluated as the product of probability density function of the data model over each independent observation.
This likelihood calculation assumes a univariate normal distribution with m independent observations.
Let η be the surrogate model's prediction of the response variabeles, y, for the given θ values specified in `theta`.
The log likelihood is calculated following:
l(θ,σ^2|y)=∏^m [log(N(η,σ))]
In this implementation, the surrogate model prediction is calculated in this function.
"""
function loglik(data::DataStr,theta::Vector{Float64},sig2::Float64,model)
    eta = predict_y_all(theta,model)[1]

    mean = eta

    response = data.exp.y

    log_likelihood = sum(logpdf.(Normal(mean,sqrt(sig2)),response))

    return log_likelihood
end

"""
    loglik(data::DataStr,theta::Vector{Float64},sig2::Float64,model)
Function to evaluate the data model's log likelihood function.
Implementation for a univariate normal distribution data model where the surrogate model's estimate is already known.

---
Positional arguments
* `data::DataStr` A data structure containing the computer simulator and experimental data.
* `theta::Vector{Float64}` Vector of theta sample values.
* `sig2::Float64` Data model error variance sample.
* `model` Surrogate model.

---
Returns
* `log_likelihood` A scalar Float64 value of the likelihood function.

---
Details
The likelihood function is evaluated as the product of probability density function of the data model over each independent observation.
This likelihood calculation assumes a univariate normal distribution with m independent observations.
Let η be the surrogate model's prediction of the response variabeles, y, for the given θ values specified in `theta`.
The log likelihood is calculated following:
l(θ,σ^2|y)=∏^m [log(N(η,σ))]
"""
function loglik(data::DataStr,eta::Float64,sig2::Float64)
    mean = eta

    response = data.exp.y

    log_likelihood = sum(logpdf.(Normal(mean,sqrt(sig2)),response))

    return log_likelihood
end
