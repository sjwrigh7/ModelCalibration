#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########################### Define Posterior Functions ############################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
"""
    expit(c1::Float64,c2::Float64,gamma::Float64)
Function to transform the variable of γ ∈ in [-∞,∞] to [c2,c2+c1].

---
Keyword arguments
* `c1::Float64` Scale parameter of original variable space. If the variable is defined in [A,B], this parameter is B-A.
* `c2::Float64` Minimum parameter of the original variable space. If the variable is defined in [A,B], this parameter is A.
* `gamma::Float64` Value to be converted to the [A,B] interval.

---
Returns
* `x` A scalar Float transformed into [A,B] using c1*exp(γ)/(1+exp(γ)-c2)
"""
function expit(c1::Float64,c2::Float64,gamma::Float64)
    x = c1*(exp.(gamma))/(1+exp.(gamma)-c2) #transformed variable
    return x
end

"""
    logit(c1::Float64,c2::Float64,var::Float64)
Function to transform a variable defined in [c2,c1+c2] to γ space defined in [-∞,∞].

---
Keyword arguments
* `c1::Float64` Scale parameter of original variable space. If the variable is defined in [A,B], this parameter is B-A.
* `c2::Float64` Minimum parameter of the original variable space. If the variable is defined in [A,B], this parameter is A.
* `var::Float64` Variable to be converted to the [-∞,∞] interval.

---
Returns
* `gamma` A scalar Float transformed into [-∞,∞] by scaling `x` to [0,1] with x'=(x-c2)/c1 and then using the logit function, ln(x'/(1-x'))

"""
function logit(c1::Float64,c2::Float64,x::Float64)
    x_scale = (x-c2)/c1
    gamma = log.((x_scale)/(1-x_scale))
    return gamma #pribability transformation
end

"""
    metropolis_theta(prior_data::PriorData,data::DataStr,vars::UpdatedVars,k::Int64,stepsize::Float64)
Function to draw a sample of θ's posterior distribution using the Metropolis-Hastings algorithm.

---
Keyword arguments
* `prior_data::PriorData` Data structure containing the prior distribution parameters.
* `data::DataStr` Data structure containing the computer simulator and experimental data.
* `vars::UpdatedVars` Data structure containing the values of parameters to use for sampling θ.
* `k::Int64` Indexing integer prescribing which θ dimension is being sampled.
* `stepsize::Float64` Prescribed stepsize to use in the proposal distribution.

---
Returns
* `output::MetropolisInfo` A data struct containing the value sample from the Metropolis-Hastings algorithm, the acceptance probability for the proposed value, and the Boolean value indicating if the proposed value was accepted or rejected.

---
Details
The current value of θ[k] is transformed to the real number line using the `expit` function.
A new value is proposed on the real number line from a normal distribution using this transformed value as the mean and a standard deviation equal to `stepsize`.
The proposed value is transformed back to the θ space, θ*.
The proportional posterior probabilities are calculated for the current value of θ and the proposed value, θ*. These are denoted as π(θ) and π(θ*).
The proposal probabilities are calculated using θ and θ*, denoted as J(θ) and J(θ*).
The acceptance probability of θ* is calculated as π(θ*)/π(θ)*J(θ*)/J(θ).
A random sample form U(0,1) determines if the proposed value is accepted or rejected.
"""
function metropolis_theta(model,prior_data::PriorData,data::DataStr,
    theta::Vector{Float64},delta::Vector{Float64},sig2::Float64,
    k::Int64,stepsize::Float64,nloc::Int)
    c1 = 1.0              #scale of theta
    c2 = 0.0              #min value for theta

    current_theta = theta[k]        #pull current theta
    current_gamma = logit(c1,c2,current_theta) #transform to gamma
    prop_gamma = rand(Normal(current_gamma,stepsize)) #propose new gamma
    prop_theta = expit(c1,c2,prop_gamma)   #transform back to theta

    eta_current = predict_y_all(theta,model)
    log_lik_current = loglik(data,delta,sig2,eta_current,nloc)[1]
    theta[k] = prop_theta     #replace theta at index k with prop val
    eta_prop = predict_y_all(theta,model)
    log_lik_prop = loglik(data,delta,sig2,eta_prop,nloc)[1] #calc likelihood
    #calculate jump distribution values
    log_jump_current = logpdf(Normal(current_gamma,stepsize),prop_gamma)+
    log(abs(-(c1)/((prop_theta+c2)*(-c1+prop_theta+c2))))
    log_jump_propose = logpdf(Normal(prop_gamma,stepsize),current_gamma)+
    log(abs(-(c1)/((current_theta+c2)*(-c1+current_theta+c2))))

    #calculate acceptance
    log_lik_ratio = log_lik_prop - log_lik_current
    log_jump_ratio = log_jump_propose - log_jump_current
    ratio = min(exp(log_lik_ratio + log_jump_ratio), 1)
    accept = rand(Uniform(0,1))<ratio

    new_value = ifelse(accept,prop_theta,current_theta) #determine acceptance
    eta = ifelse(accept,eta_prop,eta_current)
    theta[k] = new_value
    output = MetropolisInfo(new_value,ratio,accept)
    return output,eta
end

"""
    metropolis_theta_nox(prior_data::PriorData,data::DataStr,vars::UpdatedVars,k::Int64,stepsize::Float64)
A special implementation of `metropolis_theta` for a case where there are no control independent (x) variables in the experimental data, making the data model a univariate normal distribution.
This function takes advantage of this by calling the special implementation of the likelihood function for greater computational efficiency. All other aspects of the function are identical to the general case.

---
Keyword arguments
* `prior_data::PriorData` Data structure containing the prior distribution parameters.
* `data::DataStr` Data structure containing the computer simulator and experimental data.
* `vars::UpdatedVars` Data structure containing the values of parameters to use for sampling θ.
* `k::Int64` Indexing integer prescribing which θ dimension is being sampled.
* `stepsize::Float64` Prescribed stepsize to use in the proposal distribution.

---
Returns
* `output::MetropolisInfo` A data struct containing the value sample from the Metropolis-Hastings algorithm, the acceptance probability for the proposed value, and the Boolean value indicating if the proposed value was accepted or rejected.

---
Details
The current value of θ[k] is transformed to the real number line using the `expit` function.
A new value is proposed on the real number line from a normal distribution using this transformed value as the mean and a standard deviation equal to `stepsize`.
The proposed value is transformed back to the θ space, θ*.
The proportional posterior probabilities are calculated for the current value of θ and the proposed value, θ*. These are denoted as π(θ) and π(θ*).
The proposal probabilities are calculated using θ and θ*, denoted as J(θ) and J(θ*).
The acceptance probability of θ* is calculated as π(θ*)/π(θ)*J(θ*)/J(θ).
A random sample form U(0,1) determines if the proposed value is accepted or rejected.
"""
function metropolis_theta_nox(model,prior_data::PriorData,data::DataStr,
    theta::Vector{Float64},delta::Vector{Float64},sig2::Float64,
    k::Int64,stepsize::Float64)
    c1 = 1.0              #scale of theta
    c2 = 0.0              #min value for theta

    current_theta = theta[k]        #pull current theta
    current_gamma = logit(c1,c2,current_theta) #transform to gamma
    prop_gamma = rand(Normal(current_gamma,stepsize)) #propose new gamma
    prop_theta = expit(c1,c2,prop_gamma)   #transform back to theta

    log_lik_current = loglik_nox(data,theta,delta,sig2,model)[1]
    theta[k] = prop_theta     #replace theta at index k with prop val

    log_lik_prop = loglik_nox(data,theta,delta,sig2,model)[1] #calc likelihood

    #calculate jump distribution values
    log_jump_current = logpdf(Normal(current_gamma,stepsize),prop_gamma)+
    log(abs(-(c1)/((prop_theta+c2)*(-c1+prop_theta+c2))))
    log_jump_propose = logpdf(Normal(prop_gamma,stepsize),current_gamma)+
    log(abs(-(c1)/((current_theta+c2)*(-c1+current_theta+c2))))
    #calculate acceptance
    log_lik_ratio = log_lik_prop - log_lik_current
    log_jump_ratio = log_jump_propose - log_jump_current
    ratio = min(exp(log_lik_ratio + log_jump_ratio), 1)
    accept = rand(Uniform(0,1))<ratio

    new_value = ifelse(accept,prop_theta,current_theta) #determine acceptance
    theta[k] = new_value
    output = MetropolisInfo(new_value,ratio,accept)
    return output
end

"""
    metropolis_rho(prior_data::PriorData,data::DataStr,vars::UpdatedVars,k::Int64,stepsize::Float64,nx::Int64,nobs::Int64)
Function to draw a sample of ρ's posterior distribution using the Metropolis-Hastings algorithm.

---
Keyword arguments
* `prior_data::PriorData` Data structure containing the prior distribution parameters.
* `data::DataStr` Data structure containing the computer simulator and experimental data.
* `vars::UpdatedVars` Data structure containing the values of parameters to use for sampling ρ.
* `k::Int64` Indexing integer prescribing which ρ dimension is being sampled.
* `stepsize::Float64` Prescribed stepsize to use in the proposal distribution.
* `nx::Int64` Integer describing the number of independent control variables (x).
* `nobs::Int64` Integer describing the number of data points in a single independent observation of the data model's multivariate normal distribution.

---
Returns
* `output::MetropolisInfo` A data struct containing the value sample from the Metropolis-Hastings algorithm, the acceptance probability for the proposed value, and the Boolean value indicating if the proposed value was accepted or rejected.

---
Details
The current value of ρ[k] is transformed to the real number line using the `expit` function.
A new value is proposed on the real number line from a normal distribution using this transformed value as the mean and a standard deviation equal to `stepsize`.
The proposed value is transformed back to the ρ space, ρ*.
The proportional posterior values are calculated for the current value of ρ and the proposed value, ρ*. These are denoted as π(ρ) and π(ρ*).
The proposal probabilities are calculated using ρ and ρ*, denoted as J(ρ) and J(ρ*).
The acceptance probability of ρ* is calculated as π(ρ*)/π(ρ)*J(ρ*)/J(ρ).
A random sample form U(0,1) determines if the proposed value is accepted or rejected.
"""
function metropolis_rho(prior_data::PriorData,data::DataStr,
    rho::Vector{Float64},delta::Vector{Float64},tau2::Float64,
    k::Int64,stepsize::Float64,nx::Int64,nobs::Int64)

    c1 = 1.0              #rho scale
    c2 = 0.0              #rho min val
    current_rho = rho[k]     #pull current rho at index k
    current_gamma = logit(c1,c2,current_rho) #transform to gamma
    prop_gamma = rand(Normal(current_gamma,stepsize)) #propose new gamma
    prop_rho = expit(c1,c2,prop_gamma)           #transform back to rho

    corr_current = correlation_construct(rho,data.exp.x,nx,nobs)
    log_delta_prop = log_prior_delta(delta,corr_current,tau2)[1]
    rho[k] = prop_rho               #replace rho at k with prop val
    corr_prop = correlation_construct(rho,data.exp.x,nx,nobs)
    log_delta_current = log_prior_delta(delta,corr_prop,tau2)[1]

    #calc jump distribution pdf
    log_jump_current = logpdf(Normal(current_gamma,stepsize),prop_gamma)+
    log(abs(-(c1)/((prop_rho+c2)*(-c1+prop_rho+c2))))
    log_jump_propose = logpdf(Normal(prop_gamma,stepsize),current_gamma)+
    log(abs(-(c1)/((current_rho+c2)*(-c1+current_rho+c2))))

    #calculate rho prior pdf
    prior_current = prior_rho(current_rho,prior_data,k)
    prior_prop = prior_rho(prop_rho,prior_data,k)

    #calculate acceptance
    log_delta_ratio = log_delta_prop - log_delta_current
    log_jump_ratio = log_jump_propose - log_jump_current
    ratio = min(exp(log_delta_ratio + log_jump_ratio), 1)
    accept = rand(Uniform(0,1))<ratio
    #println("accept = $accept")
    new_value = ifelse(accept,prop_rho,current_rho)  #determine acceptance
    corr = ifelse(accept,corr_prop,corr_current)
    output = MetropolisInfo(new_value,ratio,accept)
    return output,corr
end

"""
    gibbs:tau2(prior_data::PriorData,data::DataStr,vars::UpdatedVars,nx::Int64,nobs::Int64)
Function to draw a sample from the posterior distribution of the discrepancy variance term (σ^2).

---
Keyword arguments
* `prior_data::PriorData` Data structure containing the prior distribution parameters.
* `data::DataStr` Data structure containing the computer simulator and experimental data.
* `vars::UpdatedVars` Data structure containing the values of parameters to use for sampling ρ.
* `nx::Int64` Integer describing the number of independent control variables (x).
* `nobs::Int64` Integer describing the number of data points in a single independent observation of the data model's multivariate normal distribution.

---
Returns
* `sample` A scalar Float containing the new sample of σ^2 from its posterior distribution.

---
Details
The full conditional distribution of σ^2 is proportional to a multivariate normal distribution (the discrepancy term's prior distribution) multiplied by an inverse gamma distribution (the σ^2 prior distribution).
This can be simplified to a well known inverse gamma posterior distribution form for Gibbs updates.
Given prior distribution parameters for σ^2 of IG(α,β) and prior distribution for δ of MVN(0,σ^2*C), the posterior distribution is solved as the following:
p(σ^2|.) ∼ IG(α+n/2,β+0.5*δ'*C^-1*δ)
Where n is the length of the discrepancy term (the same length as the data model's multivariate normal distribution).
"""
function gibbs_tau2(prior_data::PriorData,
    delta::Vector{Float64},corr::Array{Float64,2},nloc::Int64)
    
    par1 = prior_data.tau2.par1 + 0.5*nloc   #calculate posterior params
    par2 = prior_data.tau2.par2 + 0.5*delta'*inv(corr)*delta

    sample = rand(InverseGamma(par1,par2))
    return sample   #sample from posterior
end

"""
    gibbs_sig2(prior_data::PriorData,data::DataStr,vars::UpdatedVars)
Function to draw a sample from the posterior distribution of the data model variance term (τ^2).

---
Keyword arguments
* `prior_data::PriorData` Data structure containing the prior distribution parameters.
* `data::DataStr` Data structure containing the computer simulator and experimental data.
* `vars::UpdatedVars` Data structure containing the values of parameters to use for sampling ρ.

---
Returns
* `sample` A scalar float containing the new sample of τ^2 from its posterior distribution.

---
Details
The full conditional distribution of τ^2 is proportional to a multivariate normal likelihood function multiplied by an inverse gamma distribution (the τ^2 prior distribution).
This can be simplified to a well known inverse gamma posterior distribution form for Gibbs updates.
Given prior distribution parameters for τ^2 of IG(α,β) and likelihood function of ∏MVN(η+δ,τ^2*I), the posterior distribution is solved as the following:
p(τ^2|.) ∼ IG(α+n*m/2,β+0.5*λ'*λ)
Where n is the length of the multivariate normal distribution in the data model, m is the number of independent observations from this data model, and λ is the vector y_i - η - δ.
"""
function gibbs_sig2(prior_data::PriorData,data::DataStr,
    eta::Vector{Float64},delta::Vector{Float64},
    lik_power::Int,nrep::Int)      

    par1 = prior_data.sig2.par1 + 0.5*lik_power  #calcualte posterior params

    sse = Vector{Float64}(undef,nrep)
    for i in axes(data.exp.y)[2]
        sse[i] = (data.exp.y[:,i] - eta - delta)'*(data.exp.y[:,i] - eta - delta)
    end

    par2 = prior_data.sig2.par2 + 0.5*sum(sse)

    sample = rand(InverseGamma(par1,par2))
    return sample  #sample from posterior
end

"""
    gibbs_delta(data::DataStr,vars::UpdatedVars)
Function to draw a sample from the posterior distribution of the discrepancy term, δ.

---
Keyword arguments
* `data::DataStr` Data structure containing the computer simulator and experimental data.
* `vars::UpdatedVars` Data structure containing the values of the variables to use for drawing the new sample of δ.

---
Returns
* `sample` A Vector of length n containing the posterior draw of δ.

---
Details
The full conditional distribution of δ is proportional to the multivariate normal likelihood function multiplied by the multivarite normal prior of δ.
This can be simplified to a well known multivariate normal posterior distribution.
Given p(δ) = MVN(0,Σ) and L(.|x,y) = ∏MVN(η+δ,τ^2*I), the posterior distribution for δ can be solved as the following:
p(δ|.) ∼ MVN(bn*An,An)
where
An = (Σ^-1 + m^2(τ^2*I)^-1)^-1
bn = m^2((τ^2*I)^-1)*(ȳ-η)
m is the number of independent observations of the multivariate normal data.
"""
function gibbs_delta(data::DataStr,sig2::Float64,tau2::Float64,
    eta::Vector{Float64},corr::Array{Float64,2},nloc::Int,nrep::Int)
    #calc covar matrix
    sig = tau2*corr

    #calculate values for posterior
    An = inv(sig) + nrep*(1/(sig2)) .* Matrix(1.0I,nloc,nloc)

    covar = inv(An)
    covar = 0.5*(covar + covar') #ensures symmetry for stability
    bn_vec = Array{Float64}(undef,nloc,nrep)
    for i in 1:nrep
        bn_vec[:,i] = (data.exp.y[:,i]-eta)
    end
    bn = nrep*1/sig2*mean(bn_vec,dims=2)
    sample = rand(MvNormal(vec(covar*bn),covar))
    return sample  #sample from posterior
end
