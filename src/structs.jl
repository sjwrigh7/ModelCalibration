#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
############################# Define Data Structures ##############################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

"""
    ScalePar(min::Float64,max::Float64)
Structure containing the minimum and maximum vales for a variable.

---
Keyword arguments
* `min::Union{Float64,Vector{Float64}}`
* `max::Union{Float64,Vector{Float64}}`
"""
@with_kw struct ScalePar
    min::Union{Float64,Vector{Float64}}
    max::Union{Float64,Vector{Float64}}
end

"""
    GridPar(density::Int,bounds::ScalePar)
Structure containing the density and bounds information of a parameter for the Griddy Gibbs precomputation.

---
Keyword arguments
* `density::Int` Integer specifying the number of grid points for the parameter to use during precomputation.
* `bounds::ScalePar` Data structure specifying the lower and upper bounds of the parameter to use for precomputation.
"""
@with_kw mutable struct GridPar
    density::Int
    bounds::ScalePar
end

"""
    GridData(sig_star2::Int=100,rho::Int=100)
Structure containing the grid information for σ*^2 and ρ for the Griddy Gibbs precomputation.

---
Keyword arguments
* `sig_star2::GridPar` Structure specifying the bounds and number of grid points to evaluate for σ*^2 during precomputation.
* `rho::GridPar` Structure specifying the bounds and number of grid points to evaluate for ρ during precomputation. The bounds should be in (0,1).
"""
@with_kw mutable struct GridData
    sig_star2::GridPar
    rho::GridPar
end

"""
    Scaling(y::ScalePar,theta::ScalePar,x::ScalePar)
Structure containing scaling information for the response variable, y, and the unknown model parameters, θ.

---
Keyword arguments
* `y::ScalePar`
* `theta::ScalePar`
* `x::ScalePar`
"""
@with_kw struct Scaling
    y::ScalePar
    theta::ScalePar
    x::ScalePar
end

"""
    MetropolisInfo(new_value::Float64,ratio::Float64,accept::Bool)
Structure used for storing the acceptance information for a single step in the Metropolis-Hastings algorithm.

---
Keyword arguments
* `new_value::Float64` The new value, θ^(t+1), for the Metropolis-Hastings algorithm. Taken as either θ* or θ^(t) based on acceptance criteria of the proposed value, θ*.
* `ratio::Float64` The acceptance probability, α, for the proposed value, calculated from p(θ*)/p(θt).
* `accept::Bool` The logical value determining whether the proposed value was accepted or rejected.
"""
@with_kw struct MetropolisInfo
    new_value::Float64
    ratio::Float64
    accept::Bool
end

"""
    SimStr(x::Array{Float64},y::Vector{Float64},theta::Array{Float64})
Structure containing the data from the computer simulation, organized for building a surrogate model.

---
Keyword arguments
* `x::Array{Float64}` Array containing independent control variable settings for the simulation data, of dimensions n and nx.
* `y::Array{Float64}` Array containing the response variable outputs for the simulation data, of dimensions n and m.
* `theta::Array{Float64}` Array containing the independent unknown variables for the simulation data, of dimensions m and ntheta.
* `x_reps::Array{Float64}` Array containing independent control variable settings for the simulation data, of dimensions n*m and nx.
* `y_reps::Vector{Float64}` Vector containing the response variable outputs for the simulation data, of length n*m.
* `theta_reps::Array{Float64}` Array containing the independent unknown variables for the simulation data, of dimensions n*m and ntheta.
"""
@with_kw struct SimStr
    x::Array{Float64}
    y::Array{Float64}
    theta::Array{Float64}
    x_reps::Array{Float64}
    y_reps::Vector{Float64}
    theta_reps::Array{Float64}
end

"""
    ExpStr(x::Array{Float64},y::Array{Float64},x_reps::Array{Float64},y_reps::Vector{Float64})
Structure containing the experimental data to which the surrogate model is calibrated.

---
Keyword arguments
* `x::Array{Float64}` Array containing all unique settings of the independent control variables in the experimental data. Dimensions n and p where n is the number of unique settings and p is the number of x variables.
* `y::Array{Float64}` Array containing all repeated observations of the unique x settings in the experimental data. Dimensions n and m, where n is the number of unique x settings and m is the number of repeated observations.
* `x_reps::Array{Float64}` Array containing all settings of the independednt control variables in the experimental data *not* trimmed to each be unique. Dimensions n*m and p.
* `y_reps::Vector{Float64}` Vector containing all response observations in the experimental data *not* arranged based on unique locations, length n*m.
"""
@with_kw struct ExpStr
    x::Array{Float64}
    y::Array{Float64}
    x_reps::Array{Float64}
    y_reps::Array{Float64}
end

"""
    DataStr(sim::SimStr,exp::ExpStr)
Structure containing both the computer simulator and experimental data structures.

---
Keyword arguments
* `sim::SimStr` A data structure containing the computer simulator data
* `exp::ExpStr` A data structure containing the experimental data
"""
@with_kw struct DataStr
    sim::SimStr
    exp::ExpStr
end

"""
    PriorVar(par1::Union{Float64,Vector{Float64}},par2::Union{Float64,Vector{Float64}})
Structure containing prior distribution parameters. Currently, only two-parameter prior distributions are supported herein.
Parameters are intended to be specified in the typical ordering.
Ex: for a normal distribution, `par1` = μ, `par2` = σ
Ex: for a uniform distribution, `par1` = min, `par2` = max

---
Keyword arguments
* `par1::Union{Float64,Vector{Float64}}` The first parameter of the prior distribution
* `par2::Union{Float64,Vector{Float64}}` The second parameter of the prior distribution
"""
@with_kw struct PriorVar
    par1::Union{Float64,Vector{Float64}}
    par2::Union{Float64,Vector{Float64}}
end

"""
    PriorData(theta::PriorVar,sig2::PriorVar,tau2::PriorVar,rho::PriorVar)
Structure containing the `PriorVar` prior distribution parameters for all parameters in the Bayesian calibration model.

---
Keyword arguments
* `theta::PriorVar` Prior distribution parameters for the unknown model parameters, θ. The two prior parameters are m length Vectors containing minimum and maximum values corresponding to a uniform prior distribution for each of the m θ variables, respectively.
* `sig2::PriorVar` Prior distribution parameters for the data error variance. The two prior parameters are the shapre (α) and scale (β) of an inverse gamma distribution, respectively.
* `tau2::PriorVar` Prior distribution parameters for the discrepancy term variance. The two prior parameters are the shapre (α) and scale (β) of an inverse gamma distribution, respectively.
* `rho::PriorVar` Prior distribution parameters for the discrepancy term correlation structure. The two parameters are p length Vectors containing the α and β parameters for a Beta distribution for each of the p x variables, respectively.
"""
@with_kw struct PriorData
    theta::PriorVar
    sig2::PriorVar
    tau2::PriorVar
    rho::PriorVar
end

"""
    BulkVarsStruct(theta::Array{Float64},sig2::Vector{Float64},tau2::Vector{Float64},delta::Array{Float64},rho::Array{Float64},accept::Array{Bool},ratio::Array{Float64})
Data structure containing the samples and Metropolis-Hastings information from the Markov Chain Monte Carlo simulation sampling of the posterior distribution in the Bayesian model calibration.

---
Keyword arguments
* `theta::Array{Float64}` An Array to store the samples of θ during the MCMC simulation, dimensions n and m.
* `sig2::Vector{Float64}` A Vector to store the samples of τ^2 during the MCMC simulation, length n.
* `tau2::Vector{Float64}` A Vector to store the samples of σ^2 during the MCMC simulation, length n.
* `delta::Array{Float64}` An Array to store the samples of δ during the MCMC simulation, dimensions n and q.
* `rho::Array{Float64}` An Array to store the samples of ρ during the MCMC simulation, dimensions n and p.
* `accept::Array{Bool}` An Array to store the acceptance status of each Metropolis-Hastings update during the MCMC, dimensions n and p+m.
* `ratio::Array{Float64}` An Array to store the acceptance probability of each Metropolis-Hastings update during the MCMC, dimensions n and p+m.
"""
@with_kw mutable struct BulkVarsStruct
    theta::Array{Float64}
    sig2::Vector{Float64}
    tau2::Vector{Float64}
    delta::Array{Float64}
    eta::Array{Float64}
    rho::Array{Float64}
    accept::Array{Bool}
    ratio::Array{Float64}
end

"""
    GriddyVarsStruct(theta::Vector{Int},sig2::Vector{Float64},sig_star2::Vector{Float64},rho::Vector{Float64})
Data structure containing posterior index samples for the griddy Gibbs sampler.

---
Keyword arguments
* `theta::Vector{Int}` A Vector to store the index of the values of θ during the MCMC simulation, length n.
* `sig2::Vector{Float64}` A Vector to store the samples of τ^2 during the MCMC simulation, length n.
* `sig_star2::Vector{Int}` A Vector to store the samples of σ*^2 during the MCMC simulation, length n.
* `rho::Vector{Int}` A Vector to store the samples of ρ during the MCMC simulation, length n.
"""
@with_kw mutable struct GriddyVarsStruct
    theta::Vector{Int}
    sig2::Vector{Float64}
    sig_star2::Vector{Int}
    rho::Vector{Int}
end

"""
    GriddyPosteriors(theta::Array{Float64},sig2::Vector{Float64},sig_star2::Vector{Float64},rho::Vector{Float64})
Struct to store the posterior samples from the griddy Gibbs sampler, converted from the posterior indices.

---
Keyword arguments
* `theta::Array{Float64}` Array of unknown computer model parameter samples.
* `sig2::Vector{Float64}` Vector of data model variance parameter samples.
* `sig_star2::Vector{Float64}` Vector of discrepancy variance parameter samples.
* `rho::Vector{Float64}` Vector of discrepancy correlation parameter samples.
"""
@with_kw mutable struct GriddyPosteriors
    theta::Array{Float64}
    sig2::Vector{Float64}
    sig_star2::Vector{Float64}
    rho::Vector{Float64}
end

"""
    UpdatedVars(theta::Vector{Float64},delta::Vector{Float64},sig2::Float64,tau2::Float64,rho::Vector{Float64})
Data structure containing the most recently sampled values for each parameter during the Markov Chain Monte Carlo simulation sampling of the posterior distribution in the Bayesian model calibration.

---
Keyword arguments
* `theta::Vector{Float64}` A Vector of length m containing the most recently sampled values of each of the m θ parameters.
* `delta::Vector{Float64}` A Vector of length q containing the most recently sampled values of δ.
* `sig2::Float64` The most recently sampled value of τ^2.
* `tau2::Float64` The most recently sampled value of σ^2.
* `rho::Vector{Float64}` A Vector of length p containing the most recently sampled values of each of the p ρ parameters.
"""
@with_kw mutable struct UpdatedVars
    theta::Vector{Float64}
    delta::Vector{Float64}
    sig2::Float64
    tau2::Float64
    rho::Vector{Float64}
end

"""
    StepSize(theta::Vector{Float64},rho::Vector{Float64})
Data structure containing the calculated optimal step sizes for θ and ρ for the Metropolis-Hastings algorithm

---
Keyword arguments
* `theta::Vector{Float64}` A Vector of length m containin the optimal step sizes for each of the m θ for use in the Metropolis-Hastings algorithm.
* `rho::Vector{Float64}` A Vector of length p containing the optimal step sizes for each of the p ρ for use in the Metropolis-Hastings algorithm.
"""
@with_kw struct StepSize
    theta::Vector{Float64}
    rho::Vector{Float64}
end

"""
    CovarPars(ta2::Float64,rho::Float64,sig2::Float64)
Struct to store mutable params for covar construction.

---
Keyword arguments
* `sig2::Float64` Variance parameter for identity portion of covaraince.
* `rho::Float64` Correlation parameter.
* `tau2::Float64` Variance parameter for correlation portion of covariance.
"""
@with_kw mutable struct CovarPars
    sig2::Float64
    rho::Float64
    tau2::Float64
end
