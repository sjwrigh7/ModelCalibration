#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
###################### Define Griddy Posterior Functions ##########################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

"""
    loglik_theta!(c_sse::Vector{Float64},sig2_inv::Float64,loglik_theta::Vector{Float64})
Function to calculate the optimized log-likelihood for θ in the griddy Gibbs sampling method.

---
Keyword arguments
* `c_sse::Vector{Float64}` Correlation sum of of squared error. This is calculated by sum([y-μ]'Σ*^-1[y-μ]).
* `sig2_inv::Float64` Date model error variance.
* `loglik_theta::Vector{Float64}` Pre-allocated Vector to store the log likelihood values in.
"""
function loglik_theta!(c_sse::Array{Float64},sig2_inv::Float64,loglik_arr::Array{Float64})
    loglik_arr .= -0.5 .* sig2_inv .* c_sse
end

function loglik_theta(c_sse::Array{Float64},sig2_inv::Float64)
    return -0.5 .* sig2_inv .* c_sse
end

"""
    loglik_covar!(c_sse::Array{Float64},log_det_sig::Array{Float64},sig2_inv::Float64,loglik_theta::Array{Float64})
Function to calculate the optimized log-likelihood for Σ* in the griddy Gibbs sampling method.

---
Keyword arguments
* `c_sse::Array{Float64}` Correlation sum of of squared error. This is calculated by sum([y-μ]'Σ*^-1[y-μ]).
* `log_det_sig::Array{Float64}` Determinant of Σ* raised to the power of (-m/2), where mis the number of observations.
* `sig2_inv::Float64` Date model error variance.
* `loglik_theta::Array{Float64}` Pre-allocated Vector to store the log likelihood values in.
"""
function loglik_covar!(c_sse::Array{Float64},log_det_sig::Array{Float64},sig2_inv::Float64,loglik_covar::Array{Float64})
    loglik_covar .= log_det_sig .+ (-0.5 .* sig2_inv .* c_sse)
end

"""
    griddy_sample!(posterior_vals::Vector{Float64},stable_vals::Vector{Float64},norm_vals::Vector{Float64},cumsum_vals::Vector{Float64})
Function to draw a griddy Gibbs sample for a variable.

---
Keyword arguments
* `posterior_vals::Vector{Float64}` Vector of length n containing the proportional posterior values for a variable.
* `stable_vals::Vector{Float64}` Preallocated Vector of length n for storing the `posterior_vals`, numerically stablized by their maximum value.
* `norm_vals::Vector{Float64}` Preallocated Vector of length n for storing the `stablized_vals`, normalized.
* `cumsum_vals::Vector{Float64}` Preallocated Vector of length n for storing the cumulative sum of `norm_vals`.
---
Returns
* `sample_idx::Int` Index of the posterior sample of the variable.
"""

function griddy_sample!(posterior_vals::Vector{Float64},stable_vals::Vector{Float64},norm_vals::Vector{Float64},cumsum_vals::Vector{Float64})
    #posterior_vals .= exp.(posterior_vals)
    #println(posterior_vals)
    stability_const = maximum(posterior_vals)
    #println(stability_const)
    stable_vals .= exp.(posterior_vals .+ stability_const)
    #println(stable_vals)
    norm_vals .= stable_vals ./sum(stable_vals)
    #println(stable_vals)
    cumsum_vals .= cumsum(norm_vals)
    #println(cumsum_vals)
    uchance = rand(Uniform(0,1))
    #println("minimum = ",minimum(cumsum_vals),", max = ",maximum(cumsum_vals))
    #println("sample = $uchance")
    sample_idx = findfirst(x->x==true,cumsum_vals .> uchance)
    return sample_idx
end
