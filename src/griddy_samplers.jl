#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########################## Define Sampling Functions ##############################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

"""
    griddy_gibbs!(data::DataStr,sample_vals::GriddyVarsStruct,c_sse::Array{Float64,3},
    log_det_sig::Vector{Float64,2},sig_design::Array{Float64,3},
    priors::PriorData,nmcmc::Int)
Function to perform griddy Gibbs sampling on the model and data.
Implementation for a multivariate normal distribution data model.

---
Positional arguments
* `data::DataStr` Struct containing computer simulator and experimental data.
* `sample_vals` Structure to store the sampled values during MCMC.
* `c_sse::Array{Float64,3}` Array to store precomputed values of the SSE scaled by the covariance matrix.
* `log_det_sig::Array{Float64,2}` Array to store the precomputed values for the determinant of the covariance matrix.
* `sig_design::Array{Float64,3}` Array to containing the ρ and ϕ input values for computing Σ.
* `priors::PriorData` Struct containing prior distribution hyperparameters.
* `nmcmc::Int` Number of samples to draw.
"""
function griddy_gibbs!(data::DataStr,sample_vals::GriddyVarsStruct,c_sse::Array{Float64,3},
            log_det_sig::Array{Float64,2},sig_design::Array{Float64,3},
            priors::PriorData,nmcmc::Int)
    
    # initialize constant integers
    n_loc = size(data.exp.y)[1]
    n_rep = size(data.exp.y)[2]
    #initialize values for sampling variables
    rho = 1
    phi = 1
    theta = 1
    sig2 = 0.1

    #initialize vectors for storing computations
    loglik_theta = similar(c_sse[:,rho,phi])
    loglik_covar = similar(c_sse[theta,:,:])
    
    stable_theta = similar(c_sse[:,rho,phi])
    norm_theta = similar(c_sse[:,rho,phi])
    cumsum_theta = similar(c_sse[:,rho,phi])

    stable_rho = similar(c_sse[theta,:,phi])
    norm_rho = similar(c_sse[theta,:,phi])
    cumsum_rho = similar(c_sse[theta,:,phi])

    phi_post = similar(c_sse[theta,rho,:])
    stable_phi = similar(c_sse[theta,rho,:])
    norm_phi = similar(c_sse[theta,rho,:])
    cumsum_phi = similar(c_sse[theta,rho,:])
    
    @inbounds @showprogress 1 "Sampling..." for i in 2:nmcmc
        
        #presolve for the inverse of σ^2 so it only needs to be done once per iteration
        sig2_inv = 1/sig2

        #calculate the log-likelihood of θ and store in pre-allocated arrays
        loglik_theta!(c_sse[:,rho,phi],sig2_inv,loglik_theta)
        
        #calculate proportional posterior of θ
        #this calculation would go on this line a non-uniform prior is specified
        
        #sample θ index using full conditional with likelihood Array over θ dimension
        theta = griddy_sample!(loglik_theta,
        stable_theta,norm_theta,cumsum_theta)
        
        #store θ index into results struct
        sample_vals.theta[i] = theta

        #calculate log-likelihood for the entire covariance matrix and store into pre-allocated arrays
        loglik_covar!(c_sse[theta,:,:],log_det_sig,sig2_inv,loglik_covar)
        
        #calculate proportional posterior of ρ
        #this calculation would go on this line a non-uniform prior is specified

        #sample ρ using full conditional with likelihood Array over ρ dimension
        rho = griddy_sample!(loglik_covar[:,phi],
        stable_rho,norm_rho,cumsum_rho)
        sample_vals.rho[i] = rho

        #calculate proportional posterior of ϕ
        phi_post .= loglik_covar[rho,:] .+ logpdf(InverseGamma(
            priors.tau2.par1,priors.tau2.par2),sig_design[rho,phi,2])

        #sample ϕ using full conditional with likelihood Array over ϕ dimension
        phi = griddy_sample!(phi_post,stable_phi,norm_phi,cumsum_phi)
        sample_vals.phi[i] = phi

        #sample σ^2 using semi-conjugate posterior distributions
        sig2 = rand(InverseGamma(priors.sig2.par1+n_loc*n_rep/2,
        priors.sig2.par2 + 0.5*c_sse[theta,rho,phi]))

        sample_vals.sig2[i] = sig2 
    end
end

"""
    griddy_gibbs!(data::DataStr,sample_vals::GriddyVarsStruct,c_sse::Vector{Float64},
    log_det_sig::Vector{Float64,2},sig_design::Array{Float64,3},
    priors::PriorData,nmcmc::Int)
Function to perform griddy Gibbs sampling on the model and data.
Implementation for a univarite normal distribution data model.
The same arguments that would be passed to the multivariate normal distribution are accepted here to allow the function call to be the same for either data model.
The key diffence is that this accepts a Vector for c_sse instead of an `Array{Float64,3}` and the variables for the corrleation structure are `Nothing`.
These types are automatically determined by and returned from the `preallocate()` function, allowing for identical function calls.

---
Positional arguments
* `data::DataStr` Struct containing computer simulator and experimental data.
* `sample_vals` Structure to store the sampled values during MCMC.
* `c_sse::Vector{Float64}` Array to store precomputed values of the SSE scaled by the covariance matrix.
* `log_det_sig::Nothing` Dummy argument to allow same function call as MVN implementation.
* `sig_design::Nothing` Dummy argument to allow same function call as MVN implementation.
* `priors::PriorData` Struct containing prior distribution hyperparameters.
* `nmcmc::Int` Number of samples to draw.
"""
function griddy_gibbs!(data::DataStr,sample_vals::GriddyVarsStruct,c_sse::Vector{Float64},
    log_det_sig::Nothing,sig_design::Nothing,priors::PriorData,nmcmc::Int)
    
    #initialize constant integers
    n_loc = size(data.exp.y)[1]
    n_rep = size(data.exp.y)[2]

    #initialize values for sampling variables
    theta = 1
    sig2 = 0.1

    #initialize Vectors to store calculations
    loglik_theta = similar(c_sse)

    stable_theta = similar(c_sse)
    norm_theta = similar(c_sse)
    cumsum_theta = similar(c_sse)
    
    @inbounds @showprogress 1 "Sampling..." for i in 2:nmcmc
        
        #presolve for the inverse of σ^2 so it only needs to be done once per iteration
        sig2_inv = 1/sig2

        #calculate the log-likelihood of θ and store in pre-allocated arrays
        loglik_theta!(c_sse[:,rho,phi],sig2_inv,loglik_theta)
        
        #calculate proportional posterior of θ
        #this calculation would go on this line a non-uniform prior is specified
        
        #sample θ index using full conditional with likelihood Array over θ dimension
        theta = griddy_sample!(loglik_theta,
        stable_theta,norm_theta,cumsum_theta)
        
        #store θ index into results struct
        sample_vals.theta[i] = theta

        #sample σ^2 using semi-conjugate posterior distributions
        sig2 = rand(InverseGamma(priors.sig2.par1+n_loc*n_rep/2,
        priors.sig2.par2 + 0.5*c_sse[theta]))

        sample_vals.sig2[i] = sig2
        
    end
    
end
