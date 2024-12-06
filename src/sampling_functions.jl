#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########################## Define Sampling Functions ##############################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

"""
    griddy_gibbs!(data::DataStr,sample_vals::GriddyVarsStruct,c_sse::Array{Float64},log_det_sig::Vector{Float64},sig_design::Array{Float64})
Function to perform griddy Gibbs sampling on the model and data.

---
Keyword arguments
* `data::DataStr` Struct containing computer simulator and experimental data.
* `sample_vals` Structure to store the sampled values during MCMC.
* `c_sse::Array{Float64}` Array to store precomputed values of the SSE scaled by the covariance matrix.
* `log_det_sig::Array{Float64}` Array to store the precomputed values for the determinant of the covariance matrix.
* `sig_design::Array{Float64}` Array to containing the ρ and σ*^2 input values for computing Σ.
* `priors::PriorData` Struct containing prior distribution hyperparameters.
"""
function griddy_gibbs!(data::DataStr,sample_vals::GriddyVarsStruct,c_sse::Array{Float64},
    log_det_sig::Array{Float64},sig_design::Array{Float64},priors::PriorData)
    
    n_loc = size(data.exp.y)[1]
    n_rep = size(data.exp.y)[2]
    rho = 1
    sig_star2 = 1
    theta = 1
    tau2 = 0.1


    loglik_theta = similar(c_sse[:,rho,sig_star2])
    loglik_covar = similar(c_sse[theta,:,:])
    loglik_pass = similar(c_sse[:,rho,sig_star2])

    #theta_post = similar(c_sse[:,rho,sig_star2])
    stable_theta = similar(c_sse[:,rho,sig_star2])
    norm_theta = similar(c_sse[:,rho,sig_star2])
    cumsum_theta = similar(c_sse[:,rho,sig_star2])

    #rho_post = similar(c_sse[theta,:,sig_star2])
    stable_rho = similar(c_sse[theta,:,sig_star2])
    norm_rho = similar(c_sse[theta,:,sig_star2])
    cumsum_rho = similar(c_sse[theta,:,sig_star2])

    sig_post = similar(c_sse[theta,rho,:])
    stable_sig = similar(c_sse[theta,rho,:])
    norm_sig = similar(c_sse[theta,rho,:])
    cumsum_sig = similar(c_sse[theta,rho,:])
    
    @inbounds @showprogress 1 "Sampling..." for i in 2:nmcmc
        
        #presolve for the inverse of τ^2 so it only needs to be done once per iteration
        tau2_inv = 1/tau2
        #println("τ^-1")
        #println(tau2_inv)

        #calculate the log-likelihood of θ and store in pre-allocated arrays
        #loglik_pass .= c_sse[:,rho,sig_star2]
        loglik_theta!(c_sse[:,rho,sig_star2],tau2_inv,loglik_theta)
        
        #calculate proportional posterior of θ
        #this calculation would go on this line a non-uniform prior is specified
        
        #sample θ index using full conditional with likelihood Array over θ dimension
        theta = griddy_sample!(loglik_theta,
        stable_theta,norm_theta,cumsum_theta)
        
        #store θ index into results struct
        sample_vals.theta[i] = theta
        
        #println("θ")
        #println(theta)
        
        #println("SSE")
        #println(describe(c_sse[1,:,:]))
        #println("τ")
        #println(tau2)
        #calculate log-likelihood for the entire covariance matrix and store into pre-allocated arrays
        loglik_covar!(c_sse[theta,:,:],log_det_sig,tau2_inv,loglik_covar)
        #println("Σ")
        #println(describe(loglik_covar))
        
        #calculate proportional posterior of ρ
        #this calculation would go on this line a non-uniform prior is specified

        #sample ρ using full conditional with likelihood Array over ρ dimension
        #println("ρ")
        #println(describe(loglik_covar[:,sig_star2]))
        rho = griddy_sample!(loglik_covar[:,sig_star2],
        stable_rho,norm_rho,cumsum_rho)
        #println(sig_design[rho,sig_star2,1])
        sample_vals.rho[i] = rho

        #calculate proportional posterior of σ*^2
        sig_post .= loglik_covar[rho,:] .+ logpdf(InverseGamma(
            priors.sig2.par1,priors.sig2.par2),sig_design[rho,sig_star2,2])

        #sample σ*^2 using full conditional with likelihood Array over σ*^2 dimension
        #println("σ*")
        #println(describe(sig_post))
        sig_star2 = griddy_sample!(sig_post,stable_sig,norm_sig,cumsum_sig)
        #println(sig_star2)
        sample_vals.sig_star2[i] = sig_star2
        #println(sig_design[rho,sig_star2,2])

        #sample τ^2 using semi-conjugate posterior distributions
        tau2 = rand(InverseGamma(priors.tau2.par1+n_loc*n_rep/2,
        priors.tau2.par2 + 0.5*c_sse[theta,rho,sig_star2]))

        sample_vals.tau2[i] = tau2
        
    end
    
end