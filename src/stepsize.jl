#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########################### Define Step Size Functions ############################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
"""
    stepsize_adjust(diff::Float64,scale::Float64,shape::Float64,offset::Float64)
Function to adjust a given stepsize based on the difference between its calculated acceptance rate and the target.

---
Keyword arguments
* `diff::Float64` The difference between the calculated acceptance ratio and the target.
* `scale::Float64` Scaling factor for the equation.
* `shape::Float64` Shape parameter for the equation.
* `offset::Float64` Offset parameter for skewing the adjustment to correct greater in the positive or negative direction

---
Returns
* `factor::Float64` The adjustment by which to multiply the current step size.

---
Details
The function uses an arc-tangent-based form for calculating the adjustment.
The arc-tangent is calculated for the product of the difference and the `shape`.
The arc-tangent is them multiplied by `scale`
This value is then shifted based on the `offset`
* positive values for `offset` correspond to greater corrections of negative differences (the acceptance ratio is too small)
* negative values for `offset` correspond to greater corrections of positive differences (the acceptance ratio is too large)
The exponentiation of this value is returned
"""
function stepsize_adjust(diff::Float64,scale::Float64,shape::Float64,offset::Float64)
    logscale = scale * atan(shape*diff)
    logscale = logscale + offset*sign(logscale) - offset
    factor = exp(logscale)
    return factor
end

"""
    update_stepsize(acceptance::Array{Bool},old_stepsize::StepSize,nx::Int,ntheta::Int)
    update_stepsize(acceptance::Array{Bool},old_stepsize::StepSize,nx::Int,ntheta::Int,target::Tuple{Float64},eta::Float64,factor::Float64,offset::Float64)
Function to update the Metropolis-Hastings algorithm step sizes based on the calculated acceptance ratios compared to their target values.

---
Keyword arguments
* `acceptance::Array{Bool}` Array containing the acceptance value for each M-H update.
* `old_stepsize::StepSize` Struct containing the step sizes used during the M-H updates for θ and ρ.
* `ntheta::Int` The number of θ variables using M-H updates.
* `nx::Int` The number of ρ variables using M-H updates.
Optional arguments
* `target::Tuple{Float64}` A length 2 Tuple containing the target acceptance rates for θ and ρ, respectively.
  * By defualt, 0.3 is used for both.
* `eta::Float64` Scaling parameter to pass into the `stepsize_adjust` function.
  * By default, 2.0 is used.
* `factor::Float64` Shape parameter to pass into the `stepsize_adjust` function.
  * By defualt, 30.0 is used.
* `offset::Float64` Offset parameter to pass into the `stepsize_adjust` function.
  * By default, 1.5 is used.
    """
function update_stepsize(acceptance::Array{Float64},old_stepsize::StepSize,
    nx::Int,ntheta::Int,target::Tuple{Float64}=Tuple([0.3,0.3]),
    eta::Float64 = 2.0,factor::Float64=30.0,offset::Float64=1.5)

    @inbounds for i in 1:ntheta
        rate = mean(acceptance[:,i+nx])
        old_stepsize.theta[i] = old_stepsize.theta[i]

        diff = rate - target[1]

        old_stepsize.theta[i] = old_stepsize.theta[i] *
            stepsize_adjust(diff,eta,factor,offset)
    end

    @inbounds for i in 1:nx
        rate = mean(acceptance[:,i])
        old_stepsize.rho[i] = old_stepsize.rho[i]

        diff = rate - target[2]
        
        old_stepsize.rho[i] = old_stepsize.rho[i] *
            stepsize_adjust(diff,eta,factor,offset)
    end

    return old_stepsize
end

"""

"""
function auto_stepsize(data::DataStr,nruns::Int,prior_data::PriorData,
    nsize::Int,nx::Int,ntheta::Int,nobs::Int,theta_init::Vector{Float64},
    init::Float64=1e-3)

    old_stepsize = StepSize(repeat([init],ntheta),repeat([init],nx))
    
    bulk_vars = init_vars(data,nruns,nx,ntheta,theta_init)
    step_vars = update_vars(bulk_vars.theta[1,:],bulk_vars.delta[1,:],
    bulk_vars.tau2[1],bulk_vars.sig2[1],bulk_vars.rho[1,:])

    nrep = trunc(Int,nruns/nsize)

    stepsize = Matrix(0.0I,nrep,nx+ntheta)
    acceptance = copy(stepsize)

    @inbounds @showprogress 1 "Computing Stepsize..." for i in 1:nrep
        start = (i-1)*nsize + 1 + ifelse(i==1,1,0)
        stop = i*nsize

        mcmc!(data,nsize,prior_data,bulk_vars,start,stop,
            old_stepsize,nx,ntheta,nobs)
        
        @inbounds for j in 1:nx
            stepsize[i,j] = old_stepsize.rho[j]
            acceptance[i,j] = mean(bulk_vars.accept[1:stop,j])
            old_stepsize.rho[j] = mean(stepsize[1:i,j])
        end

        @inbounds for j in 1:ntheta
            stepsize[i,j+nx] = old_stepsize.theta[j]
            acceptance[i,j+nx] = mean(bulk_vars.accept[1:stop,j+nx])
            old_stepsize.theta[j] = mean(stepsize[1:i,j+nx])
        end
        old_stepsize = update_stepsize(bulk_vars,old_stepsize,start,
            stop,nx,ntheta,nobs)
    end
    return old_stepsize,stepsize,acceptance
end

"""

"""
function stepsize_gd(bulk_vars::BulkVarsStruct,old_stepsize::StepSize,
    start::Int,stop::Int,nx::Int,ntheta::Int,nobs::Int,target::Tuple{Float64}=Tuple([0.3,0.3]),
    eta::Float64 = 2.0,factor::Float64=30,offset::Float64=1.5,theta_init::Vector{Float64},init::Float64=1e-3)
    
    old_stepsize = StepSize(repeat([init],ntheta),repeat([init],nx))
    current_stepsize = StepSize(repeat([init],ntheta),repeat([init],nx))

    bulk_vars = init_vars(data,nruns,nx,ntheta,theta_init)
    
    nrep = trunc(Int,nruns/nsize)

    stepsize = Matrix(0.0I,nrep,nx+ntheta)
    acceptance = copy(stepsize)

    function acceptance_objfn(vars::Vector{Float64})
        mcmc!(data,nsize,prior_data,bulk_vars,start,stop,
        old_stepsize,nx,ntheta,nobs)
        acceptance = mean(bulk_vars.accept[start:stop,:],dims=1)
        return (acceptance .- target).^2
    end

    @inbounds @showprogress 1 "Computing Stepsize..." for i in 1:nrep
        start = (i-1)*nsize + 1 + ifelse(i==1,1,0)
        stop = i*nsize

        points = hcat(old_stepsize.theta,old_stepsize.rho)
        grad = gradient(x -> acceptance_objfn(x),points)

        points -= eta .* points

        old_stepsize.theta = points[1:ntheta]
        old_stepsize.rho = points[(ntheta+1):(nx+ntheta)]

        end
    return old_stepsize,stepsize,acceptance
end