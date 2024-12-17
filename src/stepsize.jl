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
    #println(logscale)
    logscale = logscale + offset*sign(logscale) - offset
    #println(logscale)
    factor = exp(logscale)
    #println("diff = $diff")
    #println("factor = $factor")
    return factor
end

"""
    update_stepsize!(acceptance::Array{Bool},old_stepsize::StepSize,nx::Int,ntheta::Int,target::Tuple{Float64},eta::Float64,factor::Float64,offset::Float64)
Function to update the Metropolis-Hastings algorithm step sizes based on the calculated acceptance ratios compared to their target values.

---
Keyword arguments
* `acceptance::Array{Bool}` Array containing the acceptance value for each M-H update.
* `old_stepsize::StepSize` Struct containing the step sizes used during the M-H updates for θ and ρ.
* `ntheta::Int` The number of θ variables using M-H updates.
* `nx::Int` The number of ρ variables using M-H updates.
* `target::Tuple{Float64}` A length 2 Tuple containing the target acceptance rates for θ and ρ, respectively.
* `scale::Float64` Scaling parameter to pass into the `stepsize_adjust` function.
* `shape::Float64` Shape parameter to pass into the `stepsize_adjust` function.
* `offset::Float64` Offset parameter to pass into the `stepsize_adjust` function.
"""
function update_stepsize!(acceptance::Vector{Float64},stepsize::StepSize,
    history::Array{Float64},nx::Int,ntheta::Int,target::Vector{Float64},
    scale::Float64,shape::Float64,offset::Float64,weight::Float64)

    @inbounds for i in 1:ntheta
        rate = acceptance[i+nx]

        diff = rate - target[1]

        adjustment = stepsize_adjust(diff,scale,shape,offset)
        stepsize.theta[i] = weighted_avg(history[1:(end-1),i+nx],
            history[end,i+nx],weight)
        stepsize.theta[i] = weighted_avg(repeat([stepsize.theta[i]],
        size(history)[1]),stepsize.theta[i]*adjustment,weight)
        #println("theta diff = $diff")
        #println("theta adjustment = $adjustment")
    end

    @inbounds for i in 1:nx
        rate = acceptance[i]

        diff = rate - target[2]
        #println("rho $i old stepsize = $(stepsize.rho[i])")
        adjustment = stepsize_adjust(diff,scale,shape,offset)
        stepsize.rho[i] = weighted_avg(history[1:(end-1),i],
            history[end,i],weight)
        stepsize.rho[i] = weighted_avg(repeat([stepsize.rho[i]],
        size(history)[1]),stepsize.rho[i]*adjustment,weight)
        
        #println("rho $i acceptance = $rate")
        #println("rho $i difference = $diff")
        #println("rho $i adjustment = $adjustment")
        #println("rho $i proposed stepsize = $(stepsize.rho[i])")
    end

    return stepsize
end

"""
"""
function weighted_avg(values::Vector{Float64},new::Float64,weight::Float64)
    n = sum(values) + weight*new
    d = length(values) + weight
    avg = n/d
    return avg
end

"""
"""
function weighted_avg(values::Vector{Bool},new::Vector{Bool},weight::Float64)
    n = sum(values) + weight*sum(new)
    d = length(values) + weight*length(new)
    avg = n/d
    return avg
end
"""
    auto_stepsize(data::DataStr,nruns::Int,prior_data::PriorData,nsize::Int,nx::Int,ntheta::Int,nobs::Int,theta_init::Vector{Float64})
    auto_stepsize(data::DataStr,nruns::Int,prior_data::PriorData,nsize::Int,nx::Int,ntheta::Int,nobs::Int,theta_init::Vector{Float64},init::Float64,target::Tuple{Float64},eta::Float64,factor::Float64,offset::Float64)
Function to calculate the appropriate step size for the Metropolis-Hastings algorithm, for a target acceptance ratio.

---
Keyword arguments
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `nbatch::Int` The number of batches of MCMC simulations to run.
* `batchsize::Int` The number of MCMC iterations ro run per batch.
* `nx::Int` The number of x dimensions.
* `ntheta::Int` The number of θ dimensions.
* `nobs::Int` The number of unique settings of x in the experimental data.
* `theta_init::Vector{Float64}` The settings of θ at which to initialize the MCMC.
Optional arguments
* `init::Float64` The inital step size for the start of this algorithm.
  * By default, 1E-3 is used.
* `target::Tuple{Float64}` A length 2 Tuple containing the target acceptance rates for θ and ρ, respectively.
  * By defualt, 0.3 is used for both.
* `scale::Float64` Scaling parameter to pass into the `stepsize_adjust` function.
  * By default, 2.0 is used.
* `shape::Float64` Shape parameter to pass into the `stepsize_adjust` function.
  * By defualt, 30.0 is used.
* `offset::Float64` Offset parameter to pass into the `stepsize_adjust` function.
  * By default, 1.5 is used.
---
Returns
* `stepsize::StepSize` Struct containing the calculated stepsizes that will result in the target acceptance rate.
* `stepsize_hist::Array{Float64}` An Array containing the historic data of the stepsizes used in this algorithm.
* `acceptance_hist::Array{Float64}` An Array containing the historic data of the acceptance rates used in this algorithm.

---
Details
This algorithm runs `nbatch` batches of MCMC with M-H updates, each batch having a length of `batchsize`.
The algorithm starts at a step size specified by `init` for the first batch.
After each batch, the algorithm calculates the average acceptance rate of all M-H samples from all preceding batches.
This value is passed to `update_stepsize` to calculate the proposed adjustment to the step size for the next batch.
The step size for the next batch is then calculated as the average of all previous step sizes and the value calculated from `update_stepsize`.
"""
function auto_stepsize(model,data::DataStr,nbatch::Int,batchsize::Int,prior_data::PriorData,
    nx::Int,ntheta::Int,nloc::Int,theta_init::Vector{Float64},
    init::Float64,target::Vector{Float64},
    scale::Float64,shape::Float64,offset::Float64)

    stepsize = StepSize(repeat([init],ntheta),repeat([init],nx))
    
    total_length = batchsize*nbatch
    mcmc_vars = init_vars(data,total_length,nx,ntheta,theta_init)

    stepsize_hist = Array{Float64}(undef,nbatch,nx+ntheta)
    acceptance_hist = Array{Float64}(undef,nbatch,nx+ntheta)

    @inbounds @showprogress 1 "Computing Stepsize..." for i in 1:nbatch
        #println(i)
        #println(i==1)
        #println(ifelse(i==1,1,0))
        weight = sqrt(i)
        start = (i-1)*batchsize + 1 + ifelse(i==1,1,0)
        stop = i*batchsize

        mcmc!(model,data,prior_data,mcmc_vars,start,stop,
            stepsize,nx,ntheta,nloc)
        
        @inbounds for j in 1:nx
            #println("rho $j actual acceptance = $(mean(mcmc_vars.accept[start:stop,j]))")
            stepsize_hist[i,j] = stepsize.rho[j]
            #println("rate_2 = $(mean(mcmc_vars.accept[start:stop,j]))")
            acceptance_hist[i,j] = weighted_avg(mcmc_vars.accept[1:(start-1),
            j],mcmc_vars.accept[start:stop,j],weight)
        end

        @inbounds for j in 1:ntheta
            stepsize_hist[i,j+nx] = stepsize.theta[j]
            acceptance_hist[i,j+nx] = weighted_avg(mcmc_vars.accept[1:(start-1),
            j+nx],mcmc_vars.accept[start:stop,j+nx],weight)
        end

        stepsize = update_stepsize!(acceptance_hist[i,:],stepsize,
            stepsize_hist[1:i,:],nx,ntheta,target,scale,shape,offset,
            weight)
    end
    return stepsize,stepsize_hist,acceptance_hist
end

"""
    stepsize_gd(data::DataStr,nbatch::Int,nsize::Int,prior_data::PriorData,nx::Int,ntheta::Int,nobs::Int,theta_init::Vector{Float64})
    stepsize_gd(data::DataStr,nbatch::Int,nsize::Int,prior_data::PriorData,nx::Int,ntheta::Int,nobs::Int,theta_init::Vector{Float64},init::Float64,target::Tuple{Float64},eta::Float64)
Function to optimize the M-H step size via gradient descent.

---
Keyword arguments
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `nbatch::Int` The number of batches of MCMC simulations to run.
* `batchsize::Int` The number of MCMC iterations ro run per batch.
* `nx::Int` The number of x dimensions.
* `ntheta::Int` The number of θ dimensions.
* `nobs::Int` The number of unique settings of x in the experimental data.
* `theta_init::Vector{Float64}` The settings of θ at which to initialize the MCMC.
Optional arguments
* `init::Float64` The inital step size for the start of this algorithm.
  * By default, 1E-3 is used.
* `target::Tuple{Float64}` A length 2 Tuple containing the target acceptance rates for θ and ρ, respectively.
  * By defualt, 0.3 is used for both.
* `eta::Float64` Learning rate for the gradient descent.
  * By default, 0.1 is used.
---
Returns
* `stepsize::StepSize` Struct containing the calculated stepsizes that will result in the target acceptance rate.
* `stepsize_hist::Array{Float64}` An Array containing the historic data of the stepsizes used in this algorithm.
* `acceptance_hist::Array{Float64}` An Array containing the historic data of the acceptance rates used in this algorithm.

---
Details
This algorithm runs `nbatch` batches of MCMC with M-H updates, each batch having a length of `batchsize`.
The algorithm starts at a step size specified by `init` for the first batch.
After each batch, the algorithm calculates the acceptance rate of that batch.
The gradient of a SSE objective function (as a function of the batch acceptance rate compared to the target) is calculated as a function of the log step size of θ and ρ.
The the log step size for each θ and ρ is updated by subtracting the gradient multiplied by `eta`.
"""
function stepsize_gd(model,data::DataStr,nbatch::Int,batchsize::Int,prior_data::PriorData,
    nx::Int,ntheta::Int,nobs::Int,theta_init::Vector{Float64},init::Float64,
    target::Vector{Float64},eta::Float64)
    
    stepsize = StepSize(repeat([log(init)],ntheta),repeat([log(init)],nx))
    
    total_length = batchsize*nbatch
    mcmc_vars = init_vars(data,total_length,nx,ntheta,theta_init)

    stepsize_hist = Array{Float64}(undef,nbatch,nx+ntheta)
    acceptance_hist = Array{Bool}(undef,nbatch,nx+ntheta)

    function acceptance_objfn(vars::Vector{Float64},model,start::Int,stop::Int)
        stepsize.rho[:] .= exp.(vars[1:nx])
        stepsize.theta[:] .= exp.(vars[(nx+1):(ntheta+nx)])

        mcmc!(model,data,prior_data,mcmc_vars,start,stop,
        stepsize,nx,ntheta,nobs)
        
        acceptance = mean(bulk_vars.accept[start:stop,:],dims=1)
        acceptance_hist[i,:] = acceptance
        return (acceptance .- target).^2
    end

    @inbounds @showprogress 1 "Computing Stepsize..." for i in 1:nbatch
        start = (i-1)*batchsize + 1 + ifelse(i==1,1,0)
        stop = i*batchsize

        points = vcat(stepsize.rho,stepsize.theta)
        grad = gradient(x -> acceptance_objfn(x,model,start,stop),points)

        points -= eta .* grad

        stepsize_hist[i,:] = exp.(points)

        end
    return stepsize,stepsize_hist,acceptance_hist
end

"""
    plot_stepsize_opt(stepsize::Array{Float64},acceptance::Array{Float64},nx::Int,ntheta::Int,show::Bool,save::Bool)
Function to plot the results of the M-H stepsize optimization algorithm.

---
Keyword arguments
* `stepsize::Array{Float64}` A n by `nx`+`ntheta` Array containing the M-H step sizes used in the stepsize optimization algorithm.
* `acceptance::Array{Float64}` A n by `nx`+`ntheta` Array containing the calculated M-H acceptance rates corresponding to `stepsize`.
* `nx::Int` The number of x dimesions.
* `ntheta::Int` The number of θ dimensions.
* `save::Bool` An indication of whether the plots should be saved.
* `show::Bool` An indication of whether the plots should be displayed.
"""
function plot_stepsize_opt(stepsize::Array{Float64},acceptance::Array{Float64},
    nx::Int,ntheta::Int,show_plots::Bool,save_plots::Bool,mdl_apnd::String)
    function plot_stepsize(epochs::Int,stepsize::Vector{Float64},var::String,iter::Int)
        p = Plots.plot(1:epochs,stepsize,label=false,top_margin=5mm,left_margin=5mm,
            titlelocation=[0.5,1.05])
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Stepsize over Epochs"))
        xlabel!("Epoch")
        ylabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Stepsize"))
        save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-stepsize.png") : nothing
        show_plots ? Plots.display(p) : nothing
    end

    function plot_acceptance(epochs::Int,acceptance::Vector{Float64},var::String,iter::Int)
        p = Plots.plot(1:epochs,acceptance,label=false,top_margin=5mm,left_margin=5mm,
        titlelocation=[0.5,1.05])
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance over Epochs"))
        xlabel!("Epoch")
        ylabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance"))
        save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-acceptance.png") : nothing
        show_plots ? Plots.display(p) : nothing
    end

    function plot_step_acc(epochs::Int,acceptance::Vector{Float64},stepsize::Vector{Float64},
        var::String,iter::Int)
        epochs = collect(1:epochs)
        p = Plots.scatter(stepsize,acceptance,zcolor=epochs,label=false,top_margin=5mm,left_margin=5mm,
        titlelocation=[0.5,1.05])
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance vs. Stepsize"))
        xlabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Stepsize"))
        ylabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance"))
        save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-acceptance_v_stepsize.png") : nothing
        show_plots ? Plots.display(p) : nothing
    end

    function plot_secants(epochs::Int,acceptance::Vector{Float64},stepsize::Vector{Float64},
        var::String,iter::Int)
        p = Plots.plot(1:epochs,stepsize,labe=false,top_margin=5mm,left_margin=5mm,
        titlelocation=[0.5,1.05])
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Stepsize Convergence"))
        xlabel!("Epoch")
        ylabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Stepsize Rate of Change"))
        save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-stepsize_convergence.png") : nothing
        show_plots ? Plots.display(p) : nothing

        p = Plots.plot(1:epochs,acceptance,label=false,top_margin=5mm,left_margin=5mm,
        titlelocation=[0.5,1.05])
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance Convergence"))
        xlabel!("Epoch")
        ylabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Acceptance Rate of Change"))
        save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-acceptance_convergence.png") : nothing
        show_plots ? Plots.display(p) : nothing
    end

    secants = assess_convergence(acceptance,stepsize,nx,ntheta)
    epochs = size(acceptance)[1]
    for rho in 1:nx
        plot_stepsize(epochs,stepsize[:,rho],"rho",rho)
        plot_acceptance(epochs,acceptance[:,rho],"rho",rho)
        plot_step_acc(epochs,acceptance[:,rho],stepsize[:,rho],"rho",rho)
        plot_secants(epochs-1,secants[1][:,rho],secants[2][:,rho],"rho",rho)
    end

    for theta in 1:ntheta
        plot_stepsize(epochs,stepsize[:,nx+theta],"theta",theta)
        plot_acceptance(epochs,acceptance[:,nx+theta],"theta",theta)
        plot_step_acc(epochs,acceptance[:,nx+theta],stepsize[:,nx+theta],"theta",theta)
        plot_secants(epochs-1,secants[1][:,nx+theta],secants[2][:,nx+theta],"theta",theta)
    end
end

"""
    assess_convergence(stepsize::Array{Float64},acceptance::Array{Float64},nx::Int,ntheta::Int)
Function to assess the convergence of the M-H stepsize optimization results.

---
Keyword arguments
* `stepsize::Array{Float64}` A n by `nx`+`ntheta` Array containing the M-H step sizes used in the stepsize optimization algorithm.
* `acceptance::Array{Float64}` A n by `nx`+`ntheta` Array containing the calculated M-H acceptance rates corresponding to `stepsize`.
* `nx::Int` The number of x dimesions.
* `ntheta::Int` The number of θ dimensions.

---
Returns
* `secants::Tuple{Array{Float64}}` A Tuple of length 2, with each element being a (n-1) by `nx`+`ntheta` Array.
  * The Arrays contain the calculated secants of the acceptance rates and stepsizes, respectively, calculated as a function of the epochs, relative to the final epoch.

---
Details
This function calculates the slope of the secant line between each element of the `acceptance` and `stepsize` Arrays relative to the last element.
"""
function assess_convergence(stepsize::Array{Float64},acceptance::Array{Float64},
    nx::Int,ntheta::Int)

    epochs = size(acceptance)[1]
    breaks = collect(1:epochs)
    temp = Vector{Float64}(undef,length(breaks)-1)
    secants_acceptance = Array{Float64}(undef,length(temp),(nx+ntheta))
    secants_stepsize = similar(secants_acceptance)

    function calc_secant!(breaks::Vector{Int},y::Vector{Float64},temp::Vector{Float64})
        for i in eachindex(temp)
            dy = y[end] - y[breaks[i]]
            dx = length(temp) - 1
            temp[i] = dy/dx
        end
    end

    for rho in 1:nx
        calc_secant!(breaks,acceptance[:,rho],temp)
        secants_acceptance[:,rho] .= temp
        calc_secant!(breaks,stepsize[:,rho],temp)
        secants_stepsize[:,rho] .= temp
    end
    for theta in 1:ntheta
        calc_secant!(breaks,acceptance[:,nx+theta],temp)
        secants_acceptance[:,nx+theta] .= temp
        calc_secant!(breaks,stepsize[:,nx+theta],temp)
        secants_stepsize[:,nx+theta] .= temp
    end
    return secants_acceptance,secants_stepsize
end

"""
    find_stepsize(data::DataStr,nbatch::Int,batchsize::Int,prior_data::PriorData,nx::Int,ntheta::Int,nobs::Int,theta_init::Vector{Float64},method::Int,make_plots::Bool,show_plots::Bool,save_plots::Bool)
    find_stepsize(data::DataStr,nbatch::Int,batchsize::Int,prior_data::PriorData,nx::Int,ntheta::Int,nobs::Int,theta_init::Vector{Float64},method::Int,make_plots::Bool,show_plots::Bool,save_plots::Bool,init::Float64,target::Tuple{Float64},scale::Float64,shape::Float64,offset::Float64)
Function to solve for the appropriate step size for θ and ρ in the Metropolis-Hastings algorithm for Bayesian calibration.

---
Keyword arguments
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `nbatch::Int` The number of batches of MCMC simulations to run.
* `batchsize::Int` The number of MCMC iterations ro run per batch.
* `nx::Int` The number of x dimensions.
* `ntheta::Int` The number of θ dimensions.
* `nobs::Int` The number of unique settings of x in the experimental data.
* `theta_init::Vector{Float64}` The settings of θ at which to initialize the MCMC.
* `method::Int` Indicator of which method to use. Below are the available options
  * 1 -> `auto_stepsize` a function using automatic adjustment of averaged stepsizes.
  * 2 -> `stepsize_gd` a function that performs gradient descent optimization on the stepsizes.
* `make_plots::Bool` Indicator of whether to generate plots from the results of the algorithm.
* `show_plots::Bool` Indicator of whether to display the plots generated.
* `save_plots::Bool` Indicator of whether to save the plots generated.
Optional arguments
* `init::Float64` The inital step size for the start of this algorithm.
  * By default, 1E-3 is used.
* `target::Tuple{Float64}` A length 2 Tuple containing the target acceptance rates for θ and ρ, respectively.
  * By defualt, 0.3 is used for both.
* `scale::Float64` Scaling parameter to pass into the `stepsize_adjust` function.
  * By default, 2.0 is used.
* `shape::Float64` Shape parameter to pass into the `stepsize_adjust` function.
  * By defualt, 30.0 is used.
* `offset::Float64` Offset parameter to pass into the `stepsize_adjust` function.
  * By default, 1.5 is used.
* `eta::Float64` Learning rate for the gradient descent.
  * By default, 0.1 is used.
---
Returns
* `stepsize::StepSize` Struct containing the calculated stepsizes that will result in the target acceptance rate.
"""
function find_stepsize(model,data::DataStr,nbatch::Int,batchsize::Int,prior_data::PriorData,
    nx::Int,ntheta::Int,nloc::Int;theta_init::Union{Vector{Float64},Float64}=0.5,
    make_plots::Bool=true,show_plots::Bool=true,save_plots::Bool=true,
    init::Float64=1e-3,target::Vector{Float64}=[0.3,0.3],eta::Float64=0.3,
    scale::Float64 = 2.0,shape::Float64=10.0,offset::Float64=1.5,mdl_apnd::String="")

    if typeof(theta_init) == Float64
        theta_init = repeat([theta_init],ntheta)
    end

    stepsize,stepsize_hist,acceptance_hist = auto_stepsize(model,data,nbatch,batchsize,prior_data,nx,
        ntheta,nloc,theta_init,init,target,scale,shape,offset)

    make_plots ? plot_stepsize_opt(stepsize_hist,acceptance_hist,nx,ntheta,show_plots,save_plots,mdl_apnd) : nothing

    return stepsize
end