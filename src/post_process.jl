#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
############################# Define Misc Functions ###############################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
"""
    normalize_samples(samples::BulkVarsStruct,scales::Scaling)
    normalize_samples(samples::BulkVarsStruct,scales::Scaling,rev::Bool)
Function to normalize or reverse the normalization of the posterior samples.
This is the implementation for the continuous sampler results.

---
Keyword arguments
* `samples::BulkVarsStruct` Struct containing the posterior samples.
* `scales::Scaling` Struct containing the minimum and maximum values for each variables.
Optional arguments
* `rev::Bool` Indicator of whether to reverse the normalization or not.
"""
function normalize_samples(samples::BulkVarsStruct,scales::Scaling,rev::Bool=true)

    theta .= rev ? unnormalize_var(samples.theta,scales.theta) : 
        normalize_var(samples.theta,scales.theta)
    sig2 .= rev ? (unnormalize_var(sqrt.(samples.sig2),scales.y)).^2 :
        (normalize_var(sqrt.(samples.sig2),scales.y)).^2
    delta .= rev ? unnormalize_var(samples.delta,scales.y) :
        normalize_var(samples.delta,scales.y)
    tau2 .= rev ? (unnormalize_var(sqrt.(samples.tau2),scales.y)).^2 :
        (normlize_var(sqrt.(samples.tau2),scales.y)).^2
    
    samples_norm = BulkVarsStruct(theta=theta,tau2=tau2,sig2=sig2,delta=delta,
        rho=samples.rho,accept=samples.accept,ratio=samples.ratio)
    
    return samples_norm
end

"""
    normalize_samples(samples::GriddyVarsStruct,scales::Scaling,theta_grid::Array{Float64},sig_grid::Array{Float64})
    normalize_samples(samples::GriddyVarsStruct,scales::Scaling,theta_grid::Array{Float64},sig_grid::Array{Float64},rev::Bool)
Function to normalize or reverse the normalization of the posterior samples.
This implementation is for the griddy Gibbs sampler.

---
Keyword arguments
* `samples::GriddyVarsStruct` Struct containing the posterior samples (in index form) from the griddy Gibbs sampler.
* `scales::Scaling` Struct containing the minimum and maximum values for each variables.
* `theta_grid::Array{Float64}` Array containing the sampling grid for θ that was used for the griddy Gibbs sampler.
* `sig_grid::Array{Float64}` Array containing the sampling grid for the proportional covariance matrix used for the griddy Gibbs sampler.
Optional arguments
* `rev::Bool` Indicator of whether to reverse the normalization or not.

---
Returns
* `converted_samples::GriddyPosteriors` A struct containing the real value samples from the griddy Gibbs sampler.
"""
function normalize_samples(samples::GriddyVarsStruct,scales::Scaling,theta_grid::Array{Float64},
    sig_grid::Array{Float64},rev::Bool=true)

    theta = rev ? unnormalize_var(theta_grid[samples.theta],scales.theta) :
        normalize_var(design[samples.theta],scales.theta)
    tau2 = rev ? unnormalize_var(samples.tau2,scales.y) :
        normalize_var(samples.tau2,scales.y)
    rho = sig_grid[samples.rho,1,1]
    sig_star2 = sig_grid[1,samples.sig_star2,2] .* tau2

    converted_samples = GriddyPosteriors(theta=theta,tau2=tau2,rho=rho,sig_star2=sig_star2)

    return converted_samples
end

"""
    remove_burn(samples::BulkVarsStruct,nburn::Int)
Function to extract the posterior samples after the burned values.

---
Keyword arguments
* `samples::BulkVarsStruct` Struct containing the posterior samples from the continuous sampler.
* `nburn::Int` The number of samples to discard

---
Returns
* `truncated::BulkVarsStruct` Bulk vars struct that contains the samples retained after discarding the burned samples.

---
Details
This selects all posterior samples from `nburn` + 1 through the end of the samples.
These values are retained and placed into a new posterior samples struct, which is then returned.
"""
function remove_burn(samples::BulkVarsStruct,nburn::Int)
    if nburn >= length(samples.sig2)
        error("The number of specified to burn is greater than or equal to the total number of samples.
        Please specify a number of samples to burn that is less than the total number of samples.
        Total number of samples = $(length(samples.sig2))
        Specified burn = $nburn")
    end
    keep = nburn + 1
    theta = samples.theta[retained,:]
    rho = samples.rho[retained,:]
    tau2 = samples.tau2[retained]
    sig2 = samples.sig2[retained]
    delta = samples.delta[retained,:]
    accept = samples.delta[retained,:]
    ratio = samples.ratio[retained,:]

    truncated = BulkVarsStruct(theta=theta,rho=rho,tau2=tau2,sig2=sig2,
    delta=delta,accept=accept,ratio=ratio)
    
    return truncated
end

"""
    remove_burn(samples::GriddyPosteriors,nburn::Int)
Function to extract the posterior samples after the burned values.

---
Keyword arguments
* `samples::GriddyPosteriors` Struct containing the posterior samples from the griddy Gibbs sampler.
* `nburn::Int` The number of samples to discard

---
Returns
* `truncated::GriddyPosteriors` Bulk vars struct that contains the samples retained after discarding the burned samples.

---
Details
This selects all posterior samples from `nburn` + 1 through the end of the samples.
These values are retained and placed into a new posterior samples struct, which is then returned.
"""
function remove_burn(samples::GriddyPosteriors,nburn::Int)
    if nburn >= length(samples.tau2)
        error("The number of specified to burn is greater than or equal to the total number of samples.
        Please specify a number of samples to burn that is less than the total number of samples.
        Total number of samples = $(length(samples.tau2))
        Specified burn = $nburn")
    end
    keep = nburn + 1
    theta = samples.theta[retained,:]
    tau2 = samples.tau2[retained]
    sig_star2 = samples.sig_star2[retained]
    rho = samples.rho[retained]

    truncated = GriddyPosteriors(theta=theta,tau2=tau2,sig_star2,rho=rho)

    return truncated
end

"""
    thin_samples(samples::BulkVarsStruct,nthin::Int)
Function to thin the posterior samples, helping remove autocorrelated values from the samples.
This implementation is for the continuous sampler.

---
Keyword arguments
* `samples::BulkVarsStruct` Struct storing the posterior samples.
* `nthin::Int` The number of samples to skip.

---
Returns
* `thinned::BulkVarsStruct` Struct containing the thinned posterior samples.

---
Details
This function selects retains every `nthin`th sample from the posterior samples, starting at the first index and going through to the end.
"""
function thin_samples(samples::BulkVarsStruct,nthin::Int)
    if nthin >= length(samples.sig2)
        error("The number of specified to burn is greater than or equal to the total number of samples.
        Please specify a number of samples to burn that is less than the total number of samples.
        Total number of samples = $(length(samples.sig2))
        Specified thinning = $nthin")
    end

    retained = 1:nthin:length(samples.sig2)
    if length(retained) < 100
        @warn "The specified thinning rate, $nthin, seems large for the number of posterior samples.
        Total number of samples =  $(length(samples.sig2))
        Number of samples retained = $(length(retained))
        Number of samples discarded = $(length(samples.sig2) - length(retained))
        Consider using a smaller thinning rate or drawing more posterior samples."
    end

    theta = samples.theta[retained,:]
    rho = samples.rho[retained,:]
    tau2 = samples.tau2[retained]
    sig2 = samples.sig2[retained]
    delta = samples.delta[retained,:]
    accept = samples.delta[retained,:]
    ratio = samples.ratio[retained,:]

    thinned = BulkVarsStruct(theta=theta,rho=rho,tau2=tau2,sig2=sig2,
    delta=delta,accept=accept,ratio=ratio)
    
    return thinned
end

"""
    thin_samples(samples::GriddyPosteriors,nthin::Int)
Function to thin the posterior samples, helping remove autocorrelated values from the samples.
This implementation is for the continuous sampler.

---
Keyword arguments
* `samples::GriddyPosteriors` Struct storing the posterior samples.
* `nthin::Int` The number of samples to skip.

---
Returns
* `thinned::GriddyPosteriors` Struct containing the thinned posterior samples.

---
Details
This function selects retains every `nthin`th sample from the posterior samples, starting at the first index and going through to the end.
"""
function thin_samples(samples::GriddyPosteriors,nthin::Int)
    if nthin >= length(samples.tau2)
        error("The number of specified to burn is greater than or equal to the total number of samples.
        Please specify a number of samples to burn that is less than the total number of samples.
        Total number of samples = $(length(samples.tau2))
        Specified thinning = $nthin")
    end

    retained = 1:nthin:length(samples.tau2)
    if length(retained) < 100
        @warn "The specified thinning rate, $nthin, seems large for the number of posterior samples.
        Total number of samples =  $(length(samples.tau2))
        Number of samples retained = $(length(retained))
        Number of samples discarded = $(length(samples.tau2) - length(retained))
        Consider using a smaller thinning rate or drawing more posterior samples."
    end

    theta = samples.theta[retained,:]
    rho = samples.rho[retained]
    tau2 = samples.tau2[retained]
    sig_star2 = samples.sig_star2[retained]
    
    thinned = GriddyPosteriors(theta=theta,rho=rho,tau2=tau2,sig_star2=sig_star2)
    
    return thinned
end

"""
    sqrt_variance(samples::BulkVarsStruct)
Function to take the square root of the posterior samples of the variance parameters, making them standard deviations.
This implementation is for the continuous sampler posterior samples.

---
Keyword arguments
* `samples::BulkVarsStruct` Posterior samples from the continuous sampler.
"""
function sqrt_variance!(samples::BulkVarsStruct)
    samples.tau2 .= sqrt.(samples.tau2)
    samples.sig2 .= sqrt.(samples.sig2)
end

"""
    sqrt_variance(samples::GriddyPosteriors)
Function to take the square root of the posterior samples of the variance parameters, making them standard deviations.
This implementation is for the griddy Gibbs sampler posterior samples.

---
Keyword arguments
* `samples::GriddyPosteriors` Posterior samples from the griddy Gibbs sampler.
"""
function sqrt_variance!(samples::GriddyPosteriors)
    samples.tau2 .= sqrt.(samples.tau2)
    samples.sig_star2 .= sqrt.(samples.sig_star2)
end

"""
    posterior_hist!(data::Array{Float64},nbins::Int,var::String,iter::Int,save_plots::Bool,show_plots::Bool)
Function to generate a combined posterior histogram and kernel density plot for the posterior samples of a variable.
Optionally, display and/or save the plot.

---
Keyword arguments
* `data::Array{Float64}` Array of posterior samples to plot.
* `nbins::Int` The number of bins to use for the histogram.
* `var::String` A string specifying the name of the variable being plotted.
* `iter::Int` An Integer specifying the iterate of a multidimensional variable, e.g. θ.
  * for a univariate variable, e.g. τ^2, specify 0.
* `save_plots::Bool` Indicator of whether to save the plot that is generated.
* `show_plots::Bool` Indicator of whether to display the plot that is generated.
"""
function posterior_hist!(data::Array{Float64},nbins::Int,var::String,iter::Int,
    save_plots::Bool,show_plots::Bool)

    params = fit(Histogram,data,nbins=nbins)
    if (length(params.edges)-1) != length(params.weights)
        error("There was an error in the bin and count calculations for the posterior histograms.
        Please try using a different number of bins.")
    end
    areas = [(params.edges[1][i+1]-params.edges[1][i])*
        params.weights[i] for i in eachindex(params.weights)]
    p = Plots.bar(params.edges,params.weights/sum(areas),label=false)
    StatsPlots.density!(data,label=false,lw=5)
    if iter == 0
        title!(LaTexString("\$"*"\\"*"$(var)\$ Posterior Distribution"))
        xlabel!(LaTexString("\$"*"\\"*"$(var)\$"))
        ylabel!(LaTexString("\$p("*"\\"*"$(var)|y)\$"))
    else    
        title!(LaTexString("\$"*"\\"*"$(var)_{$iter}\$ Posterior Distribution"))
        xlabel!(LaTexString("\$"*"\\"*"$(var)_{$iter}\$"))
        ylabel!(LaTexString("\$p("*"\\"*"$(var)_{$iter}|y)\$"))
    end
    save_plots ? Plots.savefig(p,"$var-posterior_dist.png") : nothing
    show_plots ? Plots.display(p) : nothing
end

"""

"""
function plot_correlation(data::Array{Float64},var::String,save_plots::Bool,show_plots::Bool)
    labs = [LaTeXString("\$\\$var_{$i}\$") for i in axes(data)[2]]
    p = StatsPlots.corrplot(data,label=labs)
    title!(LaTexString("\$"*"\\"*"$(var)\$ Correlation"))

    save_plots ? Plots.savefig(p,"$var-correlation.png") : nothing
    show_plots ? Plots.display(p) : nothing
end

"""
    trace_plot!(data::Array{Float64},var::String,iter::Int,save_plots::Bool,show_plots::Bool)
Function to generate a combined posterior histogram and kernel density plot for the posterior samples of a variable.
Optionally, display and/or save the plot.

---
Keyword arguments
* `data::Array{Float64}` Array of posterior samples to plot.
* `var::String` A string specifying the name of the variable being plotted.
* `iter::Int` An Integer specifying the iterate of a multidimensional variable, e.g. θ.
  * for a univariate variable, e.g. τ^2, specify 0.
* `save_plots::Bool` Indicator of whether to save the plot that is generated.
* `show_plots::Bool` Indicator of whether to display the plot that is generated.
"""
function trace_plot!(data::Array{Float64},var::String,iter::Int,
    save_plots::Bool,show_plots::Bool)

    p = Plots.plot(data,label=false)
    if iter == 0
        title!(LaTexString("\$"*"\\"*"$(var)\$ Trace Plot"))
        ylabel!(LaTexString("\$"*"\\"*"$(var)\$ Draw Value"))
        xlabel!(LaTexString("\$"*"\\"*"$(var)\$ Iteration"))
    else    
        title!(LaTexString("\$"*"\\"*"$(var)_{$iter}\$ Trace Plot"))
        ylabel!(LaTexString("\$"*"\\"*"$(var)_{$iter}\$ Draw Value"))
        xlabel!(LaTexString("\$"*"\\"*"$(var)_{$iter}\$ Iteration"))
    end
    save_plots ? Plots.savefig(p,"$var-trace_plot.png") : nothing
    show_plots ? Plots.display(p) : nothing
end

"""
    plot_credible_bounds(data::Array{Float64},x_coords::Array{Float64})
Function to generate a plot of a function over the x coordinates of the data along with its 95% credible bounds.

---
Keyword arguments
* `data::Array{Float64}` Function's data to plot.
* `x_coords::Array{Float64}` the x coordinates of the data.

---
Returns
* `p` The generated plot.
"""
function plot_credible_bounds(data::Array{Float64},x_coords::Array{Float64})
    ticks = Vector{String}(undef,size(x_coords)[1])
    for i in axes(x_coords)[1]
        temp = [LaTeXString("x_{$j}") for j in axes(x_coords)[2]]
        ticks[i] = join(temp,";\n")
    end
    p = StatsPlots.errorline(data',errorstyle=:plume,label=false)
    StatsPlots.errorline!(data',errorstyle=:ribbon,errortype=:percentile,percentiles=[2.5,97.5])
    xticks!(axes(x_coords)[1],ticks)
    xlabel!("Control Variable Coordinates")
    return p
end

"""
    plot_disc!(delta::Array{Float64},x_coords::Array{Float64},nbins::Int,show_plots::Bool,save_plots::Bool)
Function to generate plots for the discrepancy function.

---
Keyword arguments
* `delta::Array{Float64}` Array containing the posterior samples of the discrepancy function, each row of the Matrix is a sample of δ.
* `x_coords::Array{Float64}` Array containing the x coordinates from the experimental data.
* `nbins::Int` The number of bins to use for the histograms.
* `save_plots::Bool` Indicator of whether to save the plot that is generated.
* `show_plots::Bool` Indicator of whether to display the plot that is generated.
"""
function plot_disc!(delta::Array{Float64},x_coords::Array{Float64},
    nbins::Int,save_plots::Bool,show_plots::Bool)

    p = plot_creidlbe_bounds(delta,x_coords)
    title!("Discrepancy Function")
    ylabel!(LaTeXString("\\delta(\\mathbf{x})"))

    save_plots ? Plots.savefig(p,"$var-discrepancy_function.png") : nothing
    show_plots ? Plots.display(p) : nothing

    posterior_hist!(delta',nbins,"delta",0,save_plots,show_plots)
    trace_plot!(delta',"delta",0,save_plots,show_plots)

end

"""
    plot_prediction!(thetas::Array{Float64},data::BulkVarsStruct,x_coords::Array{Float64},scales::Scaling)
Function to plot the surrogate model and data model estimations.

---
* `thetas::Array{Float64}` Array of the posterior samples of theta, normalized to the [0,1] interval.
* `samples::BulkVarsStruct` Struct containing non-normalized posterior samples.
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `scales::Scaling` Struct containing the minimum and maximum values for each variable.
* `save_plots::Bool` Indicator of whether to save the plot that is generated.
* `show_plots::Bool` Indicator of whether to display the plot that is generated.
"""
function plot_prediction!(thetas::Array{Float64},samples::BulkVarsStruct,data::DataStr,
    scales::Scaling,save_plots::Bool,show_plots::Bool)
    
    exp_resp = normalize_var(data.exp.y,scales.y)

    response = Array{Float64}(undef,size(thetas)[1],size(x_coords)[1])
    for i in eachindex(response)
        response[i,:] .= predict_y_all(thetas[i,:])
    end

    response .= normalize_var(response,scales.y)

    p = plot_credible_bounds(response,data.exp.x)
    if length(size(exp_resp)) > 1
        Plots.scatter!(exp_resp',color=4,label=false)
    else
        Plots.scatter!(exp_resp,color=4,label=false)
    end
    title!("Surrogate Model Prediction")
    ylabel!(LaTeXString("\\eta(\\mathbf{x})"))

    save_plots ? Plots.savefig(p,"calibrated_surrogate_model.png") : nothing
    show_plots ? Plots.display(p) : nothing

    p = plot_credible_bounds(response .+ samples.delta .+ samples.tau2,data.exp.x)
    if length(size(exp_resp)) > 1
        Plots.scatter!(exp_resp',color=4,label=false)
    else
        Plots.scatter!(exp_resp,color=4,label=false)
    end
    title!("Data Model Prediction")
    ylabel!(LaTeXString("\\eta(\\mathbf{x})+\\delta(\\mathbf{x})"))

    save_plots ? Plots.savefig(p,"calibrated_discrepancy_model.png") : nothing
    show_plots ? Plots.display(p) : nothing

    if length(size(data.exp.y)) > 1
        error = response .+ samples.delta .- mean(exp_resp)
    else
        error = response .+ samples.delta .- exp_resp
    end

    p = plot_credible_bounds(error,data.exp.x)
    title!("Calibrated Model Error Estimation")
    ylabel!(LaTeXString("y(\\mathbf{x}-\\eta(\\mathbf{x})+\\delta(\\mathbf{x})"))

    save_plots ? Plots.savefig(p,"calibrated_model_error.png") : nothing
    show_plots ? Plots.display(p) : nothing
end

"""
    plot_prediction!(thetas::Array{Float64},data::GriddyPosteriors,x_coords::Array{Float64},scales::Scaling)
Function to plot the surrogate model and data model estimations.

---
* `thetas::Array{Float64}` Array of the posterior samples of theta, normalized to the [0,1] interval.
* `samples::GriddyPosteriors` Struct containing non-normalized posterior samples.
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `scales::Scaling` Struct containing the minimum and maximum values for each variable.
* `save_plots::Bool` Indicator of whether to save the plot that is generated.
* `show_plots::Bool` Indicator of whether to display the plot that is generated.
"""
function plot_prediction!(thetas::Array{Float64},samples::GriddyPosteriors,data::DataStr,
    scales::Scaling,save_plots::Bool,show_plots::Bool)
    
    exp_resp = normalize_var(data.exp.y,scales.y)

    response = Array{Float64}(undef,size(thetas)[1],size(x_coords)[1])
    for i in eachindex(response)
        response[i,:] .= predict_y_all(thetas[i,:])
    end

    response .= normalize_var(response,scales.y)

    p = plot_credible_bounds(response,x_coords)
    if length(size(exp_resp)) > 1
        Plots.scatter!(exp_resp',color=4,label="Experimental Observations")
    else
        Plots.scatter!(exp_resp,color=4,label="Experimental Observations")
    end
    title!("Surrogate Model Prediction")
    ylabel!(LaTeXString("\\eta(\\mathbf{x})"))

    save_plots ? Plots.savefig(p,"calibrated_surrogate_model.png") : nothing
    show_plots ? Plots.display(p) : nothing

    if length(size(data.exp.y)) > 1
        error = response .+ samples.delta .- mean(exp_resp)
    else
        error = response .+ samples.delta .- exp_resp
    end

    p = plot_credible_bounds(error,data.exp.x)
    title!("Calibrated Model Error Estimation")
    ylabel!(LaTeXString("y(\\mathbf{x}-\\eta(\\mathbf{x})+\\delta(\\mathbf{x})"))

    save_plots ? Plots.savefig(p,"calibrated_model_error.png") : nothing
    show_plots ? Plots.display(p) : nothing
end

"""
    get_estimates(data::Vector{Float64})
Function to get the mean, median, and 95% credible bounds of a set of posterior samples.

---
Keyword arguments
* `data::Vector{Float64}`

---
Returns
* `temp::Vector{Float64}` Vector containing the estimates.
"""
function get_estimates(data::Vector{Float64})
    temp = Vector{Float64}(undef,4)
    temp[1] = quantile(data,0.025)
    temp[2] = median(data)
    temp[3] = mean(data)
    temp[4] = quantile(data,0.975)
    return temp
end

"""
    make_estimate_table(samples::BulkVarsStruct,nx::Int,ntheta::Int)
Function to make the table of estimates from the posterior samples.

---
Keyword arguments
* `samples::BulkVarsStruct` struct containing the posterior samples from the continuous sampler.
* `nx::Int` number of x variables.
* `ntheta::Int` number of theta variables.
"""
function make_estimate_table(samples::BulkVarsStruct,nx::Int,ntheta::Int)
    if nx > 0
        estimates = Array{Float64}(undef,4,ntheta+nx+3)
        for i in 1:nx
            estimates[:,i+1] = get_estimates(samples.rho[:,i])
        end
        estimates[:,end-1] = get_estimates(samples.sig2)
        var_labs = vcat(["Value"],[LaTeXString("\\rho_{$i}") for i in 1:nx],
            [LaTeXString("\\theta_{$i}") for j in 1:ntheta],
            [LaTeXString("\\sigma"),LaTeXString("\\tau")])
    else
        estimates = Array{Float64}(undef,4,ntheta+3)
        var_labs = vcat(["Value"],[LaTeXString("\\theta_{$i}") for j in 1:ntheta],
            [LaTeXString("\\tau")])
    end
    for i in 1:ntheta
        estimates[:,i+nx+1] = get_estimates(samples.theta[:,i])
    end
    estimates[:,end] = get_estimates(samples.tau2)
    param_labs = ["95% Lower Bound","Median","Mean","95% Upper Bound"]
    
    estimates = string.(round(estimates,digits=5))
    estimates[:,1] = param_labs

    pretty_table(estimates;header=var_lab)

    open("param_estimates.txt","w") do file
        pretty_table(file,estimates;header=var_lab)
    end
end

"""
    make_estimate_table!(samples::GriddyPosteriors,nx::Int,ntheta::Int)
Function to make the table of estimates from the posterior samples.

---
Keyword arguments
* `samples::GriddyPosteriors` struct containing the posterior samples from the griddy Gibbs sampler.
* `nx::Int` number of x variables.
* `ntheta::Int` number of theta variables.
"""
function make_estimate_table!(samples::GriddyPosteriors,nx::Int,ntheta::Int)
    if nx > 0
        estimates = Array{Float64}(undef,4,ntheta+4)
        estimates[:,2] = get_estimates(samples.rho)
        estimates[:,end-1] = get_estimates(samples.sig_star2)
        var_labs = vcat(["Value"],[LaTeXString("\\rho")],
            [LaTeXString("\\theta_{$i}") for j in 1:ntheta],
            [LaTeXString("\\sigma^{*}"),LaTeXString("\\tau")])
    else
        estimates = Array{Float64}(undef,4,ntheta+3)
        var_labs = vcat(["Value"],[LaTeXString("\\theta_{$i}") for j in 1:ntheta],
            [LaTeXString("\\tau")])
    end
    for i in 1:ntheta
        estimates[:,i+nx+1] = get_estimates(samples.theta[:,i])
    end
    estimates[:,end] = get_estimates(samples.tau2)
    param_labs = ["95% Lower Bound","Median","Mean","95% Upper Bound"]
    
    estimates = string.(round(estimates,digits=5))
    estimates[:,1] = param_labs

    pretty_table(estimates;header=var_lab)

    open("param_estimates.txt","w") do file
        pretty_table(file,estimates;header=var_lab)
    end
end

"""
    post_process(samples::BulkVarsStruct,nx::Int,ntheta::Int,nburn::Int,scales::Scaling,make_plots::Bool,save_plots::Bool,show_plots::Bool)
    post_process(samples::BulkVarsStruct,nx::Int,ntheta::Int,nburn::Int,scales::Scaling,make_plots::Bool,save_plots::Bool,show_plots::Bool,nbins::Int,nthin::Int)
Wrapper function to post-process the posterior samples.

---
Keyword arguments
* `samples::BulkVarsStruct` Struct containing the posterior samples.
* `nx::Int` Number of x dimensions.
* `ntheta::Int` number of θ dimensions.
* `nburn::Int` The number of samples to burn.
* `make_plots::Bool` Indicator of whether to make plots.
* `save_plots::Bool` Indicator of whether to save the plots.
* `show_plots::Bool` Indicator of whether to show the plots.
Optional arguments
* `nthin::Int` The number of samples to skip when thinning.
  * default value of 20
* `nbins::Int` The number of bins to use for histograms.
  * default value of 30
"""
function post_process(samples::BulkVarsStruct,nx::Int,ntheta::Int,nburn::Int,
    scales::Scaling,make_plots::Bool,save_plots::Bool,show_plots::Bool,
    nthin::Int=20,nbins::Int=30)

    samples = remove_burn(samples,nburn)
    samples = thin_samples(samples,nthin)

    scaled_samples = normalize_samples(samples,scales)

    sqrt_variance!(scaled_samples)

    if make_plots
        posterior_hist!(scaled_samples.tau2,nbins,"tau",0,save_plots,show_plots)
        trace_plot!(scaled_samples.tau2,"tau",0,save_plots,show_plots)
        posterior_hist!(scaled_samples.sig2,nbins,"sigma",0,save_plots,show_plots)
        trace_plot!(scaled_samples.sig2,"sigma",0,save_plots,show_plots)
        for theta in 1:ntheta
            posterior_hist!(scaled_samples.theta[:,theta],nbins,"theta",
                theta,save_plots,show_plots)
            trace_plot!(scaled_samples.theta[:,theta],"theta",
                theta,save_plots,show_plots)
        end
        for rho in 1:nx
            posterior_hist!(scaled_samples.rho[:,rho],nbins,"rho",rho,
                save_plots,show_plots)
            trace_plot!(scaled_samples.rho[:,rho],"rho",rho,
                save_plots,show_plots)
        end
        plot_disc!(scaled_samples.delta,data.exp.x,nbins,save_plots,show_plots)
        plot_prediction!(samples.theta,scaled_samples,data,scales,save_plots,show_plots)
    end
    make_estimate_table(scaled_samples,nx,ntheta)
end