#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
############################# Define Misc Functions ###############################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
"""
    normalize_samples(samples::BulkVarsStruct,scales::Scaling)
    normalize_samples(samples::BulkVarsStruct,scales::Scaling,rev::Bool)
Function to normalize or reverse the normalization of the posterior samples.
This is the implementation for the continuous sampler results.

---
Positional arguments
* `samples::BulkVarsStruct` Struct containing the posterior samples.
* `scales::Scaling` Struct containing the minimum and maximum values for each variables.
Optional arguments
* `rev::Bool` Indicator of whether to reverse the normalization or not.
  * default value of true.
"""
function normalize_samples(samples::BulkVarsStruct,scales::Scaling,rev::Bool=true)

    theta = rev ? unnormalize_var(samples.theta,scales.theta) : 
        normalize_var(samples.theta,scales.theta)
    tau2 = rev ? (unnormalize_var(sqrt.(samples.tau2),scales.y) .- scales.y.min).^2 :
        (normalize_var(sqrt.(samples.tau2 .+ scales.y.min),scales.y)).^2
    delta = rev ? unnormalize_var(samples.delta,scales.y) .- scales.y.min :
        normalize_var(samples.delta .+ scales.y.min,scales.y)
    eta = rev ? unnormalize_var(samples.eta,scales.y) :
        normalize_var(samples.eta,scales.y)
    sig2 = rev ? (unnormalize_var(sqrt.(samples.sig2),scales.y) .- scales.y.min).^2 :
        (normlize_var(sqrt.(samples.sig2 .+ scales.y.min),scales.y)).^2
    
    samples_norm = BulkVarsStruct(theta=theta,sig2=sig2,tau2=tau2,delta=delta,
        eta=eta,rho=samples.rho,accept=samples.accept,ratio=samples.ratio)
    
    return samples_norm
end

"""
    normalize_samples(samples::GriddyVarsStruct,scales::Scaling,theta_grid::Array{Float64},
    sig_grid::Array{Float64})
    normalize_samples(samples::GriddyVarsStruct,scales::Scaling,theta_grid::Array{Float64},
    sig_grid::Array{Float64},rev::Bool)
Function to normalize or reverse the normalization of the posterior samples.
This implementation is for the griddy Gibbs sampler.

---
Positional arguments
* `samples::GriddyVarsStruct` Struct containing the posterior samples (in index form) from the griddy Gibbs sampler.
* `scales::Scaling` Struct containing the minimum and maximum values for each variables.
* `theta_grid::Array{Float64}` Array containing the sampling grid for θ that was used for the griddy Gibbs sampler.
* `sig_grid::Array{Float64}` Array containing the sampling grid for the proportional covariance matrix used for the griddy Gibbs sampler.
Optional arguments
* `rev::Bool` Indicator of whether to reverse the normalization or not.
  * default value of true.

---
Returns
* `converted_samples::GriddyPosteriors` A struct containing the real value samples from the griddy Gibbs sampler.
"""
function normalize_samples(samples::GriddyVarsStruct,scales::Scaling,theta_grid::Array{Float64},
    sig_grid::Array{Float64},rev::Bool=true)

    theta = rev ? unnormalize_var(theta_grid[samples.theta],scales.theta) :
        normalize_var(design[samples.theta],scales.theta)
    sig2 = rev ? (unnormalize_var(sqrt.(samples.sig2),scales.y) .- scales.y.min).^2 :
    (normlize_var(sqrt.(samples.sig2 .+ scales.y.min),scales.y)).^2
    rho = sig_grid[samples.rho,1,1]
    sig_star2 = sig_grid[1,samples.sig_star2,2] .* sig2

    converted_samples = GriddyPosteriors(theta=theta,sig2=sig2,rho=rho,sig_star2=sig_star2)

    return converted_samples
end

"""
    remove_burn(samples::BulkVarsStruct,nburn::Int)
Function to extract the posterior samples after the burned values.
Implementation for the continuous sampler results.

---
Positional arguments
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
    if nburn >= length(samples.tau2)
        error("The number of specified to burn is greater than or equal to the total number of samples.
        Please specify a number of samples to burn that is less than the total number of samples.
        Total number of samples = $(length(samples.tau2))
        Specified burn = $nburn")
    end
    keep = nburn + 1
    retained = keep:length(samples.tau2)
    theta = samples.theta[retained,:]
    rho = samples.rho[retained,:]
    sig2 = samples.sig2[retained]
    tau2 = samples.tau2[retained]
    delta = samples.delta[retained,:]
    eta = samples.eta[retained,:]
    accept = samples.accept[retained,:]
    ratio = samples.ratio[retained,:]

    truncated = BulkVarsStruct(theta=theta,rho=rho,sig2=sig2,tau2=tau2,
    delta=delta,eta=eta,accept=accept,ratio=ratio)
    
    return truncated
end

"""
    remove_burn(samples::GriddyPosteriors,nburn::Int)
Function to extract the posterior samples after the burned values.
Implementation for the griddy Gibbs sampler results.

---
Positional arguments
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
    if nburn >= length(samples.sig2)
        error("The number of specified to burn is greater than or equal to the total number of samples.
        Please specify a number of samples to burn that is less than the total number of samples.
        Total number of samples = $(length(samples.sig2))
        Specified burn = $nburn")
    end
    keep = nburn + 1
    retained = keep:length(samples.sig2)
    theta = samples.theta[retained,:]
    sig2 = samples.sig2[retained]
    sig_star2 = samples.sig_star2[retained]
    rho = samples.rho[retained]

    truncated = GriddyPosteriors(theta=theta,sig2=sig2,sig_star2=sig_star2,
        rho=rho)

    return truncated
end

"""
    remove_burn(samples::GriddyVarsStruct,nburn::Int)
Function to extract the posterior samples after the burned values.
Implementation for the griddy Gibbs sampler results, still in index form.

---
Positional arguments
* `samples::GriddyVarsStruct` Struct containing the posterior samples from the griddy Gibbs sampler.
* `nburn::Int` The number of samples to discard

---
Returns
* `truncated::GriddyVarsStruct` Bulk vars struct that contains the samples retained after discarding the burned samples.

---
Details
This selects all posterior samples from `nburn` + 1 through the end of the samples.
These values are retained and placed into a new posterior samples struct, which is then returned.
"""
function remove_burn(samples::GriddyVarsStruct,nburn::Int)
    if nburn >= length(samples.sig2)
        error("The number of specified to burn is greater than or equal to the total number of samples.
        Please specify a number of samples to burn that is less than the total number of samples.
        Total number of samples = $(length(samples.sig2))
        Specified burn = $nburn")
    end
    keep = nburn + 1
    retained = keep:length(samples.sig2)
    theta = samples.theta[retained]
    sig2 = samples.sig2[retained]
    sig_star2 = samples.sig_star2[retained]
    rho = samples.rho[retained]

    truncated = GriddyVarsStruct(theta=theta,sig2=sig2,sig_star2=sig_star2,
        rho=rho)

    return truncated
end

"""
    thin_samples(samples::BulkVarsStruct,nthin::Int)
Function to thin the posterior samples, helping remove autocorrelated values from the samples.
This implementation is for the continuous sampler results.

---
Positional arguments
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
    rho = samples.rho[retained,:]
    sig2 = samples.sig2[retained]
    tau2 = samples.tau2[retained]
    delta = samples.delta[retained,:]
    eta = samples.eta[retained,:]
    accept = samples.accept[retained,:]
    ratio = samples.ratio[retained,:]

    thinned = BulkVarsStruct(theta=theta,rho=rho,sig2=sig2,tau2=tau2,
    delta=delta,eta=eta,accept=accept,ratio=ratio)
    
    return thinned
end

"""
    thin_samples(samples::GriddyPosteriors,nthin::Int)
Function to thin the posterior samples, helping remove autocorrelated values from the samples.
This implementation is for the griddy Gibbs sampler results.

---
Positional arguments
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
    rho = samples.rho[retained]
    sig2 = samples.sig2[retained]
    sig_star2 = samples.sig_star2[retained]
    
    thinned = GriddyPosteriors(theta=theta,rho=rho,sig2=sig2,sig_star2=sig_star2)
    
    return thinned
end

"""
    thin_samples(samples::GriddyVarsStruct,nthin::Int)
Function to thin the posterior samples, helping remove autocorrelated values from the samples.
This implementation is for the griddy Gibbs sampler, with the results still in index form.

---
Positional arguments
* `samples::GriddyVarsStruct` Struct storing the posterior samples.
* `nthin::Int` The number of samples to skip.

---
Returns
* `thinned::GriddyVarsStruct` Struct containing the thinned posterior samples.

---
Details
This function selects retains every `nthin`th sample from the posterior samples, starting at the first index and going through to the end.
"""
function thin_samples(samples::GriddyVarsStruct,nthin::Int)
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

    theta = samples.theta[retained]
    rho = samples.rho[retained]
    sig2 = samples.sig2[retained]
    sig_star2 = samples.sig_star2[retained]
    
    thinned = GriddyVarsStruct(theta=theta,rho=rho,sig2=sig2,sig_star2=sig_star2)
    
    return thinned
end

"""
    sqrt_variance(samples::BulkVarsStruct)
Function to take the square root of the posterior samples of the variance parameters, making them standard deviations.
This implementation is for the continuous sampler posterior samples.

---
Positional arguments
* `samples::BulkVarsStruct` Posterior samples from the continuous sampler.
"""
function sqrt_variance!(samples::BulkVarsStruct)
    samples.sig2 .= sqrt.(samples.sig2)
    samples.tau2 .= sqrt.(samples.tau2)
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
    samples.sig2 .= sqrt.(samples.sig2)
    samples.sig_star2 .= sqrt.(samples.sig_star2)
end

"""
    posterior_hist!(data::Array{Float64},nbins::Int,var::String,iter::Int,save_plots::Bool,
    show_plots::Bool,mdl_apnd::String)
Function to generate a combined posterior histogram and kernel density plot for the posterior samples of a variable.
Optionally, display and/or save the plot.

---
Positional arguments
* `data::Array{Float64}` Array of posterior samples to plot.
* `nbins::Int` The number of bins to use for the histogram.
* `var::String` A string specifying the name of the variable being plotted.
* `iter::Int` An Integer specifying the iterate of a multidimensional variable, e.g. θ.
  * for a univariate variable, e.g. τ^2, specify 0.
* `save_plots::Bool` Indicator of whether to save the plot that is generated.
* `show_plots::Bool` Indicator of whether to display the plot that is generated.
* `mdl_apnd::String` String to append to the front of the saved plot's file name.
"""
function posterior_hist!(data::Array{Float64},nbins::Int,var::String,iter::Int,
            save_plots::Bool,show_plots::Bool,mdl_apnd::String)

    if length(size(data)) == 1
        params = fit(Histogram,data,nbins=nbins)
        if (length(params.edges[1])-1) != length(params.weights)
            error("There was an error in the bin and count calculations for the posterior histograms.
            Please try using a different number of bins.
            Number of edges = $(length(params.edges))
            Number of weights = $(length(params.weights))")
        end
        areas = [(params.edges[1][i+1]-params.edges[1][i])*
            params.weights[i] for i in eachindex(params.weights)]
        p = Plots.bar(params.edges,params.weights/sum(areas),label=false,
            left_margin=5mm,top_margin=5mm,titleposition=[0.5,1.05])
        StatsPlots.density!(data,label=false,lw=5,trim=true)
    elseif length(size(data)) == 2
        params = [fit(Histogram,data[:,i],nbins=nbins) for i in axes(data)[2]]
        bounds = [collect(params[i].edges[1]) for i in eachindex(params)]
        bounds = reduce(vcat,bounds)
        if (length(params[1].edges[1])-1) != length(params[1].weights)
            error("There was an error in the bin and count calculations for the posterior histograms.
            Please try using a different number of bins.
            Number of edges = $(length(params.edges))
            Number of weights = $(length(params.weights))")
        end
        areas = [[(params[j].edges[1][i+1]-params[j].edges[1][i])*
            params[j].weights[i] for i in eachindex(params[j].weights)] for j in eachindex(params)]
        p = Plots.bar(params[1].edges,params[1].weights/sum(areas[1]),label=false,
        left_margin=5mm,top_margin=5mm,titleposition=[0.5,1.05])
        for i in 2:length(areas)
            Plots.bar!(params[i].edges,params[i].weights/sum(areas[i]),label=false)
        end
        for i in eachindex(areas)
            StatsPlots.density!(data[:,i],color=i,label=false,trim=true)
        end
        #println(minimum(bounds))
        #println(maximum(bounds))
        xlims!((0.95*minimum(bounds),1.05*maximum(bounds)))
    end

    if iter == 0
        title!(LaTeXString("\$"*"\\"*"$(var)\$ Posterior Distribution"))
        xlabel!(LaTeXString("\$"*"\\"*"$(var)\$"))
        ylabel!(LaTeXString("\$p("*"\\"*"$(var)|y)\$"))
    else    
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Posterior Distribution"))
        xlabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$"))
        ylabel!(LaTeXString("\$p("*"\\"*"$(var)_{$iter}|y)\$"))
    end
    save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-posterior_dist.png") : nothing
    show_plots ? Plots.display(p) : nothing
end

"""
    plot_correlation(data::Array{Float64,2},var::String,save_plots::Bool,show_plots::Bool,
    mdl_apnd::String)
Function to generate a correlation plot for the samples of a multidimensional variable.
Optionally, display and/or save the plot.

---
Positional arguments
* `data::Array{Float64,2}` Data to plot.
* `var::String` Name of the variable being plotted.
* `save_plots::Bool` Indicator of whether to save the plot that is generated.
* `show_plots::Bool` Indicator of whether to display the plot that is generated.
* `mdl_apnd::String` String to append to the front of the saved plot's file name.
"""
function plot_correlation(data::Array{Float64,2},var::String,save_plots::Bool,
            show_plots::Bool,mdl_apnd::String)

    labs = [LaTeXString("\$\\$(var)_{$i}\$") for i in axes(data)[2]]
    p = StatsPlots.corrplot(data,label=labs)
    title!(LaTeXString("\$"*"\\"*"$(var)\$ Correlation"))

    save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-correlation.png") : nothing
    show_plots ? Plots.display(p) : nothing
end

"""
    trace_plot!(data::Array{Float64},var::String,iter::Int,save_plots::Bool,
    show_plots::Bool,mdl_apnd::String)
Function to generate a combined posterior histogram and kernel density plot for the posterior samples of a variable.
Optionally, display and/or save the plot.

---
Positional arguments
* `data::Array{Float64}` Array of posterior samples to plot.
* `var::String` A string specifying the name of the variable being plotted.
* `iter::Int` An Integer specifying the iterate of a multidimensional variable, e.g. θ.
  * for a univariate variable, e.g. τ^2, specify 0.
* `save_plots::Bool` Indicator of whether to save the plot that is generated.
* `show_plots::Bool` Indicator of whether to display the plot that is generated.
* `mdl_apnd::String` String to append to the front of the saved plot's file name.
"""
function trace_plot!(data::Array{Float64},var::String,iter::Int,
    save_plots::Bool,show_plots::Bool,mdl_apnd::String)

    p = Plots.plot(data,label=false,
    left_margin=5mm,top_margin=5mm,titleposition=[0.5,1.05])
    if iter == 0
        title!(LaTeXString("\$"*"\\"*"$(var)\$ Trace Plot"))
        ylabel!(LaTeXString("\$"*"\\"*"$(var)\$ Draw Value"))
        xlabel!(LaTeXString("\$"*"\\"*"$(var)\$ Iteration"))
    else    
        title!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Trace Plot"))
        ylabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Draw Value"))
        xlabel!(LaTeXString("\$"*"\\"*"$(var)_{$iter}\$ Iteration"))
    end
    save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-trace_plot.png") : nothing
    show_plots ? Plots.display(p) : nothing
end

"""
    plot_credible_bounds(data::Array{Float64},x_coords::Array{Float64},scales::ScalePar)
Function to generate a plot of a function over the x coordinates of the data along with its 95% credible bounds.

---
Positional arguments
* `data::Array{Float64}` Function's data to plot.
* `x_coords::Array{Float64}` the x coordinates of the data.
* `scales::ScalePar` Struct containing the minimum and maximum value of the variable in `data`.

---
Returns
* `p` The generated plot.
"""
function plot_credible_bounds(data::Array{Float64},x_coords::Array{Float64},scales::ScalePar)
    x_coords = round.(unnormalize_var(x_coords,scales),digits=4)
    println(size(data))
    ticks = Vector{String}(undef,size(x_coords)[1])
    for i in axes(x_coords)[1]
        temp = [LaTeXString("\$x_{$j}=$(x_coords[i,j])\$") for j in axes(x_coords)[2]]
        ticks[i] = join(temp,";\n")
    end
    p = StatsPlots.errorline(permutedims(data),errorstyle=:plume,label="Evaluations",secondarylinealpha=0.2,
        right_margin=4mm)
    StatsPlots.errorline!(permutedims(data),errorstyle=:ribbon,errortype=:percentile,
        percentiles=[2.5,97.5],label=false,secondarylinealpha=0.2,right_margin=4mm)
    #xticks!(axes(x_coords)[1],ticks)
    xlabel!("Control Variable Indices")
    return p
end

"""
    plot_disc!(delta::Array{Float64},x_coords::Array{Float64},scales::Scaling,
    show_plots::Bool,save_plots::Bool,mdl_apnd::String)
Function to generate plots for the discrepancy function.
This implementation does not generate a histogram or trace plot.

---
Positional arguments
* `delta::Array{Float64}` Array containing the posterior samples of the discrepancy function, each row of the Matrix is a sample of δ.
* `x_coords::Array{Float64}` Array containing the x coordinates from the experimental data.
* `scales::Scaling` Struct containing scaling information for the variables in the data.
* `save_plots::Bool` Indicator of whether to save the plot that is generated.
* `show_plots::Bool` Indicator of whether to display the plot that is generated.
* `mdl_apnd::String` String to append to the front of the saved plot's file name.
"""
function plot_disc!(delta::Array{Float64},x_coords::Array{Float64},scales::Scaling,
            save_plots::Bool,show_plots::Bool,mdl_apnd::String)

    p = plot_credible_bounds(delta,x_coords,scales.x)
    title!("Discrepancy Function")
    ylabel!(LaTeXString("\$\\delta(\\mathbf{x})\$"))

    save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-discrepancy_function.png") : nothing
    show_plots ? Plots.display(p) : nothing
end

"""
    plot_disc!(delta::Array{Float64},x_coords::Array{Float64},scales::Scaling,nbins::Int,
    show_plots::Bool,save_plots::Bool,mdl_apnd::String)
Function to generate plots for the discrepancy function.
This implementation generates a histogram and a trace plot.

---
Positional arguments
* `delta::Array{Float64}` Array containing the posterior samples of the discrepancy function, each row of the Matrix is a sample of δ.
* `x_coords::Array{Float64}` Array containing the x coordinates from the experimental data.
* `scales::Scaling` Struct containing scaling information for the variables in the data.
* `nbins::Int` The number of bins to use for the histograms.
* `save_plots::Bool` Indicator of whether to save the plot that is generated.
* `show_plots::Bool` Indicator of whether to display the plot that is generated.
* `mdl_apnd::String` String to append to the front of the saved plot's file name.
"""
function plot_disc!(delta::Array{Float64},x_coords::Array{Float64},scales::Scaling,
            nbins::Int,save_plots::Bool,show_plots::Bool,mdl_apnd::String)

    p = plot_credible_bounds(delta,x_coords,scales.x)
    title!("Discrepancy Function")
    ylabel!(LaTeXString("\$\\delta(\\mathbf{x})\$"))

    save_plots ? Plots.savefig(p,"$(mdl_apnd)_$var-discrepancy_function.png") : nothing
    show_plots ? Plots.display(p) : nothing

    posterior_hist!(permutedims(delta),nbins,"delta",0,save_plots,show_plots,mdl_apnd)
    trace_plot!((delta),"delta",0,save_plots,show_plots,mdl_apnd)

end

"""
    plot_prediction!(samples::BulkVarsStruct,data::DataStr,scales::Scaling,
    save_plots::Bool,show_plots::Bool,mdl_apnd::String)
Function to plot the surrogate model and data model estimations.

---
Positional arguments
* `samples::BulkVarsStruct` Struct containing non-normalized posterior samples.
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `scales::Scaling` Struct containing the minimum and maximum values for each variable.
* `save_plots::Bool` Indicator of whether to save the plot that is generated.
* `show_plots::Bool` Indicator of whether to display the plot that is generated.
* `mdl_apnd::String` String to append to the front of the saved plot's file name.
"""
function plot_prediction!(samples::BulkVarsStruct,data::DataStr,
            scales::Scaling,save_plots::Bool,show_plots::Bool,mdl_apnd::String)
    
    exp_resp = unnormalize_var(data.exp.y,scales.y)

    response = samples.eta

    p = plot_credible_bounds(response,data.exp.x,scales.x)
    Plots.scatter!(exp_resp[:,1],color=8,label="Experiments")
    Plots.scatter!(exp_resp[:,2:end],color=8,label=false)
    title!("Calibrated Surrogate Model")
    ylabel!(LaTeXString("\$\\eta (\\mathbf{x})\$"))

    save_plots ? Plots.savefig(p,"$(mdl_apnd)_calibrated_surrogate_model.png") : nothing
    show_plots ? Plots.display(p) : nothing

    
    p = plot_credible_bounds(response .+ samples.delta,data.exp.x,scales.x)
    Plots.scatter!(exp_resp[:,1],color=8,label="Experiments")
    Plots.scatter!(exp_resp[:,2:end],color=8,label=false)
    title!("Data Model Samples")
    ylabel!(LaTeXString("\$\\eta (\\mathbf{x})+\\delta (\\mathbf{x})\$"))

    save_plots ? Plots.savefig(p,"$(mdl_apnd)_calibrated_discrepancy_model.png") : nothing
    show_plots ? Plots.display(p) : nothing

    mean_eta = vec(mean(response,dims=1))
    eta_bounds = Array{Float64}(undef,2,size(response)[2])
    delta_bounds = similar(eta_bounds)
    tau_bound = quantile(samples.sig2,[0.975])[1]
    epsilon_bounds = quantile(Normal(0,tau_bound),[0.025,0.975])
    for i in axes(response)[2]
        eta_bounds[:,i] = quantile(response[:,i],[0.025,0.975])
        delta_bounds[:,i] = quantile(samples.delta[:,i],[0.025,0.975])
    end
    lb = eta_bounds[1,:] .+ delta_bounds[1,:] .+ epsilon_bounds[1]
    ub = eta_bounds[2,:] .+ delta_bounds[2,:] .+ epsilon_bounds[2]
    lb = vec(lb)
    ub = vec(ub)
    p = Plots.plot(mean_eta,label="Mean Surrogate Response")
    Plots.scatter!(exp_resp[:,1],color=8,label="Experiments")
    Plots.scatter!(exp_resp[:,2:end],color=8,label=false)
    Plots.plot!(lb,fillrange=ub,label="Uncertainty Bounds",alpha=0.2,lw=1)
    Plots.plot!(hcat(lb,ub),label=false,lw=1,color=3)
    title!("Estimated Uncertainty Bounds")
    xlabel!("Control Variable Indices")
    ylabel!(LaTeXString("\$y(\\mathbf{x})\$"))
    
    save_plots ? Plots.savefig(p,"$(mdl_apnd)_uncertainty.png") : nothing
    show_plots ? Plots.display(p) : nothing



    if size(data.exp.y)[2] > 1
        error = response .+ samples.delta .- mean(exp_resp,dims=2)'
    else
        error = response .+ samples.delta .- exp_resp'
    end

    p = plot_credible_bounds(error,data.exp.x,scales.x)
    title!("Data Model Mean Error Estimation")
    ylabel!(LaTeXString("\$\\bar{y}(\\mathbf{x})-[\\eta(\\mathbf{x})+\\delta(\\mathbf{x})]\$"))


    save_plots ? Plots.savefig(p,"$(mdl_apnd)_calibrated_model_error.png") : nothing
    show_plots ? Plots.display(p) : nothing
end

"""
    plot_prediction!(theta_idx::Vector{Int},samples::GriddyPosteriors,x_coords::Array{Float64},
    grid_resp::Array{Float64,2},data::DataStr,scales::Scaling,save_plots::Bool,
    show_plots::Bool,mdl_apnd::String)
Function to plot the surrogate model and data model estimations.

---
Positional arguments
* `theta_idx::Vector{Int}` Sampled indices of theta from the griddy Gibbs sampler.
* `samples::GriddyPosteriors` Struct containing non-normalized posterior samples.
* `grid_resp::Array{Float64,2}` Matrix of grid responses.
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `scales::Scaling` Struct containing the minimum and maximum values for each variable.
* `save_plots::Bool` Indicator of whether to save the plot that is generated.
* `show_plots::Bool` Indicator of whether to display the plot that is generated.
* `mdl_apnd::String` String to append to the front of the saved plot's file name.
"""
function plot_prediction!(theta_idx::Vector{Int},samples::GriddyPosteriors,
            grid_resp::Array{Float64,2},data::DataStr,scales::Scaling,
            save_plots::Bool,show_plots::Bool,mdl_apnd::String)
    
    nloc = size(data.exp.y)[1]
    nx = size(data.exp.x)[2]
    exp_resp = unnormalize_var(data.exp.y,scales.y)

    response = unnormalize_var(grid_resp[theta_idx,:],scales.y)

    p = plot_credible_bounds(response,data.exp.x,scales.x)
    Plots.scatter!(exp_resp[:,1],color=8,label="Experiments")
    Plots.scatter!(exp_resp[:,2:end],color=8,label=false)
    title!("Calibrated Surrogate Model")
    ylabel!(LaTeXString("\$\\eta (\\mathbf{x})\$"))

    save_plots ? Plots.savefig(p,"$(mdl_apnd)_calibrated_surrogate_model.png") : nothing
    show_plots ? Plots.display(p) : nothing

    mean_eta = vec(mean(response,dims=1))
    eta_bounds = Array{Float64}(undef,2,nloc)
    error_bounds = similar(eta_bounds)
    error_samples = Array{Float64}(undef,size(response)[1],nloc)
    sigma = Array{Float64}(undef,nloc,nloc)
    identity = Matrix(1.0I,nloc,nloc)
    for i in axes(error_samples)[1]
        rho_vec = repeat([samples.rho[i]],nx)
        corr = correlation_construct(rho_vec,data.exp.x,nx,nloc)
        sigma .= samples.sig2[i] .* (identity) .+ samples.sig_star2[i] .*
            corr
        error_samples[i,:] .= rand(MvNormal(sigma),1)
    end

    for i in axes(response)[2]
        eta_bounds[:,i] = quantile(response[:,i],[0.025,0.975])
        error_bounds[:,i] = quantile(error_samples[:,i],[0.025,0.975])
    end
    lb = eta_bounds[1,:] .+ error_bounds[1,:]
    ub = eta_bounds[2,:] .+ error_bounds[2,:]
    lb = vec(lb)
    ub = vec(ub)
    p = Plots.plot(mean_eta,label="Mean Surrogate Response")
    Plots.scatter!(exp_resp[:,1],color=8,label="Experiments")
    Plots.scatter!(exp_resp[:,2:end],color=8,label=false)
    Plots.plot!(lb,fillrange=ub,label="Uncertainty Bounds",alpha=0.2,lw=1)
    Plots.plot!(hcat(lb,ub),label=false,lw=1,color=3)
    title!("Estimated Uncertainty Bounds")
    xlabel!("Control Variable Indices")
    ylabel!(LaTeXString("\$y(\\mathbf{x})\$"))
    
    save_plots ? Plots.savefig(p,"$(mdl_apnd)_uncertainty.png") : nothing
    show_plots ? Plots.display(p) : nothing


    if size(data.exp.y)[2] > 1
        error = response .- mean(exp_resp,dims=2)'
    else
        error = response .- exp_resp'
    end

    p = plot_credible_bounds(error,data.exp.x,scales.x)
    title!("Data Model Mean Error Estimation")
    ylabel!(LaTeXString("\$y(\\mathbf{x})-\\eta(\\mathbf{x})\$"))
    #if size(exp_resp)[2] > 1
    #    Plots.scatter!(exp_resp .- mean(exp_resp,dims=2),color=8,label=false)
    #end

    save_plots ? Plots.savefig(p,"$(mdl_apnd)_calibrated_model_error.png") : nothing
    show_plots ? Plots.display(p) : nothing
end

"""
    get_estimates(data::Vector{Float64})
Function to get the mean, median, and 95% credible bounds of a set of posterior samples.

---
Positional arguments
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
    make_estimate_table(samples::BulkVarsStruct,nx::Int,ntheta::Int,mdl_apnd::String)
Function to make the table of estimates from the posterior samples.

---
Positional arguments
* `samples::BulkVarsStruct` struct containing the posterior samples from the continuous sampler.
* `nx::Int` number of x variables.
* `ntheta::Int` number of theta variables.
* `mdl_apnd::String` String to append to the front of the saved plot's file name.
"""
function make_estimate_table(samples::BulkVarsStruct,nx::Int,ntheta::Int,mdl_apnd::String)
    if nx > 0
        estimates = Array{Float64}(undef,4,ntheta+nx+3)
        for i in 1:nx
            estimates[:,i+1] = get_estimates(samples.rho[:,i])
        end
        estimates[:,end-1] = get_estimates(samples.tau2)
        var_labs_tex = vcat(["Value"],[LaTeXString("\$\\rho_{$i}\$") for i in 1:nx],
            [LaTeXString("\$\\theta_{$j}\$") for j in 1:ntheta],
            [LaTeXString("\$\\tau\$"),LaTeXString("\$\\sigma\$")])
        var_labs = vcat(["Value"],["ρ $i" for i in 1:nx],
            ["θ $j" for j in 1:ntheta],
            ["τ","σ"])
    else
        estimates = Array{Float64}(undef,4,ntheta+3)
        var_labs_tex = vcat(["Value"],[LaTeXString("\$\\theta_{$j}\$") for j in 1:ntheta],
            [LaTeXString("\$\\sigma\$")])
        var_labs = vcat(["Value"],["θ $j" for j in 1:ntheta],
        ["σ"])
    end
    for i in 1:ntheta
        estimates[:,i+nx+1] = get_estimates(samples.theta[:,i])
    end
    estimates[:,end] = get_estimates(samples.sig2)
    param_labs = ["95% Lower Bound","Median","Mean","95% Upper Bound"]
    
    estimates = string.(round.(estimates,digits=5))
    estimates[:,1] = param_labs

    pretty_table(estimates;header=var_labs)

    open("$(mdl_apnd)_param_estimates.txt","w") do file
        pretty_table(file,estimates;header=var_labs)
    end
    open("$(mdl_apnd)_param_estimates_LaTeX.txt","w") do file
        pretty_table(file,estimates;header=var_labs_tex,backend=Val(:latex))
    end
end

"""
    make_estimate_table!(samples::GriddyPosteriors,nx::Int,ntheta::Int,mdl_apnd::String)
Function to make the table of estimates from the posterior samples.

---
Positional arguments
* `samples::GriddyPosteriors` struct containing the posterior samples from the griddy Gibbs sampler.
* `nx::Int` number of x variables.
* `ntheta::Int` number of theta variables.
* `mdl_apnd::String` String to append to the front of the saved plot's file name.
"""
function make_estimate_table(samples::GriddyPosteriors,nx::Int,ntheta::Int,mdl_apnd::String)
    if nx > 0
        estimates = Array{Float64}(undef,4,ntheta+4)
        estimates[:,2] = get_estimates(samples.rho)
        estimates[:,end-1] = get_estimates(samples.phi)
        var_labs_tex = vcat(["Value"],[LaTeXString("\$\\rho\$")],
            [LaTeXString("\$\\theta_{$j}\$") for j in 1:ntheta],
            [LaTeXString("\$\\tau\$"),LaTeXString("\$\\sigma\$")])
        var_labs = vcat(["Value"],["ρ"],
            ["θ $j" for j in 1:ntheta],
            ["τ","σ"])
    else
        estimates = Array{Float64}(undef,4,ntheta+3)
        var_labs_tex = vcat(["Value"],[LaTeXString("\$\\theta_{$i}\$") for j in 1:ntheta],
            [LaTeXString("\$\\sigma\$")])
        var_labs = vcat(["Value"],["θ $j" for j in 1:ntheta],
            ["σ"])
    end
    for i in 1:ntheta
        estimates[:,i+2] = get_estimates(samples.theta[:,i])
    end
    estimates[:,end] = get_estimates(samples.sig2)
    param_labs = ["95% Lower Bound","Median","Mean","95% Upper Bound"]
    
    estimates = string.(round.(estimates,digits=5))
    estimates[:,1] = param_labs

    pretty_table(estimates;header=var_labs)

    open("$(mdl_apnd)_param_estimates.txt","w") do file
        pretty_table(file,estimates;header=var_labs)
    end
    open("$(mdl_apnd)_param_estimates_LaTeX.txt","w") do file
        pretty_table(file,estimates;header=var_labs_tex,backend=Val(:latex))
    end
end

"""
    post_process(samples::BulkVarsStruct,data::DataStr,nx::Int,ntheta::Int,nburn::Int,
    scales::Scaling)
    post_process(samples::BulkVarsStruct,data::DataStr,nx::Int,ntheta::Int,nburn::Int,
    scales::Scaling;make_plots::Bool,save_plots::Bool,show_plots::Bool,
    nbins::Int,nthin::Int,mdl_apnd::String)
Wrapper function to post-process the posterior samples.

---
Positional arguments
* `samples::BulkVarsStruct` Struct containing the posterior samples.
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `nx::Int` Number of x dimensions.
* `ntheta::Int` number of θ dimensions.
* `nburn::Int` The number of samples to burn.
* `scales::Scalint` Struct containing the scaling information for the data.

Keyword arguments
* `make_plots::Bool` Indicator of whether to make plots.
  * default value of true
* `save_plots::Bool` Indicator of whether to save the plots.
  * default value of true
* `show_plots::Bool` Indicator of whether to show the plots.
  * default value of true
* `nthin::Int` The number of samples to skip when thinning.
  * default value of 5
* `nbins::Int` The number of bins to use for histograms.
  * default value of 30
* `mdl_apnd::String` String to append to the front of the generated plots' file names.
  * default value of ""
"""
function post_process(samples::BulkVarsStruct,data::DataStr,nx::Int,ntheta::Int,
    nburn::Int,scales::Scaling;make_plots::Bool=true,save_plots::Bool=true,
    show_plots::Bool=true,nthin::Int=5,nbins::Int=30,mdl_apnd::String="")
    
    samples = remove_burn(samples,nburn)
    samples = thin_samples(samples,nthin)
    scaled_samples = normalize_samples(samples,scales)
    sqrt_variance!(scaled_samples)

    if make_plots
        posterior_hist!(scaled_samples.sig2,nbins,"sigma",0,save_plots,show_plots,mdl_apnd)
        trace_plot!(scaled_samples.sig2,"sigma",0,save_plots,show_plots,mdl_apnd)
        posterior_hist!(scaled_samples.tau2,nbins,"tau",0,save_plots,show_plots,mdl_apnd)
        trace_plot!(scaled_samples.tau2,"tau",0,save_plots,show_plots,mdl_apnd)
        #plot_correlation(samples.theta,"theta",save_plots,show_plots,mdl_apnd)
        for theta in 1:ntheta
            posterior_hist!(scaled_samples.theta[:,theta],nbins,"theta",
                theta,save_plots,show_plots,mdl_apnd)
            trace_plot!(scaled_samples.theta[:,theta],"theta",
                theta,save_plots,show_plots,mdl_apnd)
        end
        if nx > 0
            #plot_correlation(samples.rho,"rho",save_plots,show_plots,mdl_apnd)
            for rho in 1:nx
                posterior_hist!(scaled_samples.rho[:,rho],nbins,"rho",rho,
                    save_plots,show_plots,mdl_apnd)
                trace_plot!(scaled_samples.rho[:,rho],"rho",rho,
                    save_plots,show_plots,mdl_apnd)
            end
            plot_disc!(scaled_samples.delta,data.exp.x,scales,nbins,save_plots,show_plots,mdl_apnd)
            plot_prediction!(scaled_samples,data,scales,save_plots,show_plots,mdl_apnd)
        end
    end
    make_estimate_table(scaled_samples,nx,ntheta,mdl_apnd)
end

"""
    post_process(samples::GriddyVarsStruct,data::DataStr,nx::Int,ntheta::Int,nburn::Int,
    scales::Scaling,theta_grid::Array{Float64,2},sig_grid::Union{Nothing,Array{Float64,2}},
    grid_resp::Array{Float64,2})
    post_process(samples::GriddyVarsStruct,data::DataStr,nx::Int,ntheta::Int,nburn::Int,
    scales::Scaling,theta_grid::Array{Float64,2},sig_grid::Union{Nothing,Array{Float64,2}},
    grid_resp::Array{Float64,2};make_plots::Bool,save_plots::Bool,show_plots::Bool,
    nbins::Int,nthin::Int,mdl_apnd::String)
Wrapper function to post-process the posterior samples.

---
Positional arguments
* `samples::GriddyVarsStruct` Struct containing the posterior samples.
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `nx::Int` Number of x dimensions.
* `ntheta::Int` number of θ dimensions.
* `nburn::Int` The number of samples to burn.
* `scales::Scalint` Struct containing the scaling information for the data.
* `theta_grid::Array{Float64,2}` Matrix of theta grid values.
* `sig_grid::Union{Nothing,Array{Float64,2}}` Matrix of integrated discrepancy covaraince grid values.
* `grid_resp::Array{Float64,2}` Responses of the surrogate model corresponding `theta_grid`.

Keyword arguments
* `make_plots::Bool` Indicator of whether to make plots.
  * default value of true
* `save_plots::Bool` Indicator of whether to save the plots.
  * default value of true
* `show_plots::Bool` Indicator of whether to show the plots.
  * default value of true
* `nthin::Int` The number of samples to skip when thinning.
  * default value of 5
* `nbins::Int` The number of bins to use for histograms.
  * default value of 30
* `mdl_apnd::String` String to append to the front of the generated plots' file names.
  * default value of ""
"""
function post_process(samples::GriddyVarsStruct,data::DataStr,nx::Int,ntheta::Int,
            nburn::Int,scales::Scaling,theta_grid::Array{Float64},sig_grid::Union{Nothing,Array{Float64}},
            grid_resp::Array{Float64};make_plots::Bool=true,save_plots::Bool=true,
            show_plots::Bool=true,nthin::Int=20,nbins::Int=30,mdl_apnd::String="")

    scaled_samples = normalize_samples(samples,scales,theta_grid,
        sig_grid,true)
    scaled_samples = remove_burn(scaled_samples,nburn)
    scaled_samples = thin_samples(scaled_samples,nthin)
    samples = remove_burn(samples,nburn)
    samples = thin_samples(samples,nthin)
    sqrt_variance!(scaled_samples)

    if make_plots
        posterior_hist!(scaled_samples.sig2,nbins,"sigma",0,save_plots,show_plots,mdl_apnd)
        trace_plot!(scaled_samples.sig2,"sigma",0,save_plots,show_plots,mdl_apnd)
        posterior_hist!(scaled_samples.sig_star2,nbins,"tau",0,save_plots,show_plots,mdl_apnd)
        trace_plot!(scaled_samples.sig_star2,"tau",0,save_plots,show_plots,mdl_apnd)
        #plot_correlation(samples.theta,"theta",save_plots,show_plots,mdl_apnd)
        for theta in 1:ntheta
            posterior_hist!(scaled_samples.theta[:,theta],nbins,"theta",
                theta,save_plots,show_plots,mdl_apnd)
            trace_plot!(scaled_samples.theta[:,theta],"theta",
                theta,save_plots,show_plots,mdl_apnd)
        end
        if nx > 0
            #plot_correlation(samples.rho,"rho",save_plots,show_plots,mdl_apnd)
            posterior_hist!(scaled_samples.rho,nbins,"rho",0,
                save_plots,show_plots,mdl_apnd)
            trace_plot!(scaled_samples.rho,"rho",0,
                save_plots,show_plots,mdl_apnd)

            plot_prediction!(samples.theta,scaled_samples,grid_resp,
                data,scales,save_plots,show_plots,mdl_apnd)
        end
    end
    make_estimate_table(scaled_samples,nx,ntheta,mdl_apnd)
end
