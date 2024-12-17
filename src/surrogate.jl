#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########################## Define Surrogate Functions #############################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

"""
    surrogate_model(x::Array{Float64},theta::Array{Float64},y::Array{Float64},nx::Int,ntheta::Int)
Function to generate the default surrogate model for calibration (a vector of Gaussian Processes, using the `GaussianProcesses.jl` package).

---
Keyword arguments
* `x::Array{Float64}` Array of control independent variables in the computer model data set, dimensions q and p.
* `theta::Array{Float64}` Array of unknown independent variables in the computer model data set, dimensions n and m.
* `y::Array{Float64}` Array of responses from the computer model data set, dimensions n and q.
* `nx::Int` Integer indicating the number of independent control variables, p.
* `ntheta::Int` Integer indicating the number of unknown independent variables, m.

---
Returns
* `model::Vector{GPE}` A Vector of length q whose elements are `GPE` models from the `GaussianProcesses.jl` package.

---
Details
For a computer model data set where evaluated of n settings of theta, and at each setting of theta,
is evaluated at q settings of x (producing an n by q Array of responses), this function creates a vector of q Gaussian process models.
Each Gaussian process model is trained on the ith response at the ith setting of x, as a function of theta.
Each model is trained on data scaled to the [0,1] interval and uses the following initial values for hyperparameters.
* zero-mean function.
* squared exponential covariance kernel with initial hyperparameters of 0.0 for all log length scales and log standard deviation.
* Prior distributions of N(0,1) for the log length scale and log standard deviation hyperparameters.
* Prior distribution of N(-2,1) for the log noise of the model (observational error).
"""
function surrogate_model(x::Array{Float64},theta::Array{Float64},
    y::Array{Float64},nx::Int,ntheta::Int)

    num_design_settings = size(theta)[1]
    num_response_locations = size(x)[1]
    if num_response_locations != size(y)[1]
    error("The number of x settings does not match the number of response settings")
    end
    
    emulator_mean = [MeanZero() for i in 1:num_response_locations]
    emulator_kern = [SE(repeat([0.0],ntheta),0.0) for i in 1:num_response_locations]
    model = [GP(theta',y[i,:],emulator_mean[i],emulator_kern[i]) for i in 1:num_response_locations]
    
    for i in eachindex(model)
    set_priors!(emulator_kern[i],repeat([Normal()],ntheta+1))
    set_priors!(model[i].logNoise,[Normal(-2.0,0.2)])
    end
    return model
end

"""
    train_model!(model)
    train_model!(model;epochs::Int,make_plots::Bool,save_plots::Bool,show_plots::Bool,
    mdl_apnd::String)
Function to train the surrogate model with Bayesian inference via elliptical slice sampling.

---
Positional arguments
* `model` Surrogate model.

Keyword arguments
* `epochs::Int` The number of MCMC samples for training the model.
  * default value of 2500
* `make_plots::Bool` Indicator of whether to make plots.
  * default value of true
* `save_plots::Bool` Indicator of whether to save the plots.
  * default value of true
* `show_plots::Bool` Indicator of whether to show the plots.
  * default value of true
* `mdl_apnd::String` String to append to the front of the generated plots' file names.
  * default value of ""
"""
function train_model!(model;epochs::Int=2500,make_plots::Bool=true,
            save_plots::Bool=true,show_plots::Bool=true,mdl_apnd::String="")

    model_chains = [ess(model[i],nIter=epochs) for i in eachindex(model)]

    if make_plots
        for i in eachindex(model)
            p = Plots.plot(model_chains[i]',label=false)
            title!("Trace Plot for GPM $i Training")
            xlabel!("Iteration")
            ylabel!("Draw Value")

            save_plots ? Plots.savefig(p,"$(mdl_apnd)_surrogate_model_$i-trace_plot.png") :
                nothing
            show_plots ? Plots.display(p) : nothing
        end
    end
end

"""
    predict_y_all(theta_settings::Vector{Float64},model)

Function to get the default surrogate model output for a specified input setting.

---
Keyword arguments
* `theta_settings::Vector{Float64}` Vector of θ input settings for prediction.
* `model` Surrogate model.

---
Returns
* `responses:Vector{Float64}` Vector of p responses of the surrogate model, for the p control variable settings on which it is trained.
"""
function predict_y_all(theta_settings::Vector{Float64},model)
    responses = Vector{Float64}(undef,length(model))
    for i in eachindex(model)
        result = GaussianProcesses.predict_y(model[i],permutedims(theta_settings'))[1][1]
        responses[i] = result
    end
    return responses
end