#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
############################# Define Misc Functions ###############################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
"""
    setup(design::Array{Float64},simobs::Vector{Float64},expobs::Union{Array{Float64},Float64},nx::Int,ntheta::Int,alpha_tau2::Float64,beta_tau2::Float64,alpha_sig2::Float64,beta_sig2::Float64,a_rho::Float64,b_rho::Float64)
Function to perform necessary calculation prior to calibration.

---
Keyword arguments
* `design::Array{Float64}` Matrix of input values for the computer simulator, of dimensions n and p+m. The first p columns correspond to control variables (x) and the last m correspond to unknown variables (θ).
* `simobs::Vector{Float64}` Vector of response values from the computer simulator, of length n. The rows must correspond to their appropriate input settings in `design`.
* `expobs::Union{Array{Float64},Float64}` Matrix of experimental data, of dimensions v and p+1. The first p columns correspond the control independent variables (x) and the last column corresponds to the observed response.
* `nx::Int` The number of independent control variables (x) in the experimental and computer simulator data.
* `ntheta::Int` The number of independent unknown variables (θ) in the computer simulator data.
* `alpha_tau2::Float64` Inverse Gamma shape parameter for the data error variance prior (τ^2).
* `beta_tau2::Float64` Inverse Gamma scale parameter for the data error variance prior (τ^2).
* `alpha_sig2::Float64` Inverse Gamma shape parameter for the discrepancy vairiance prior (σ^2).
* `beta_sig2::Float64` Inverse Gamma scale parameter for the discrepancy variance prior (σ^2).
* `a_rho::Float64` Beta shape parameter 1 for the discrepancy correlation prior (ρ).
* `b_rho::Float64` Beta shape parameter 2 for the discrepancy correlation prior (ρ).

---
Returns
* `nobs_tot::Int` An integer decsribing the total number of data points in the experimental data.
* `nrep::Int` An integer describing the number of independent repeated observations of the same multivariate normal dsitribution data.
* `nloc::Int` An integer describing the number of unique points in the multivariate normal distribution data.
* `scales::Scaling` A data structure describing the minimum and maximum values for the x, y, and θ variables.
* `data::DataStr` A data structure containing the normalized computer simulator and experimental data.
* `priors::PriorData` A data structure containing the prior distribution hyperparameters.
"""
function setup(design_raw::Array{Float64},simobs_raw::Vector{Float64},
    expobs_raw::Union{Array{Float64},Float64},nx::Int,ntheta::Int,
    alpha_tau2::Float64,beta_tau2::Float64,alpha_sig2::Float64,
    beta_sig2::Float64,a_rho::Float64,b_rho::Float64)
    if length(size(expobs_raw)) == 0                                   # ensure data is formatted as an array (in the case of univarite normal distribution data)
        expobs_raw = [expobs_raw]
    end

    nobs_tot = size(expobs_raw)[1]                                     # calculate total number of experimental data points
    #nsim = length(unique(design_raw[:,end]))                           # calculate total number of computer simulator data points
    nsim = length(simobs_raw)

    if nx > 0                                                          # calculate number of unique observation settings in data (checks for univariate and multivariate data)
        nloc = size(unique(design_raw[:,1:nx],dims=1))[1]       
    else
        nloc = 1
    end
    
    nrep = Int(nobs_tot/nloc)                                          # calculate number of repeated independent experimental observations
    
    # calcualte the minimum and maximum values for theta in the simulations
    theta_min = [minimum(design_raw[:,i]) for i in (nx+1):nx+ntheta]
    theta_max = [maximum(design_raw[:,i]) for i in (nx+1):nx+ntheta]
    
    # check if experimental responses fall within the range of the computer simulator responses
    if (minimum(expobs_raw[:,end]) < minimum(simobs_raw))
        @warn "Experimental responses are lesser than simulation responses. Consider adjusting maximum and minimum settings for simulator input."
    elseif (maximum(expobs_raw[:,end]) > maximum(simobs_raw))
        @warn "Experimental responses are greater than simulation responses. Consider adjusting maximum and minimum settings for simulator input."
    end
    
    # calculate the minimum and maximum response values from simulation and experiments
    y_min = minimum(vcat(vec(simobs_raw),expobs_raw[:,end]))
    y_max = maximum(vcat(vec(simobs_raw),expobs_raw[:,end]))
    
    println(y_min)
    println(y_max)
    # check if experimental control settings match with computer simulator control settings
    if expobs_raw[1:nloc,1:nx] != design_raw[1:nloc,1:nx]
        @warn "Computer simulator independent control variable (x) settings do not match experimental values. This may result in errors during calibration."
    end
    
    if nx > 0
        # calculate the minimum and maximum values for x from simulation and experiments
        x_min = [minimum(vcat(design_raw[:,i],expobs_raw[:,i])) for i in 1:nx]
        x_max = [maximum(vcat(design_raw[:,i],expobs_raw[:,i])) for i in 1:nx]
    else
        x_min = 0.0
        x_max = 1.0
    end
    
    scales = Scaling(
        theta=ScalePar(min=theta_min,max=theta_max),
        y=ScalePar(min=y_min,max=y_max),
        x=ScalePar(min=x_min,max=x_max)
    )

    # normalize data
    design_norm,simobs_norm,expobs_norm = normalize_data(design_raw,simobs_raw,expobs_raw,nx,nsim,scales)

    # format Data
    data = format_data(design_norm,simobs_norm,expobs_norm,nx)

    priors = format_prior_hyperparameters(data,alpha_tau2,
    beta_tau2,alpha_sig2,beta_sig2,
    a_rho,b_rho,nx,ntheta)

    if data.exp.x != data.sim.x
        println("Number of experimental observations: ",size(data.exp.x)[1])
        println("Number of control settings in computer simulator: ", size(data.sim.x)[1])
        for i in axes(data.exp.x)[1]
            println("Experiment index $i ∈ simulator?: ",(data.exp.x[i,:] in data.sim.x))
        end
        error("The control settings do not match between the experimental and computer simulator data.
        Please ensure that the computer simulator and experimental data control settings match.")
    end
    return nobs_tot,nrep,nloc,scales,data,priors
end

"""
    unnormalize_var(var::Array{Float64},scales::ScalePar)
Function to un-normalize a normalized variable's Array data. This function takes data ∈ [0,1] and scales it based on the `scales`.

---
Keyword arguments
* `var::Array{Float64}` The array of field values to be scaled, with dimensions n and p.
* `scales::ScalePar` The data structure containing the minimum and maximum values for scaling. The minimum and maximum values specified in this structure must be Vectors of length p.
---
∀ x ∈ `var`, y = (x-`scales.min`)/(`scales.max`-`scales.min`)
---
Returns
* `output::Array{Float64}` An array containing the original field values, `var`, scaled to show the relative distance between the minimum and maximum values specified by `scales`.

"""
function unnormalize_var(var::Array{Float64},scales::ScalePar)
    output = similar(var)
    num_dims = length(size(var))

    if num_dims == 1
        output .= var .* (scales.max - scales.min) .+ scales.min
    elseif num_dims == 2
        if length(scales.min) == 1
            for i in 1:size(var)[2]
                output[:,i] .= var[:,i] .* (scales.max - scales.min) .+ scales.min
            end
        else
            for i in 1:size(var)[2]
                output[:,i] .= var[:,i] .* (scales.max[i] - scales.min[i]) .+ scales.min[i]
            end
        end
    else
        error("The Array passed to be normalized has too many dimensions and I don't know how to handle it.
        Please ensure that the `size()` of this variable returns a Tuple of length 2 or less.")
    end
    return output
end

"""
    normalize_var(var::Array{Float64},scales::ScalePar)
Function to scale a single variable's Array data to the relative distance between the minimum and maximum values specified by the `scales` structure.
The function returns an Array, `output`, of equal size to `var` with its data scaled based on the minimum and maximum value described in `values`.

---
Keyword arguments
* `var::Array{Float64}` The array of field values to be scaled, with dimensions n and p.
* `scales::ScalePar` The data structure containing the minimum and maximum values for scaling. The minimum and maximum values specified in this structure must be Vectors of length p.
---
∀ x ∈ `var`, y = (x-`scales.min`)/(`scales.max`-`scales.min`)
---
Returns
* `output::Array{Float64}` An array containing the original field values, `var`, scaled to show the relative distance between the minimum and maximum values specified by `scales`.
"""
function normalize_var(var::Array{Float64},scales::ScalePar)
    output = similar(var)
    num_dims = length(size(var))

    if num_dims == 1
        output .= (var .- scales.min) ./ (scales.max - scales.min)
    elseif num_dims == 2
        if length(scales.min) == 1
            for i in size(var)[2]
                output[:,i] .= (var[:,i] .- scales.min) ./ (scales.max - scales.min)
            end
        else
            for i in 1:size(var)[2]
                output[:,i] .= (var[:,i] .- scales.min[i]) ./ (scales.max[i] - scales.min[i])
            end
        end
    else
        error("The Array passed to be normalized has too many dimensions and I don't know how to handle it.
        Please ensure that the `size()` of this variable returns a Tuple of length 2 or less.")
    end
    return output
end

"""
    normalize_data(design_in::Array{Float64},simobs_in::Vector{Float64},expobs_in::Array{Float64},nx::Int64,nsim::Int64,scales::scaling,rev::Bool=false)
Function to normalize data to the [0,1] interval or reverse the normalization back to the original data.

---
Keyword arguments
* `design_in::Array{Float64}` Matrix of input values for the computer simulator, of dimensions n and p+m. The first p columns correspond to control variables (x) and the last m correspond to unknown variables (θ).
* `simobs_in::Vector{Float64}` Vector of response values from the computer simulator, of length n. The rows must correspond to their appropriate input settings in `design`.
* `expobs_in::Union{Array{Float64},Float64}` Matrix of experimental data, of dimensions v and p+1. The first p columns correspond the control independent variables (x) and the last column corresponds to the observed response.
* `nx::Int64` The number of independent control variables (x) in the experimental and computer simulator data.
* `nsim::Int64` The number of computer simulator data points.
* `scales::Scaling` Data structure containing minimum and maximum values for the x, y, and θ variables for scaling.
* `rev::Bool` An indicator of whether to normalize the data or reverse the normalization. Default value of false.

---
Returns
* `design_out::Array{Float64}` The `design_raw` Matrix normalized to the relative Euclidean distance between the values specified in `scales`.
* `simobs_out::Vector{Float64}` The `simobs_raw` Vector normalized to the relative Euclidean distance between the values specified in `scales`.
* `expobs_out::Union{Array{Float64},Float64}` The `expobs_raw` Matrix normalized to the relative Euclidean distance between the values specified in `scales`.

---
Details
Concatenates all values for y from the computer simulator and experimental data.
Concatenates all values for each dimension of x in the computer simulator and experimental data.
Calls `normalize_var` or `unnormalize_var` on the concatenated y data, the concatenated x data in each dimension of x, and the computer simulator θ values for each dimension of θ.
"""
function normalize_data(design_in::Array{Float64},simobs_in::Vector{Float64},
    expobs_in::Union{Array{Float64},Float64},nx::Int64,
    nsim::Int64,scales::Scaling,rev::Bool=false)
    # if there are x variables, concatenate them of across computer simulator and experimental data and then normalize them
    if nx > 0
        x_in = vcat(design_in[:,1:nx],expobs_in[:,1:nx])
        x_out = rev ? unnormalize_var(x_in,scales.x) : normalize_var(x_in,scales.x)
    end

    # extract theta values from the design matrix and normalize them
    theta_in = design_in[:,(nx+1):end]
    theta_out = rev ? unnormalize_var(theta_in,scales.theta) : normalize_var(theta_in,scales.theta)

    # concatenate response variable from experimental and computer simulator data and normalize it
    y_in = vcat(simobs_in,expobs_in[:,end])
    y_out = rev ? unnormalize_var(y_in,scales.y) : normalize_var(y_in,scales.y)

    # rearrange normalized data back into original format, checking against the number of x variables
    if nx > 0
        design_out = hcat(x_out[1:nsim,:],theta_out)
        simobs_out = y_out[1:nsim]
        expobs_out = hcat(x_out[(nsim+1):end,:],y_out[(nsim+1):end])
    else
        design_out = theta_out
        simobs_out = y_out[1:nsim]
        expobs_out = y_out[(nsim+1):end]
    end

    return design_out,simobs_out,expobs_out
end

"""
    pivot_data_wide(expobs::Array{Float64})
Function to pivot multiple observations of the same n-length multivariate normal distribution into a wider array.
Given there are m repeated observations of the same n-length multivariate normal distribution, with each repeated observation having the same independent variable settings,
this function will reorganize the m*n length Array into a wider pivoted observation data Array, which is preferrable for likelihood calculations. 

---
Keyword arguments
* `x::Array{Float64}` An Array of dimensions n*m and p, containing the settings of the p independent variables at each observational data point
* `y::Vector{Float64}` A Vector of length n*m, containing the experimental observations aligning to the settings specified in `x`

---
Returns
* `x_unique::Array{Float64}` An array of the unique settings of the independent control variables, dimensions n and p
* `y_wide::Array{Float64}` An array of the observations organized by repetition, dimensions n and m
"""
function pivot_data_wide(x::Array{Float64},y::Vector{Float64})
    # get unique x locations and data
    x_unique = unique(x,dims=1)
    num_loc = size(x_unique)[1]
    if size(x)[1] != length(y)
        error("The number of input settings does not match the number of responses!
        Number of input settings: ",size(x)[1],"
        Number of response values: ",length(y))
    end

    if size(x)[1]/num_loc - floor(size(x)[1]/num_loc) != 0
        error("There was an error pivoting the data to a wider matrix")
    else
        num_rep = round(Int,(size(x)[1]/num_loc))
    end


    if Int(num_rep)*Int(num_loc) != size(x)[1]
        error("The number of repeated observations and length of observed MVN data points per observation do not align with the total number of experimental data points
        Number of repetitions calculated = ",Int(num_rep),
        "Length of MVN observational data per repetition = ",Int(num_loc),
        "Total experimental data length = ", size(x)[1])
    end

    y_wide = reshape(y,num_loc,num_rep)

    ita = 0
    for j in 1:num_rep, i in 1:num_loc
        ita = (j-1)*num_loc + i

        wide_response = y_wide[i,j]
        orig_response = y[ita]
        dict_setting = x_unique[i,:]
        orig_setting = x[ita,:]

        if wide_response != orig_response
            error("Response variable mapping does not match the value at its original index!
            Original response value: $orig_response
            Mapped response setting: $wide_response
            Matrix indices in evaluation: i = $i, j = $j -> Vector index = $ita")

        elseif dict_setting != orig_setting
            error("Independent variable mapping does not match the setting at its original index!
            Mapping of independent variable for this index: $dict_setting
            Independent variable in original Array: $orig_setting
            Matrix indices in evaluation: i = $i, j = $j -> Vector index = $ita")
        end
        
    end
    return x_unique,y_wide
end

"""
    format_data(design::Array{Float64},simobs::Vector{Float64},expobs::Array{Float64},nx::Int64)
Function to format raw data arrays supplied by `design`, `simobs`, and `expobs` into the requisatory structures for model calibration

---
Keyword arguments
* `design::Array{Float64}` An Array containing the input settings for both the control and unknown independent variables in the computer simulator. Dimenions n and p+m. The first p columns contain the control variables and the last m columns contain the unknown variables.
* `simobs::Vector{Float64}` A Vector containing the response of the computer simulator, corresponding to the inputs specified in `design`. Length n.
* `expobs::Array{Float64}` An Array containing the experimental data to which the computer model is calibrated. Dimensions n and p+1. The first p columns contain the control variables and the last column contains the observations.
* `nx::Int` An integer specifying the number of dimensions of the independent control variables

---
Returns
* `data::data_str` The data structure containing the `sim_str` and `exp_str` data structures housing the computer simulator and experimental data, respectively.
"""
function format_data(design::Array{Float64},simobs::Vector{Float64},
    expobs::Array{Float64},nx::Int64)
    sim_x = design[:,1:nx]     #pull x vars from sim data
    sim_theta = design[:,(nx+1):end] # pull theta vars from sim data
    sim_theta_unique = unique(sim_theta,dims=1)
    sim_y = simobs              # pull response var from sim data
    exp_x = expobs[:,1:nx]      # pull x vars from exp data
    exp_y = expobs[:,end] # pull response var from exp data

    x_unique_exp,y_wide_exp = pivot_data_wide(exp_x,exp_y)
    x_unique_sim,y_wide_sim = pivot_data_wide(sim_x,sim_y)
    
    sim = SimStr(x_unique_sim,y_wide_sim,sim_theta_unique,
    sim_x,sim_y,sim_theta) # store in data structs
    exp = ExpStr(x_unique_exp,y_wide_exp,exp_x,exp_y)
    data = DataStr(sim,exp)
    return data
end

"""
    format_prior_hyperparameters(data::DataStr,alpha_tau2::Float64,beta_tau2::Float64,alpha_sig2::Float64,beta_sig2::Float64,a_rho::Float64,b_rho::Float64,nx::Int64,ntheta::Int64)
Funtion to initialize the specified prior distribution hyperparameters into the requisatory struct.

---
Keyword arguments
* `data::DataStr` Struct containing the computer simulator and experimental data.
* `alpha_tau2::Float64` Inverse Gamma shape parameter for the data error variance prior (τ^2).
* `beta_tau2::Float64` Inverse Gamma scale parameter for the data error variance prior (τ^2).
* `alpha_sig2::Float64` Inverse Gamma shape parameter for the discrepancy vairiance prior (σ^2).
* `beta_sig2::Float64` Inverse Gamma scale parameter for the discrepancy variance prior (σ^2).
* `a_rho::Float64` Beta shape parameter 1 for the discrepancy correlation prior (ρ).
* `b_rho::Float64` Beta shape parameter 2 for the discrepancy correlation prior (ρ).
* `nx::Int` Integer specifying the number of independent control variables (x).
* `ntheta::Int` Integer specifying the number of unknown computer model parameters (θ).

---
Returns
* `priors::PriorData` Struct containing the prior distribution hyperparameters for the variables in the model.

---
Details
This function is agnostic to what prior distributions are used for the model but other functions may require the following prior distributions.
* θ ∼ U(A,B)
* τ^2 ∼ IG(α,β)
* σ^2 ∼ IG(α,β)
* ρ ∼ Beta(a,b)
"""
function format_prior_hyperparameters(data::DataStr,alpha_tau2::Float64,
    beta_tau2::Float64,alpha_sig2::Float64,beta_sig2::Float64,a_rho::Float64,
    b_rho::Float64,nx::Int,ntheta::Int)
    theta = data.sim.theta                    #pull theta
    
    theta_lower = vec(minimum(theta,dims=1))
    theta_upper = vec(maximum(theta,dims=1))
    
    a_rho = repeat([a_rho],nx)
    b_rho = repeat([b_rho],nx)
    
    theta = PriorVar(theta_lower,theta_upper) #store in structs
    tau2 = PriorVar(alpha_tau2,beta_tau2)
    sig2 = PriorVar(alpha_sig2,beta_sig2)
    rho = PriorVar(a_rho,b_rho)
    priors = PriorData(theta=theta,tau2=tau2,sig2=sig2,rho=rho)
    return priors
end