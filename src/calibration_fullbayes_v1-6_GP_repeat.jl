###################################################################################
#                                                                                 #
#                   888 d8b 888                      888    d8b                   # 
#                   888 Y8P 888                      888    Y8P                   # 
#                   888     888                      888                          #
#  .d8888b  8888b.  888 888 88888b.  888d888 8888b.  888888 888  .d88b.  88888b.  #
# d88P"        "88b 888 888 888 "88b 888P"      "88b 888    888 d88""88b 888 "88b #
# 888      .d888888 888 888 888  888 888    .d888888 888    888 888  888 888  888 #
# Y88b.    888  888 888 888 888 d88P 888    888  888 Y88b.  888 Y88..88P 888  888 #
#  "Y8888P "Y888888 888 888 88888P"  888    "Y888888  "Y888 888  "Y88P"  888  888 #
#                                                                                 #
###################################################################################                                                                               
###################################################################################
#                                                                                 #
#                                           888                                   #
#                                           888                                   #
#                                           888                                   #
#                      .d8888b .d88b.   .d88888  .d88b.                           #
#                     d88P"   d88""88b d88" 888 d8P  Y8b                          #
#                     888     888  888 888  888 88888888                          #
#                     Y88b.   Y88..88P Y88b 888 Y8b.                              #
#                      "Y8888P "Y88P"   "Y88888  "Y8888                           #
#                                                                                 #
###################################################################################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
###################################################################################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
################################ Calibration Code #################################
############################ Written by Stephen Wright ############################
## This code is intended to calibrate a computer simulation to experimental data.##
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
################################## Requirements ###################################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########      This code requires the following packages for Julia:        #########
########  -- LinearAlgebra                                                #########
########  -- Statistics                                                   #########
########  -- Distributions                                                #########
########  -- Plots                                                        #########
########  -- GaussianProcesses                                            #########
########  -- Optim                                                        #########
########  -- DelimitedFiles                                               #########
########  -- StatsPlots                                                   #########
########  -- PrettyTables                                                 #########
########  -- DataFrames                                                   #########
########  -- CSV                                                          #########
########  -- PlotlyJS                                                     #########
########  -- ProgressMeter                                                #########
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
####   This code requires 3 text files                                         ####
####   1) A design file that details the inputs for the simulator.             ####
####    Each column in this file should correspond to a different variable.    ####
####    Each row should correspond to a different setting of the simulator.    ####
####    Two types of variables can be given in the design file.                ####
####    a) Control variables (these are variables that can be controlled or    ####
####      measured during experimental tests).                                 ####
####    b) Calibration variables (these variables are unknown or not           ####
####      measureable during the experiment. These variables will be           ####
####      calibrated via Bayesian machine learning in this code.)              ####
####    All control variables must be specified in the first columns of the    ####
####    file before any calibration variables (unless there are no control     ####
####    variables in the model.                                                ####
####   2) A simulation results file that gives the results of the simulator.   ####
####    Each column corresponds to a different output variable.                ####
####    Each row corresponds to a different simulator setting.                 ####
####    A single variable type should be specified in this file.               ####
####    a) Response variables (specify the response of the simulator)          ####
####   3) An experiment file that describes the input and output variables     ####
####    of the experimental tests.                                             ####
####    Each column should correspond to a different variable.                 ####
####    Each row should correspond to a different experimental setup.          ####
####    Two types of variables can be specifies in this file.                  ####
####    a) Control variables                                                   ####
####    b) Response variables (specify the response of the experiment)         ####
####    If control variables are present in the model, they should be listed   ####
####    in the first columns, before the response variables.                   ####
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
################################## Methodology ####################################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
####   This code creates a statiscial model for a computer model and observed  ####
####   data. This model can be described as y(x) = f(x) + ϵ(x). Herein, y(x)   ####
####   are the experimental observations, f(x) is some unknown function that   ####
####   describes the physical processes, and ϵ(x) is observation error.        ####
####   f(x) is unknown but can be approximated by a computer simulation,       ####
####   g(x,θ). Because this simulation will probably not completely capture    ####
####   the physical phenomenon, a discrepancy term is added in, δ(x). The      ####
####   discrepancy function is represented by a Gaussian process. The updated  ####
####   form of the model is the following y(x) = g(x,θ) + δ(x) + ϵ(x). While   ####
####   the model could be used as is, the computer simulations in question are ####
####   typically computationally expensive and it is therefore useful to use   ####
####   a surrogate model to represent them in this model, η(x,θ). This code    ####
####   trains a Gaussian process surrogate model on the computer simulations   ####
####   and then performs Bayesian inference on the following parameters in     ####
####   the model:                                                              ####
####     -- θ, the unknown parameters in the computer simulations              ####
####     -- τ, the data model error (in standard deviation form)               ####
####     -- δ, the discrepancy function                                        ####
####     -- σ, the standard deviation of the discrepancy covariance            ####
####     -- ρ, the correlation parameter in the discrepancy covariance         ####
####   Markov chain Monte Carlo (MCMC) simulation is used to sample the        ####
####   posterior distributions of each of the variables explained above.       ####
####   Standard Gibbs sampling is used for the data model error, discrepancy   ####
####   function, and the discrepancy covariance standard deviation. A Random   ####
####   walk Metropolis-Hastings algorithm is used for the theta parameters     ####
####   and the correlation parameters. A stepsize tuning algorithm has been    ####
####   implemented to select an appropriate stepsize for the Metropoli-        ####
####   Hastings algorithm. The results of the Bayesian inference are then      ####
####   processed to visualize the information easily.                          ####
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#################################### Outputs ######################################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
#### The outputs for this code consist of processed results from the Bayesian  ####
#### Inference.                                                                ####
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
###################################################################################

#load packages
using LinearAlgebra
using Statistics
using Distributions
using Plots
using GaussianProcesses
using DelimitedFiles
using PrettyTables
using DataFrames
using CSV
using ProgressMeter
using StatsPlots
using BlackBoxOptim
using Parameters
using JLD2

include("structs.jl")
include("misc_functions.jl")
include("gaussian_process_kernel.jl")
include("initialization.jl")

#change directory to source file location
cd(dirname(@__FILE__))

init_time = time()
#include("calibration_init.jl")
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
############################### Do Not Edit Below #################################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

nx = 3
ntheta = 3

val_sample = 90 #sample rate for surrogate model validation in percent
val_error_threshold = 50 #error threshold for validation model, squared

#specify variables used for Metropolis algorithm
nmcmc = 50000 #number of samples in the MCMC algorithm after the stepsize 
#                 has been trained
nburn = 20000 #number of MCMC samples to discard when processing results
ntrialburn = 100 #number of MCMC samples to run at a time in the stepsize algorithm
ntrial = 500 #number of iterations of the stepsize algorithm

#input prior distribution hyperparameters
#data error
alpha_sig2 = 0.00000001
beta_sig2 = 0.00000001
#discrepancy variance
alpha_tau2 = 0.00000001
beta_tau2 = 0.00000001
#discrepancy correlation
a_rho = 1.0
b_rho = 1.0

nys = [6,54]
for ny in nys
pth_mdl = "/home/stephenw/Nextcloud/Documents/engr/PhD/calibration code/hannahs paper/redux/v2grid/surrogates/"
model = JLD2.load(pth_mdl*"ny$ny"*"_surrogate_model_independent.jld2")["model"]
#read in files for computer model and experimental data
#pth_grid = "/home/stephenw/Nextcloud/Documents/engr/PhD/calibration code/hannahs paper/redux/disc/simulated experimental data/"
pth_grid = "/run/media/stephenw/More\\x20Storage/palmetto_backup/expobs/"
pth_exp = "/home/stephenw/Nextcloud/Documents/engr/PhD/calibration code/hannahs paper/redux/nodisc/simulated experimental data/"
#pth_exp = "/run/media/stephenw/More\\x20Storage/palmetto_backup/expobs/"

mdl = "model_exphfl_soi-25_ny-$ny"*"_nsim-343_disc-40_"
#data in original scale
design_scale = readdlm(pth_grid*mdl*"design.txt",',') #read in original scale simulation 
#                                               design
#design_scale = unique(design_scale[:,4:end],dims=1)
simobs_scale = readdlm(pth_grid*mdl*"simobs.txt",',') #read in original scale simulation 
#                                               response
#simobs_scale = transpose(reshape(simobs_scale,15,700))
simobs_scale = simobs_scale[:,end]           #ensure this is a vector
#expobs_scale = readdlm(pth_exp*"model_exphfl_soi-25_ny-$ny"*"_nsim-1_disc-40_expobs.txt",',') #read in original scale experimental
#                                               data
expobs_scale = readdlm(pth_exp*"ny_$ny-expobs_nodisc.txt",'\t')
temp = readdlm(pth_exp*"model_exphfl_soi-25_ny-$ny"*"_nsim-1_disc-40_truesim.txt",',')
expobs_scale[:,end] .= temp[:,end]

nobs,nrep,nloc,scales,data,priors = setup(design_scale,simobs_scale,
expobs_scale,nx,ntheta,alpha_sig2,beta_sig2,alpha_tau2,beta_tau2,a_rho,b_rho)

mdl = "/home/stephenw/Nextcloud/Documents/engr/PhD/calibration code/griddy_gibbs/test_res/comparison/trad/data_type/truth/ny_$ny-"

#### surrogate model setup function
# inputs: data struct, number of x and theta dims
# returns: untrained surrogate model
function surrogate_model(x::Array{Float64},theta::Array{Float64},
    y::Vector{Float64},nx::Int64,ntheta::Int64)

    design_reshaped = unique(design[:,(nx+1):end],dims=1)
    num_loc = round(Int,size(design)[1]/size(design_reshaped)[1])
    loc = unique(design[:,1:nx],dims=1)
    num_design = size(design_reshaped)[1]
    simobs_reshaped = transpose(reshape(transpose(simobs),num_loc,num_design))
    global loc2idx = Dict([loc[i,:] => i for i in 1:num_loc])

    emulator_mean = [MeanZero() for i in 1:size(simobs_reshaped)[2]]
    emulator_kern = [SE(repeat([0.0],ntheta),0.0) for i in 1:size(simobs_reshaped)[2]]
    model = [GP(design_reshaped',simobs_reshaped[:,i],emulator_mean[i],emulator_kern[i]) for i in 1:size(simobs_reshaped)[2]]
    
    for i in 1:length(model)
        set_priors!(emulator_kern[i],repeat([Normal()],ntheta+1))
        set_priors!(model[i].logNoise,[Normal(-1.0,1.0)])
    end

    return model
    #=x_train = design_reshaped
    y_train = simobs_reshaped

    @rput x_train
    @rput y_train

    R"library(e1071)"

    R"models = list()"

    R"dat = data.frame(x_train)"
    for i in 1:num_loc
        reval("dat\$y$i = y_train[,$i]")
    end
    println(R"head(dat)")
    println("Training Surrogate Model")
    for i in 1:num_loc
        reval("models[[$i]] = svm(y$i ~ sqrt(X1)+sqrt(X2)+sqrt(X3),data=dat)")
    end
    =#
end

function surrogate_model(x::Array{Float64},theta::Array{Float64},y::Array{Float64},nx::Int,ntheta::Int)
    num_design_settings = size(theta)[1]
    num_response_locations = size(x)[1]
    if num_response_locations != size(y)[1]
    error("number of x settings does not match the number of response settings")
    end
    
    emulator_mean = [MeanZero() for i in 1:num_response_locations]
    emulator_kern = [SE(repeat([0.0],ntheta),0.0) for i in 1:num_response_locations]
    model = [GP(theta',y[i,:],emulator_mean[i],emulator_kern[i]) for i in 1:num_response_locations]
    
    for i in 1:length(model)
    set_priors!(emulator_kern[i],repeat([Normal()],ntheta+1))
    set_priors!(model[i].logNoise,[Normal(-2.0,0.2)])
    end
    return model
end


#generate surrogate model
#model = surrogate_model(data.sim.x,data.sim.theta,data.sim.y,nx,ntheta)

#println("Surrogate Model Training")

#train surrogate model
#model_chain = [ess(model[i],nIter=1500) for i in 1:length(model)]

function predict_y_all(theta_settings)
    #println(theta_settings)
    #test_data = test_data[:,(1+nx):end]
    responses = Vector{Float64}(undef,length(model))
    for i in 1:length(model)
    result = GaussianProcesses.predict_y(model[i],permutedims(theta_settings))[1][1]
    responses[i] = result
    end
    #println(responses)
    return responses
end

time_post_model = time()

#### likelihood function
# inputs: surrogate model, data, MCMC iter vals, MCMC iteration theta_step
# returns: likelihood value given thetas
function lik( data::DataStr,vars::UpdatedVars,
    thetas::Vector{Float64})

    #thetas_model = repeat(thetas',size(data.exp.x)[1]) #pull thetas
    eta = predict_y_all(thetas') #surogate modle est

    delta = vars.delta                                #pull disc fun
    mean = eta + delta                                #model est
    response = data.exp.y                             #repsonse data
    sz = size(eta)[1]

    sig2 = vars.sig2[1,1]                             #pull tau^2
    covariance = sig2*Matrix(1.0I,sz,sz)              #calculate covar matrix
    covariance = covariance + Matrix(sqrt(eps(Float64))I,sz,sz)
    covariance = 0.5*(covariance' + covariance)    #ensure symmetry for stability

    likelihood = prod(pdf(MvNormal(mean,sig2*Matrix(1.0I,sz,sz)),response)) #likelihood

    return likelihood
end

#### special case of likelihood for case where there are no x variables
function lik_nox( data::DataStr,vars::UpdatedVars,
    thetas::Vector{Float64},sum_y_2::Float64,sum_y_1::Float64,num_obs::Int)
    
    thetas_model = thetas'
    eta = predict_y_all(thetas')

    mean = eta
    sig2 = vars.sig2

    response = data.exp.y

    likelihood = prod(pdf.(Normal(mean,sqrt(sig2)),response))

    return likelihood
end

#### theta prior pdf, typ. obsolete
# inputsL prior data, MCMC iter vals
# returns: pdf val for theta prior given theta
function prior_theta(prior_data::PriorData,vars::UpdatedVars)
    A = prior_data.theta.par1 #min val
    B = prior_data.theta.par2 #max val
    pdf_val = 1/(B-A)         #pdf val
    return pdf_val
end
#### discrepancy prior pdf function
#inputs: MCMC iter vals, data, rho vals, number of x and obs
# reutrns pdf val of delta
function prior_delta(vars::UpdatedVars,data::DataStr,rho::Vector{Float64},
    nx::Int64,nobs::Int64)
    
    delta = vars.delta        #pull delta
    tau2 = vars.tau2[1]       #pull tau2
    rho = rho                 #pull rho
    x = data.exp.x            #pull x

    response = delta          
    mean = repeat([0],length(delta))  #mean of delta prior

    covar = tau2*correlation_construct(rho,x,nx,nobs) #calc covar matrix
    pdf_val = pdf(MvNormal(mean,covar),response)      #calculate pdf val
    return pdf_val
end
#### rho prior pdf function
# inputs: rhos, prior data, MCMC iteration vals
# returns: #pdf val of rho
function prior_rho(rho::Float64,prior_data::PriorData,k::Integer)
    a = prior_data.rho.par1[k] #pull prior hyperparams
    b = prior_data.rho.par2[k]
#    rho = vars.rho         #pull rho
    pdf_val = pdf(Beta(a,b),rho)   #calc pdf val
    return pdf_val
end
#### g function for probability transformation
# inputs: scale and minimum value for variable, transformed probability value
# returns: probabilty of gamma in original scale
function g(c1::Float64,c2::Float64,gamma::Float64)
    return (c1*(exp.(gamma))/(1+exp.(gamma)-c2)) #probabiltity transformation
end
#### g inverse function for probability transformaiton
# inputs: scale and minimu, value for variable, probabiltiy value of variable
# returns: probability in gamma space
function g_inv(c1::Float64,c2::Float64,var::Float64)
    var = (var-c2)/c1
    return log.((var)/(1-var)) #pribability transformation
end

#### metropolis-hastings calculation for theta
#inputs: prior data, data, MCMC iter vals, model, theta index, thets stepsize
# returns: next value for theta
function metropolis_theta(prior_data::PriorData,data::DataStr,
    vars::UpdatedVars, k::Int64,stepsize::Float64)
    c1 = 1.0              #scale of theta
    c2 = 0.0              #min value for theta

    current_theta = vars.theta[k]        #pull current theta
    current_gamma = g_inv(c1,c2,current_theta) #transform to gamma
    prop_gamma = rand(Normal(current_gamma,stepsize)) #propose new gamma
    prop_theta = g(c1,c2,prop_gamma)   #transform back to theta

    thetas_propose = copy(vars.theta)  #copy current thetas
    thetas_propose[k] = prop_theta     #replace theta at index k with prop val
    thetas_current = copy(vars.theta)  #copy current thetas

    lik_prop = lik(data,vars,thetas_propose)[1] #calc likelihood
    lik_current = lik(data,vars,thetas_current)[1]

    #calculate jump distribution values
    jump_current = pdf(Normal(current_gamma,stepsize),prop_gamma)*
    abs(-(c1)/((prop_theta+c2)*(-c1+prop_theta+c2)))
    jump_propose = pdf(Normal(prop_gamma,stepsize),current_gamma)*
    abs(-(c1)/((current_theta+c2)*(-c1+current_theta+c2)))

    #calculate acceptance
    ratio = min((lik_prop/lik_current)*jump_propose/jump_current, 1)
    accept = rand(Uniform(0,1))<ratio

    new_value = ifelse(accept,prop_theta,current_theta) #determine acceptance

    output = MetropolisInfo(new_value,ratio,accept)
    return output
end

#### special case of theta metropolis update for case where there are no x variables
function metropolis_theta_nox(prior_data::PriorData,data::DataStr,
    vars::UpdatedVars, k::Int64,stepsize::Float64,sum_y_2::Float64,sum_y_1::Float64,num_obs::Int)
    c1 = 1.0              #scale of theta
    c2 = 0.0              #min value for theta

    current_theta = vars.theta[k]        #pull current theta
    current_gamma = g_inv(c1,c2,current_theta) #transform to gamma
    prop_gamma = rand(Normal(current_gamma,stepsize)) #propose new gamma
    prop_theta = g(c1,c2,prop_gamma)   #transform back to theta

    thetas_propose = copy(vars.theta)  #copy current thetas
    thetas_propose[k] = prop_theta     #replace theta at index k with prop val
    thetas_current = copy(vars.theta)  #copy current thetas

    lik_prop = lik_nox(  data,vars,thetas_propose)[1] #calc likelihood
    lik_current = lik_nox(  data,vars,thetas_current)[1]

    #calculate jump distribution values
    jump_current = pdf(Normal(current_gamma,stepsize),prop_gamma)*
    abs(-(c1)/((prop_theta+c2)*(-c1+prop_theta+c2)))
    jump_propose = pdf(Normal(prop_gamma,stepsize),current_gamma)*
    abs(-(c1)/((current_theta+c2)*(-c1+current_theta+c2)))

    #calculate acceptance
    ratio = min((lik_prop/lik_current)*jump_propose/jump_current, 1)
    accept = rand(Uniform(0,1))<ratio

    new_value = ifelse(accept,prop_theta,current_theta) #determine acceptance

    output = MetropolisInfo(new_value,ratio,accept)
    return output
end
#### metropolis-hastings calculation for rho
# inputs: prior data, data, MCMC iter vals, model, stepsize, number of x and obs
function metropolis_rho(prior_data::PriorData,data::DataStr,vars::UpdatedVars,
     k::Int64,stepsize::Float64,nx::Int64,nobs::Int64)
    c1 = 1.0              #rho scale
    c2 = 0.0              #rho min val
    current_rho = vars.rho[k]     #pull current rho at index k
    current_gamma = g_inv(c1,c2,current_rho) #transform to gamma
    prop_gamma = rand(Normal(current_gamma,stepsize)) #propose new gamma
    prop_rho = g(c1,c2,prop_gamma)           #transform back to rho

    rhos_current = copy(vars.rho)            #copy current rhos
    rhos_propose = copy(vars.rho)            #copy current rhos
    rhos_propose[k] = prop_rho               #replace rho at k with prop val

    #calc delta prior pdf
    delta_prop = prior_delta(vars,data,rhos_propose,nx,nobs)[1]
    delta_current = prior_delta(vars,data,rhos_current,nx,nobs)[1]

    #calc jump distribution pdf
    jump_current = pdf(Normal(current_gamma,stepsize),prop_gamma)*
    abs(-(c1)/((prop_rho+c2)*(-c1+prop_rho+c2)))
    jump_propose = pdf(Normal(prop_gamma,stepsize),current_gamma)*
    abs(-(c1)/((current_rho+c2)*(-c1+current_rho+c2)))

    #calculate rho prior pdf
    prior_current = prior_rho(current_rho,prior_data,k)
    prior_prop = prior_rho(prop_rho,prior_data,k)

    #calculate acceptance
    ratio = min((delta_prop/delta_current)*
    jump_propose/jump_current, 1)
    accept = rand(Uniform(0,1))<ratio

    new_value = ifelse(accept,prop_rho,current_rho)  #determine acceptance

    output = MetropolisInfo(new_value,ratio,accept)
    return output
end
#### gibbs update for sig^2
# inputs: prior data, data, MCMC iter vals, number of x and obs
# returns: gibbs sample for sig^2
function gibbs_tau2(prior_data::PriorData,data::DataStr,vars::UpdatedVars,
    nx::Int64,nobs::Int64)
    alpha = prior_data.tau2.par1 #pull prior hyperparams
    beta = prior_data.tau2.par2
    delta = vars.delta           #pull discrepance values
    x = data.exp.x               #pull exp x vals

    C = correlation_construct(vars.rho,x,nx,nobs)  #calc correlation matrix

    par1 = alpha + 0.5*size(x)[1]   #calculation posterior params
    par2 = beta + 0.5*delta'*inv(C)*delta

    return rand(InverseGamma(par1,par2))   #sample from posterior
end
#### gibbs update for tau^2
# inputs: prior data, data, MCMC iter vals, model
# returns: gibbs sample for tau^2
function gibbs_sig2(prior_data::PriorData,data::DataStr,vars::UpdatedVars)
    alpha = prior_data.sig2.par1    #pull prior hyperparams
    beta = prior_data.sig2.par2      
    theta = vars.theta              #pull theta
    delta = vars.delta              #pull discrepancy
    x = data.exp.x                  # pull x and y
    y = data.exp.y

    #specify thetas for prediction
    theta_pred = Array{Float64}(undef,size(x)[1],length(theta))
    @inbounds for i in 1:size(x)[1]
        theta_pred[i,:] = theta
    end
    #predict model for current thetas
    eta = predict_y_all(theta')

    par1 = alpha + 0.5*size(x)[1]  #calcualte posterior params
    sse = Vector{Float64}(undef,size(y)[2])
    for i in 1:size(y)[2]
        sse[i] = (y[:,i] - eta - delta)'*(y[:,i] - eta - delta)
    end

    #par2 = beta + 0.5*((y - eta - delta)'*(y - eta - delta))[1,1]
    par2 = beta + 0.5*sum(sse)

    return rand(InverseGamma(par1,par2))  #sample from posterior
end
#### gibbs update for discrepancy function
# inputsL data, MCMC iter vals, model
function gibbs_delta(data::DataStr,vars::UpdatedVars)
    x = data.exp.x    #pull x and y
    y = data.exp.y

    #calc covar matrix
    C = vars.tau2[1,1]*correlation_construct(vars.rho,x,nx,nobs)

    sig2 = vars.sig2[1,1]   #pull theta and tau^2
    theta = vars.theta

    #generate matrix of current thetas for prediction
    theta_pred = Array{Float64}(undef,size(x)[1],length(theta))
    @inbounds for i in 1:size(x)[1]
        theta_pred[i,:] = theta
    end
    #prediction from surrogate model
    eta = predict_y_all(theta')
    #calculate values for posterior
    An = inv(C) + size(y)[2]^2*(1/(sig2))*Matrix(1.0I,size(x)[1],size(x)[1])

    covar = inv(An)
    covar = 0.5*(covar + covar') #ensures symmetry for stability
    bn_vec = Array{Float64}(undef,size(x)[1],size(y)[2])
    for i in 1:size(y)[2]
        bn_vec[:,i] = (y[:,i]-eta)
    end
    bn = size(y)[2]^2*1/sig2*mean(bn_vec,dims=2)
    return rand(MvNormal(vec(covar*bn),covar))  #sample from posterior
end

#### function to find optimal starting location of the thetas for MCMC sampling
# inputs: vectors of theta
# outputs: optimized values for theta
function model_opt(theta)
    #input = hcat(data.exp.x,repeat(theta',size(data.exp.x)[1],1))
    #println(input)
    output = predict_y_all(theta')
    #println(output)
    return sum((data.exp.y .- output).^2)
end

#get optimal starting location of thetas
theta_init = bboptimize(model_opt; SearchRange = (0.0,1.0), 
NumDimensions = ntheta, MaxSteps = 4000)
theta_init = best_candidate(theta_init)
for i in 1:ntheta
    estimate = scales.theta.min[i] + theta_init[i]*(scales.theta.max[i] - scales.theta.min[i])
    println("θ_$i Optimum = $estimate")
end

#### variable initialization function for MCMC
# inputs: data, number of MCMC samples, number of x and theta dims,
#                                      starting location for thetas
# returns: data structure to store MCMC samples


#### function to generate data structure for storing most recent MCMC vals
# inputs: most recent values for theta, delta, tau^2, sig^2, and rho
# returns: data struct containing most recent values
function update_vars(theta::Vector{Float64},delta::Vector{Float64},
    sig2::Float64,tau2::Float64,rho::Vector{Float64})
    return UpdatedVars(theta,delta,sig2,tau2,rho)
end
#### main MCMC function
# inputs: data, number of MCMC samples, prior data, model, bulk variable struct,
#           starting index, stopping index, stepsizes for theta and rho,
#           number of x and theta dims, number of exp obs
# returns: samples from MCMC
function mcmc(data::DataStr,nmcmc::Int64,prior_data::PriorData,
     bulk_vars::BulkVarsStruct,start::Int64,stop::Int64,
    stepsize::StepSize,nx::Int64,ntheta::Int64,nobs::Int64)
    #initialize MCMC iter vals
    step_vars = update_vars(bulk_vars.theta[start-1,:],bulk_vars.delta[start-1,:],
    bulk_vars.sig2[start-1],bulk_vars.tau2[start-1],bulk_vars.rho[start-1,:])

    if size(data.exp.x)[2] == 0
    
        #precomputations for likelihood
        sum_y_2 = sum(data.exp.y).^2
        sum_y_1 = sum(data.exp.y)
        num_obs = length(data.exp.y)


        @inbounds @showprogress 1 "Computing..." for i in start:stop
            @inbounds for j in 1:ntheta    #loop over theta dims
                # metropolis update for theta
                theta_step = metropolis_theta_nox(prior_data,data,step_vars,  j,
                stepsize.theta[j])
                step_vars.theta[j] = theta_step.new_value
                bulk_vars.theta[i,j] = step_vars.theta[j]
                bulk_vars.accept[i,j+nx] = theta_step.accept
                bulk_vars.ratio[i,j+nx] = theta_step.ratio
            end

            #gibbs update for tua^2
            step_vars.sig2 = gibbs_sig2(prior_data,data,step_vars)
            bulk_vars.sig2[i] = step_vars.sig2[1,1]
            #gibbs update for sig^2
        end
    else
        #loop from starting index to stopping index
        @inbounds @showprogress 1 "Computing..." for i in start:stop
            @inbounds for j in 1:ntheta    #loop over theta dims
                # metropolis update for theta
                theta_step = metropolis_theta(prior_data,data,step_vars,  j,
                stepsize.theta[j])
                step_vars.theta[j] = theta_step.new_value
                bulk_vars.theta[i,j] = step_vars.theta[j]
                bulk_vars.accept[i,j+nx] = theta_step.accept
                bulk_vars.ratio[i,j+nx] = theta_step.ratio
            end
            @inbounds for j in 1:nx       #loop over x dims
                # metropolis update for rho
                rho_step = metropolis_rho(prior_data,data,step_vars,  
                j,stepsize.rho[j],nx,nobs)
                step_vars.rho[j] = rho_step.new_value
                bulk_vars.rho[i,j] = step_vars.rho[j]
                bulk_vars.accept[i,j] = rho_step.accept
                bulk_vars.ratio[i,j] = rho_step.ratio
            end

            #gibbs update for tua^2
            step_vars.sig2 = gibbs_sig2(prior_data,data,step_vars)
            bulk_vars.sig2[i] = step_vars.sig2[1,1]
            #gibbs update for sig^2
            step_vars.tau2 = gibbs_tau2(prior_data,data,step_vars,nx,nobs)
            bulk_vars.tau2[i] = step_vars.tau2[1,1]
            #gibbs update for discrepancy
            step_vars.delta = gibbs_delta(data,step_vars)
            bulk_vars.delta[i,:] = step_vars.delta

            #println(step_vars.theta)
            #println(step_vars.rho)
            #println(step_vars.sig2)
            #println(step_vars.tau2)

        end
    end

    return bulk_vars
end
#### function to update stepsizes for stepsize calculation algorithm
# inputs: bulk variables, previous stepsize, starting and stopping index,
#            number of x and theta dims, number of observations
# returns: updated stepsizes for rhos and thetas
function update_stepsize(bulk_vars::BulkVarsStruct,old_stepsize::StepSize,
    start::Int64,stop::Int64,nx::Int64,ntheta::Int64,nobs::Int64)
    target = 0.2  #define target stepsize
    @inbounds for i in 1:ntheta  #loop over theta dims
        acceptance = mean(bulk_vars.accept[1:stop,i+nx]) #calc accpetance

        #adjust stepsize based on acceptance
        old_stepsize.theta[i] = old_stepsize.theta[i]
        if (acceptance-target)>0.01
            old_stepsize.theta[i] = old_stepsize.theta[i]*1.2
        end
        if (acceptance-target)>0.05
            old_stepsize.theta[i] = old_stepsize.theta[i]*1.5
        end
        if (acceptance-target)>0.2
            old_stepsize.theta[i] = old_stepsize.theta[i]*2
        end
        if (acceptance-target)>0.5
            old_stepsize.theta[i] = old_stepsize.theta[i]*3
        end
        if (-acceptance+target)>0.01
            old_stepsize.theta[i] = old_stepsize.theta[i]/20
        end
        if (-acceptance+target)>0.05
            old_stepsize.theta[i] = old_stepsize.theta[i]/40
        end
        if (-acceptance+target)>0.2
            old_stepsize.theta[i] = old_stepsize.theta[i]/70
        end
        if (-acceptance+target)>0.5
            old_stepsize.theta[i] = old_stepsize.theta[i]/200
        end
        #else
        #    old_stepsize.theta[i] = old_stepsize.theta[i]
    end
    target = 0.3
    @inbounds for i in 1:nx   #loop over x dims
        acceptance = mean(bulk_vars.accept[1:stop,i])  #calc acceptance
        #adjust stepsize based on acceptance
        old_stepsize.rho[i] = old_stepsize.rho[i]
        if (acceptance-target)>0.01
            old_stepsize.rho[i] = old_stepsize.rho[i]*1.2
        end
        if (acceptance-target)>0.05
            old_stepsize.rho[i] = old_stepsize.rho[i]*1.5
        end
        if (acceptance-target)>0.2
            old_stepsize.rho[i] = old_stepsize.rho[i]*2
        end
        if (acceptance-target)>0.5
            old_stepsize.rho[i] = old_stepsize.rho[i]*3
        end
        if (-acceptance+target)>0.01
            old_stepsize.rho[i] = old_stepsize.rho[i]/20
        end
        if (-acceptance+target)>0.05
            old_stepsize.rho[i] = old_stepsize.rho[i]/40
        end
        if (-acceptance+target)>0.2
            old_stepsize.rho[i] = old_stepsize.rho[i]/70
        end
        if (-acceptance+target)>0.5
            old_stepsize.rho[i] = old_stepsize.rho[i]/200
        end
        #else
        #    old_stepsize.rho[i] = old_stepsize.rho[i]
    end

    return old_stepsize
end
#### stepsize calculation algorithm
# inputs: data, number of total MCMC, prior data, model, number of MCMC per run,
#            number of x and theta dims, stepsize data type, number of obs
# returns: calculated stepsize for target acceptance
function auto_stepsize(data::DataStr,nruns::Int64,prior_data::PriorData,
     nsize::Int64,nx::Int64,ntheta::Int64,
    nobs::Int64)
    #initialize stepsize values
    old_stepsize = StepSize(repeat([1e-3],ntheta),repeat([1e-3],nx))


    bulk_vars = init_vars(data,nruns,nx,ntheta,theta_init) #initialize large bulk vars
    #initialize MCMC iter vals
    step_vars = update_vars(bulk_vars.theta[1,:],bulk_vars.delta[1,:],
    bulk_vars.sig2[1],bulk_vars.tau2[1],bulk_vars.rho[1,:])

    rep_vars = init_vars(data,nsize,nx,ntheta,theta_init) #initialize small bulk vars
    
    nrep = trunc(Int,nruns/nsize) #calculate number of repititions

    stepsize = Matrix(0.0I,nrep,nx+ntheta) #initalize stepsize and acceptance arrays
    acceptance = Matrix(0.0I,nrep,nx+ntheta)

    @inbounds @showprogress 1 "Computing..." for i in 1:nrep  #loop over repititions
        start = (i-1)*nsize+1+ifelse(i==1,1,0)   #calcualte start and stop index
        stop = i*nsize
        #run MCMC between start and stop
        bulk_vars = mcmc(data,nsize,prior_data,  bulk_vars,start,stop,
        old_stepsize,nx,ntheta,nobs)
        #calcualte stepsize and acceptance for rhos and thetas
        @inbounds for j in 1:nx   #loop over x dims
            stepsize[i,j] = old_stepsize.rho[j] #store most recent stepsize
            #define acceptance as mean over all previous acceptance
            acceptance[i,j] = mean(bulk_vars.accept[1:stop,j])
            #define old stepsize and mean over all previous stepsizes
            old_stepsize.rho[j] = mean(stepsize[1:i,j])
        end

        @inbounds for j in 1:ntheta
            stepsize[i,j+nx] = old_stepsize.theta[j]  #store most recent stepsize
            #define acceptance as mean over all previous acceptance
            acceptance[i,j+nx] = mean(bulk_vars.accept[1:stop,j+nx])
            #define old stepsize and mean over all previous stepsizes
            old_stepsize.theta[j] = mean(stepsize[1:i,j+nx])
        end
        #update stepsizes
        old_stepsize = update_stepsize(bulk_vars,old_stepsize,start,stop,nx,
        ntheta,nobs)
    end
    #plot results from algorithm over theta and rho
    for i in 1:nx
        (p1_1 = Plots.plot(stepsize[:,i],title="Stepsize vs Iteration for ρ_$i",
        legend=false))
        xlabel!("Algorithm Iteration")
        ylabel!("Algorithm Stepsize")
        display(p1_1)
        Plots.savefig(mdl*"rho_$i-stepsize.png")
        (p1_2 = Plots.plot(acceptance[:,i],title="Acceptance vs Iteration for ρ_$i",
        legend=false))
        xlabel!("Algorithm Iteration")
        ylabel!("Algorithm Acceptance")
        ylims!((0,1))
        display(p1_2)
        Plots.savefig(mdl*"rho_$i-acceptance.png")
    end
    for i in 1:ntheta
        (p2_1 = Plots.plot(stepsize[:,i+nx],title="Stepsize vs Iteration for θ_$i",
        legend=false))
        xlabel!("Algorithm Iteration")
        ylabel!("Algorithm Stepsize")
        Plots.savefig(mdl*"theta_$i-stepsize.png")
        display(p2_1)
        (p2_2 = Plots.plot(acceptance[:,i+nx],title="Acceptance vs Iteration for θ_$i",
        legend=false))
        xlabel!("Algorithm Iteration")
        ylabel!("Algorithm Acceptance")
        ylims!((0,1))
        Plots.savefig(mdl*"theta_$i-acceptance.png")
        display(p2_2)
    end
    #store finalized stepsizes
    burn = trunc(Int,0.5*nrep)
    final_stepsize = old_stepsize
    for j in 1:nx
        final_stepsize.rho[j] = mean(stepsize[burn:end,j])
    end
    for j in 1:ntheta
        final_stepsize.theta[j] = mean(stepsize[burn:end,j+nx])
    end
    return final_stepsize,stepsize,acceptance
end
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
################################### Main Code #####################################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

println("Start of Stepsize Calculation Algorithm")
#run stepsize algorithm
opt_stepsize = auto_stepsize(data,ntrial*ntrialburn,priors,  
ntrialburn,nx,ntheta,nobs)
println("Start of Main MCMC")

# initialize data struct to store MCMC results
bulk_vars = init_vars(data,nmcmc,nx,ntheta,theta_init)

#run main MCMC
results = mcmc(data,nmcmc,priors,  bulk_vars,2,nmcmc,
opt_stepsize[1],nx,ntheta,nobs)

#### print results
# inputs: results data
# returns: text file containing MCMC results
function print_results(results::BulkVarsStruct,nx,ntheta,nmcmc,nburn,mdl)
    results_mat = hcat(results.rho,results.theta,results.tau2,results.sig2)
    results_trim = results_mat[nburn:end,:]
    names_rho = ["rho_$i" for i in 1:nx]
    names_theta = ["theta_$i" for i in 1:ntheta]
    names_all = vcat(names_rho,names_theta,["sigma^2";"tau^2"])
    results_df = DataFrame(results_trim,names_all)
    CSV.write(mdl*"mcmc_results.csv",results_df)
    nothing
end


#print results
print_results(results,nx,ntheta,nmcmc,nburn,mdl)

#### post-porcessing of results
# inputs: results data struct, number of x and theta dims, number of burn,
#            data struct, number of mcmc iterations, data scaling struct,
#            model, number of exp obs
# Returns: plots and tables for result interpretation
function post(results::BulkVarsStruct,nx::Int64,ntheta::Int64,nburn::Int64,
    data::DataStr,nmcmc::Int64,scaling::Scaling, nobs::Int64)

    #scale theta, delta, sig^2, and tau^2
    #also convert sig^2 and tau^2 to sig and tau
    results.theta = results.theta.*(scaling.theta.max.-scaling.theta.min)' .+
     scaling.theta.min'
    results.tau2 = sqrt.(results.tau2).*(scaling.y.max-scaling.y.min)
    results.sig2 = sqrt.(results.sig2).*(scaling.y.max-scaling.y.min)
    results.delta = results.delta.*(scaling.y.max-scaling.y.min)

    #generate density and trace plaots for sig and tau
    (p3 = Plots.plot(results.sig2,legend=false))
    xlabel!("Iteration")
    ylabel!("τ Draw Value")
    title!("Data Model Error Trace Plot")
    display(p3)
    Plots.savefig(mdl*"tautrace.png")
    (p4 = density(results.sig2[nburn:end],legend=false))
    xlabel!("τ")
    ylabel!("Density")
    title!("Data Model Error Density")
    display(p4)
    Plots.savefig(mdl*"taudens.png")

    if nx > 0
        (p5 = Plots.plot(results.tau2,legend=false))
        xlabel!("Iteration")
        ylabel!("σ Draw Value")
        title!("Discrepancy Std. Dev. Trace Plot")
        display(p5)
        Plots.savefig(mdl*"sigtrace.png")
        p6 = density(results.tau2[nburn:end],legend=false)
        xlabel!("σ")
        ylabel!("Density")
        title!("Discrepancy Std. Dev. Density")
        display(p6)
        Plots.savefig(mdl*"sigdens.png")

        #plot discrepancy function
        mean_delta = Vector{Float64}(undef,size(data.exp.x)[1])
        q1 = Vector{Float64}(undef,size(data.exp.x)[1])
        q2 = Vector{Float64}(undef,size(data.exp.x)[1])
        for i in 1:length(mean_delta)
            mean_delta[i] = mean(results.delta[nburn:end,i])
            q1[i] = sort(results.delta[nburn:end,i])[round(Int,0.025*(nmcmc-nburn))]
            q2[i] = sort(results.delta[nburn:end,i])[round(Int,0.975*(nmcmc-nburn))]
        end
        p7 = Plots.scatter(mean_delta,label = false)
        Plots.plot!(q1,fillrange = q2,fillalpha=0.35,color="red",label="95% CI")
        #scatter!(delta_true,color="black")
        title!("Discrepancy Function")
        xlabel!("Index")
        ylabel!("δ(x)")
        display(p7)
        Plots.savefig(mdl*"delta.png")

        #calculate end and middle indices of delta
        endpoint = length(mean_delta)
        midpoint = trunc(Int,0.5*endpoint)

        #generate density and trace plots at beginning, middle and end indices of delta
        p8 = Plots.plot(results.delta[:,1],legend=false)
        title!("Discrepancy Trace Plot at First Point")
        xlabel!("Iteration")
        ylabel!("δ Draw Value")
        display(p8)
        Plots.savefig(mdl*"deltastarttrace.png")
        p9 = density(results.delta[nburn:end,1],legend=false)
        title!("Discrepancy Density at First Point")
        xlabel!("δ")
        ylabel!("Density")
        display(p9)
        Plots.savefig(mdl*"deltastartdens.png")

        p10 = Plots.plot(results.delta[:,midpoint],legend=false)
        title!("Discrepancy Trace Plot at Midpoint")
        xlabel!("Iteration")
        ylabel!("δ Draw Value")
        display(p10)
        Plots.savefig(mdl*"deltamidtrace.png")
        p11 = density(results.delta[nburn:end,midpoint],legend=false)
        title!("Discrepancy Density at Midpoint")
        xlabel!("δ")
        ylabel!("Density")
        display(p11)
        Plots.savefig(mdl*"deltamiddens.png")

        p12 = Plots.plot(results.delta[:,endpoint],legend=false)
        title!("Discrepancy Trace Plot at Last Point")
        xlabel!("Iteration")
        ylabel!("δ Draw Value")
        display(p12)
        Plots.savefig(mdl*"deltaendtrace.png")
        p13 = density(results.delta[nburn:end,endpoint],legend=false)
        title!("Discrepancy Density at Last Point")
        xlabel!("δ")
        ylabel!("Density")
        display(p13)
        Plots.savefig(mdl*"deltaenddens.png")
    end
    #theta_vec = Array{Float64}(undef,length(mean_delta),ntheta)
    #for i in 1:ntheta
    #    theta_vec[:,i] = 

    #generate density and trace plots for rho and theta
    for i in 1:ntheta
        p14 = Plots.plot(results.theta[:,i],legend=false)
        xlabel!("Iteration")
        ylabel!("θ_$i Draw Value")
        title!("θ_$i Trace Plot")
        display(p14)
        Plots.savefig(mdl*"theta_$i"*"trace.png")
        p15 = density(results.theta[nburn:end,i],legend=false)
        xlabel!("θ_$i")
        ylabel!("Density")
        title!("θ_$i Density")
        display(p15)
        Plots.savefig(mdl*"theta_$i"*"dens.png")
    end

    if nx > 0
        for i in 1:nx
            p16 = Plots.scatter(results.rho[:,i],legend=false)
            xlabel!("Iteration")
            ylabel!("ρ_$i Draw Value")
            title!("Correlation_$i Trace Plot")
            display(p16)
            Plots.savefig(mdl*"rho_$i"*"trace.png")
            p17 = density(results.rho[nburn:end,i],legend=false)
            xlabel!("ρ_$i")
            ylabel!("Density")
            title!("Correlation_$i Density")
            display(p17)
            Plots.savefig(mdl*"rho_$i"*"dens.png")
        end
        #generate correlation plots for rho and theta
        p18 = corrplot(results.rho[nburn:end,:], label = ["ρ_$i" 
        for i = 1:size(results.rho)[2]])
        
        Plots.savefig(mdl*"rho_corr.png")
        display(p18)
    end
    p19 = corrplot(results.theta[nburn:end,:], label = ["θ_$i" 
    for i = 1:size(results.theta)[2]])
        
    Plots.savefig(mdl*"theta_corr.png")
    
    display(p19)

    #generate plsterior estimates and 95% CI for variables
    post_sig2 = sort(results.sig2[nburn:end])  #sort all variables
    
    if nx > 0
        post_tau2 = sort(results.tau2[nburn:end])
    end
    
    post_theta = sort(results.theta[nburn:end,:],dims=1)
    
    if nx > 0
        post_rho = sort(results.rho[nburn:end,:],dims=1)
    end
    
    if nx == 0
        estimates = Matrix{Float64}(undef,4,ntheta + 2)
    else
        estimates = Matrix{Float64}(undef,4,nx+ntheta+3) #allocate array
    end

    lb = round(Int,0.025*(nmcmc-nburn))  #calc index for lower and upper bounds
    ub = round(Int,0.975*(nmcmc-nburn))
    
    #store 95% lower value, mean, median, and 95% uppper value for all variables
    if nx > 0
        for i in 1:nx
            estimates[1,i+1] = post_rho[lb,i]
            estimates[2,i+1] = mean(post_rho[:,i])
            estimates[3,i+1] = median(post_rho[:,i])
            estimates[4,i+1] = post_rho[ub,i]
        end
    end

    for i in 1:ntheta
        estimates[1,i+nx+1] = post_theta[lb,i]
        estimates[2,i+nx+1] = mean(post_theta[:,i])
        estimates[3,i+nx+1] = median(post_theta[:,i])
        estimates[4,i+nx+1] = post_theta[ub,i]
    end

    if nx > 0
        estimates[1,end-1] = post_sig2[lb]
        estimates[2,end-1] = mean(post_sig2)
        estimates[3,end-1] = median(post_sig2)
        estimates[4,end-1] = post_sig2[ub]
    
        estimates[1,end] = post_tau2[lb]
        estimates[2,end] = mean(post_tau2)
        estimates[3,end] = median(post_tau2)
        estimates[4,end] = post_tau2[ub]
    else
        estimates[1,end] = post_sig2[lb]
        estimates[2,end] = mean(post_sig2)
        estimates[3,end] = median(post_sig2)
        estimates[4,end] = post_sig2[ub]
    end

    #store labels for table
    if nx > 0
        rho_lab = ["ρ_$i" for i in 1:nx]
    end

    theta_lab = ["θ_$i" for i in 1:ntheta]

    if nx > 0
        var_lab = vcat(["Value"],rho_lab,theta_lab,["τ";"σ"])
    else
        var_lab = vcat(["value"],theta_lab,["τ"])
    end

    param_lab = ["95% Lower Bound";"Mean";"Median";"95% Upper Bound"]
    #round and convert to strings
    estimates = round.(estimates,digits=5)
    estimates = string.(estimates)
    #store labels in table
    estimates[:,1] = param_lab
    #generate table
    pretty_table(estimates;header = var_lab)
    #save table
    open(mdl*"param_estimates.txt","w") do f
        pretty_table(f,estimates;header = var_lab)
    end

    eta_pred = Array{Float64}(undef,length(results.sig2[nburn:end]),nobs)
    theta_pred = Array{Float64}(undef,nobs,ntheta)
    for i in 1:size(eta_pred)[1]
        for j in 1:nobs
            #println(i)
            theta_pred[j,:] = [results.theta[nburn+i-1,k] for k in 1:ntheta]
        end
        theta_pred = (theta_pred .- scaling.theta.min')./
        (scaling.theta.max .- scaling.theta.min)'
        eta_pred[i,:] = predict_y_all(theta_pred[1,:]')
    end
    eta_pred .= eta_pred .* (scaling.y.max .- scaling.y.min) .+ scaling.y.min
    model_pred = eta_pred .+ results.delta[nburn:end,:]
    model_pred .= sort(model_pred,dims=1)
    model_95_ub = model_pred[ub,:]
    model_95_lb = model_pred[lb,:]

    #store matrix of posterior estimates of theta
    #theta_pred = Array{Float64}(undef,nobs,ntheta)
    for i in 1:nobs
        theta_pred[i,:] = [mean(results.theta[nburn:end,k]) for k in 1:ntheta]
    end

    #scale back down to [0,1] for surrogate model prediction
    theta_pred = (theta_pred .- scaling.theta.min')./
    (scaling.theta.max .- scaling.theta.min)'

    #prediction from surrogate model
    mean_post_eta = predict_y_all(theta_pred[1,:]')
    #scale surrogate model prediction up to original response scale
    mean_post_eta .= mean_post_eta .* (scaling.y.max .-
    scaling.y.min) .+ scaling.y.min
    #add mean posterior discrepancy function
    if nx > 0
        mean_post_response  = mean_post_eta + mean_delta
    else
        mean_post_response = mean_post_eta
    end
    #scale experimental response data
    exp_response = data.exp.y .* (scaling.y.max -
    scaling.y.min) .+ scaling.y.min
    mean_post_response = vec(mean(model_pred,dims=1))
    #plot model and data together
    plot_label = vcat(["Experimental Observations #$i" for i in 1:size(data.exp.y)[2]],
    ["Mean Posterior Response"])

    #p20 = Plots.scatter([exp_response mean_post_response], 
    #label = permutedims(plot_label))
    p20 = Plots.scatter(mean_post_response,color=1,
    label="Mean Posterior Response")
    if length(size(exp_response)) > 1
        Plots.scatter!(exp_response[:,1],color=2,
        label="Experimental Observations")
        for i in 1:size(exp_response)[2]
            Plots.scatter!(exp_response[:,i],color=2,
            label=false)
        end
    else
        Plots.scatter!(exp_response,color=2,
        label="Experimental Observations")
    end
    Plots.plot!(model_95_lb,fillrange = model_95_ub, 
    fillalpha = 0.35,color="red",label = "95% CI")
    title!("Comparison Between Model and Data")
    ylabel!("Response")
    xlabel!("Index")
    display(p20)
    Plots.savefig(mdl*"modelvdata.png")
    #plot error between data and model
    pred_error = similar(model_pred)
    pred_error .= 0
    for i in 1:size(exp_response)[2]
        pred_error += (model_pred .- exp_response[:,i]')
    end
    pred_error .= sort(pred_error,dims=1)
    error_95_ub = pred_error[ub,:]
    error_95_lb = pred_error[lb,:]
    pred_error_mean = mean(pred_error,dims=1)
    p21 = Plots.scatter(pred_error_mean', label = "Error")
    Plots.plot!(error_95_lb,fillrange = error_95_ub,
    fillalpha=0.35,color="red",label="95% CI")
    title!("Error Between Model and Data")
    xlabel!("Index")
    ylabel!("Error")
    display(p21)
    Plots.savefig(mdl*"error.png")

    post_model_estimates_mat = hcat(model_95_lb,mean_post_eta,model_95_ub,
    error_95_lb,pred_error_mean',error_95_ub)
    post_model_estimates_names = ["95% Lower Bound Model Estimate",
    "Mean Model Estimate", "95% Upper Bound Model Estimate", 
    "95% Lower Bound Error Estimate", "Mean Error Estimate", 
    "95% Upper Bound Error Estimate"]
    post_model_estimates_df = DataFrame(post_model_estimates_mat,
    post_model_estimates_names)
    CSV.write(mdl*"posterior_model_estimates.csv",post_model_estimates_df)
    #display acceptance ratios for all thetas and rhos
    for i in 1:nx
        println("Acceptance Ratio for ρ_$i = ")
        println(mean(results.accept[:,i]))
    end

    for i in 1:ntheta
        println("Acceptance Ratio for θ_$i = ")
        println(mean(results.accept[:,i+nx]))
    end

    close_time = time()

    open(mdl*"report.txt","w") do f
        for i in 1:nx
            println(f,"Acceptance Ratio for ρ_$i = ")
            println(f,mean(results.accept[:,i]))
        end
    
        for i in 1:ntheta
            println(f,"Acceptance Ratio for θ_$i = ")
            println(f,mean(results.accept[:,i+nx]))
        end
        println(f,"Total Computation Time (seconds)")
        println(f,(close_time-init_time))
        println(f,"Total Computation Time (minutes)")
        println(f,(close_time-init_time)/60)
        println(f,"Total Computation Time (hours)")
        println(f,(close_time-init_time)/3600)
        println(f,"MCMC Computation Time (seconds)")
        println(f,(close_time-time_post_model))
        println(f,"MCMC Computation Time (minutes)")
        println(f,(close_time-time_post_model)/60)
        println(f,"MCMC Computation Time (hours)")
        println(f,(close_time-time_post_model)/3600)
        for i in 1:ntheta
            estimate = scales.theta.min[i] + theta_init[i]*(scales.theta.max[i] - scales.theta.min[i])
            println(f,"θ_$i Optimum = $estimate")
        end
        #println(f,"A total of $n_error out of $n_test test 
        #data points exhibit large error;  $n_percent%")
    end
    #println("A total of $n_error out of $n_test test data
     #points exhibit large error;  $n_percent%")
    println("Results Processing Completed")
end

println("Processing Results from MCMC")
#post-process results 
post(results,nx,ntheta,nburn,data,nmcmc,scales, nobs)
#=
test_prior = rand(InverseGamma(0.01,0.01),10000)
test_prior .= sqrt.(test_prior) .* (scales.y.max .- scales.y.min)
density(test_prior)
#density!(results.sig2)
xlims!(0,10)
=#

theta_bounds = [quantile(results.theta[(nburn+1):end,i],[0.025,0.975]) for i in 1:ntheta]
end
