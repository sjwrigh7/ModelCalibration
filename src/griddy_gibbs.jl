using Statistics
using LinearAlgebra
using Distributions
using DelimitedFiles
using Parameters
using DataFrames
using ProgressMeter
using Plots
using LaTeXStrings
using StatsPlots

#include("calibration_init.jl")
include("structs.jl")
include("misc_functions.jl")
include("gaussian_process_kernel.jl")
include("precomputation.jl")
include("initialization.jl")
include("sampling_functions.jl")
include("griddy_posteriors.jl")

cd(@__DIR__)

#declare dimensions of each variable type
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
alpha_tau2 = 0.00000001
beta_tau2 = 0.00000001
#discrepancy variance
alpha_sig2 = 0.00000001
beta_sig2 = 0.00000001
#discrepancy correlation
a_rho = 1.0
b_rho = 1.0

nys = [6,15,27,54]
nsims = [4,7,10,13,19,25,37,49]
for ny in nys
for nsim in nsims
for disc in ["disc","nodisc"]
#ny = 6
#for ny in nys
theta_hist_bins = 49
#read in files for computer model and experimental data
pth_grid = "/home/stephenw/Nextcloud/Documents/engr/PhD/calibration code/hannahs paper/redux/v2grid/all_grids/"
pth_exp = "/home/stephenw/Nextcloud/Documents/engr/PhD/calibration code/hannahs paper/redux/$disc/simulated experimental data/"
trad_design_path = "/run/media/stephenw/More\\x20Storage/palmetto_backup/expobs/"


save_pth = "/home/stephenw/Nextcloud/Documents/engr/PhD/calibration code/hannahs paper/redux/v2results/$disc/ny$ny-nsim$nsim/"
println(save_pth)

if !isdir(save_pth)
    println("making path")
    mkpath(save_pth)
end

#mdl = "ny_$ny-"

#trad_csv_path = "/home/stephenw/Nextcloud/Documents/engr/PhD/calibration code/hannahs paper/bayes_results/"
trad_csv_path = "/home/stephenw/Nextcloud/Documents/engr/PhD/calibration code/griddy_gibbs/test_res/comparison/trad/data_type/nodisc/"
#trad_design_path = "/home/stephenw/Nextcloud/Documents/engr/PhD/calibration code/hannahs paper/redux/disc/simulated experimental data/"

#data in original scale
design_scale = readdlm(pth_grid*"design-ny_$ny-nsim_$nsim.txt",'\t') #read in original scale simulation 
#                                               design
#design_scale = unique(design_scale[:,4:end],dims=1)
simobs_scale = readdlm(pth_grid*"simobs-ny_$ny-nsim_$nsim.txt",'\t') #read in original scale simulation 
#                                               response
#simobs_scale = transpose(reshape(simobs_scale,15,700))
simobs_scale = simobs_scale[:,end]           #ensure this is a vector
expobs_scale = readdlm(pth_exp*"ny_$ny-expobs_$disc.txt",'\t') #read in original scale experimental
#                                               data
#expobs_scale = readdlm(pth_exp*"model_exphfl_soi-25_ny-$ny"*"_nsim-1_disc-40_expobs.txt",',')
######################################################################

nobs,nrep,nloc,scales,data,priors = setup(design_scale,simobs_scale,
expobs_scale,nx,ntheta,alpha_tau2,beta_tau2,alpha_sig2,beta_sig2,a_rho,b_rho)

################
#Precomputations
################
grid_info = GridData(
    sig_star2 = GridPar(
        density=40,
        bounds = ScalePar(
            min = 1e-2,#1e-30,
            max = 10000.0#1e-29
        )
    ),
    rho = GridPar(
        density=10,
        bounds = ScalePar(
            min=0.0001,
            max = 0.9999
        )
    )
)

c_sse,log_det_sig,sig_design = preallocate(data,grid_info)

precompute!(data,c_sse,log_det_sig,sig_design,nx)

#c_sse = c_sse[:,1,1]

sampling_vars = init_vars(data,nmcmc)

griddy_gibbs!(data,sampling_vars,c_sse,log_det_sig,sig_design,priors)

#############
scaled_tau = sqrt.(sampling_vars.tau2[nburn:end]) .* (scales.y.max - scales.y.min)

#=
p_tau_hist = Plots.histogram(scaled_tau,label=false)
title!("Histogram of Data Error StDev")
xlabel!(L"$\tau$ Value")
ylabel!("Count")
display(p_tau_hist)
Plots.savefig(p_tau_hist,save_pth*mdl*"tau-hist.png")

p_tau_trace = Plots.plot(scaled_tau,label=false)
title!("Trace Plot of Data Error StDev")
xlabel!("MCMC Iteration")
ylabel!(L"$\tau$ Value")
display(p_tau_trace)
Plots.savefig(p_tau_trace,save_pth*mdl*"tau-trace.png")
=#
theta_post = data.sim.theta[sampling_vars.theta[nburn:end],:] .*
(scales.theta.max .- scales.theta.min)' .+ scales.theta.min'
#=
trad_mdl = "ny_$ny-"
trad_mdl_design = "model_exphfl_soi-25_ny-$ny"*"_nsim-343_disc-40_"
trad_res = readdlm(trad_csv_path*trad_mdl*"mcmc_results.csv",',')
trad_design = readdlm(trad_design_path*trad_mdl_design*"design.txt",',')
trad_theta_min = vec(minimum(design[:,(nx+1):end],dims=1))
trad_theta_max = vec(maximum(design[:,(nx+1):end],dims=1))
trad_thetas = trad_res[2:end,(nx+1):(nx+ntheta)] .* (theta_max' .- theta_min') .+ theta_min'

theta_bounds_gg = [quantile(theta_post[:,i],[0.025,0.975]) for i in 1:ntheta]
theta_bounds_trad = [quantile(trad_thetas[:,i],[0.025,0.975]) for i in 1:ntheta]

open(save_pth*mdl*"95CI.txt","w") do f
    for i in 1:ntheta
        println(f,"θ $i 95% CI, GG")
        println(f,theta_bounds_gg[i])
        println(f,"θ $i 95% CI, M-H")
        println(f,theta_bounds_trad[i])
    end
end

for i in 1:ntheta
    p_theta_hist = Plots.histogram(theta_post[:,i],label="GG Sampler",normalize=:probability,color=1,linecolor=1,alpha=0.5)
    title!(L"Histogram of $\theta_{%$i}$")
    xlabel!(L"$\theta_{%$i}$ Value")
    ylabel!("Density")
    #density!(theta_post[:,i],bandwidth=10,label=false,color=1,trim=true)
    Plots.histogram!(trad_thetas[:,i],label="M-H Sampler",normalize=:probability,color=2,linecolor=2,alpha=0.5)
    #density!(trad_thetas[:,i],basndwidth=0.1,label=false,color=2,trim=true)
    display(p_theta_hist)
    Plots.savefig(p_theta_hist,save_pth*mdl*"theta_$i-hist.png")

    p_theta_trace = Plots.plot(theta_post[:,i],label=false)
    title!(L"Trace Plot of $\theta_{%$i}$")
    xlabel!("MCMC Iteration")
    ylabel!(L"$\theta_{%$i}$ Value")
    display(p_theta_trace)
    Plots.savefig(p_theta_trace,save_pth*mdl*"theta_$i-trace.png")
end
=#

rho_post = sig_design[sampling_vars.rho[nburn:end],1,1]

#=
p_rho_hist = Plots.histogram(rho_post,label=false)
title!("Histogram of Discrepancy Correlation")
xlabel!(L"$\rho$ Value")
ylabel!("Count")
display(p_rho_hist)
Plots.savefig(p_rho_hist,save_pth*mdl*"rho-hist.png")

p_rho_trace = Plots.plot(rho_post,label=false)
title!(L"Trace Plot of Discrepancy Correlation")
xlabel!("MCMC Iteration")
ylabel!(L"$\rho$ Value")
display(p_rho_trace)
Plots.savefig(p_rho_trace,save_pth*mdl*"rho-trace.png")
=#

sig_post = sqrt.(sig_design[1,sampling_vars.sig_star2[nburn:end],2]) .* scaled_tau

#=
p_sig_hist = Plots.histogram(sig_post,label=false)
title!("Histogram of Discrepancy StDev")
xlabel!(L"$\sigma$ Value")
ylabel!("Count")
display(p_sig_hist)
Plots.savefig(p_sig_hist,save_pth*mdl*"sig-hist.png")

p_sig_trace = Plots.plot(sig_post,label=false)
title!("Trace Plot of Discrepancy StDev")
xlabel!("MCMC Iteration")
ylabel!(L"$\sigma$ Value")
display(p_sig_trace)
Plots.savefig(p_sig_trace,save_pth*mdl*"sig-trace.png")
=#

writedlm(save_pth*"theta_save.txt",theta_post)
writedlm(save_pth*"tau_save.txt",scaled_tau)
writedlm(save_pth*"rho_save.txt",rho_post)
writedlm(save_pth*"sig_save.txt",sig_post)
writedlm(save_pth*"y_wide.txt",data.sim.y[:,sampling_vars.theta[nburn:end]]')
writedlm(save_pth*"truth_wide.txt",data.exp.y')
end
end
end