cd(dirname(@__FILE__))
using DelimitedFiles
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
################################# Specify Inputs ##################################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
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
alpha_sig2 = 0.00000001
beta_sig2 = 0.00000001
#discrepancy variance
alpha_tau2 = 0.00000001
beta_tau2 = 0.00000001
#discrepancy correlation
a_rho = 1.0
b_rho = 1.0

#read in files for computer model and experimental data
pth_grid = "/home/stephenw/Nextcloud/Documents/engr/PhD/calibration code/hannahs paper/redux/grid/"
pth_exp = "/home/stephenw/Nextcloud/Documents/engr/PhD/calibration code/hannahs paper/redux/disc/simulated experimental data/"

mdl = "main_"
#data in original scale
design_scale = readdlm(pth_grid*"design-ny_6-nsim_49.txt",'\t') #read in original scale simulation 
#                                               design
#design_scale = unique(design_scale[:,4:end],dims=1)
simobs_scale = readdlm(pth_grid*"simobs-ny_6-nsim_49.txt",'\t') #read in original scale simulation 
#                                               response
#simobs_scale = transpose(reshape(simobs_scale,15,700))
simobs_scale = simobs_scale[:,end]           #ensure this is a vector
expobs_scale = readdlm(pth_exp*"model_exphfl_soi-25_ny-6_nsim-1_disc-40_expobs.txt",',') #read in original scale experimental
#                                               data
#=
names = expobs_scale[1,1:end]
splits = split.(names,"TC")
order = sortperm([parse(Int,splits[i][2]) for i in 1:length(splits)])
temp = expobs_scale[2:12,order]

#cal_delta = vec(readdlm("disc.txt",','))

function set_expobs(temp,design_scale)
expobs_scale = Array{Float64}(undef,Int(sizeof(temp)/8),4)

ita = 0
for i in 1:size(temp)[1], j in 1:size(temp)[2]
    ita += 1
    expobs_scale[ita,1:3] = design_scale[j,1:3]
    expobs_scale[ita,4] = temp[i,j]
end
return expobs_scale
end
expobs_scale = set_expobs(temp,design_scale)
#select = [4,5,6,7,9,10,11]
#select = [1,13]
#select = [1,13]
select = [1,2,3,4,5,6,7,9,10,11,12,13]
dat_design = Array{Int64}(undef,700,length(select))
dat_expobs = Array{Int64}(undef,size(temp)[1],length(select))
for i in 1:length(select)
    dat_design[:,i] = collect(select[i]:15:700*15)
    dat_expobs[:,i] = collect(select[i]:15:size(temp)[1]*15)
end

design_scale = design_scale[vec(dat_design'),:]
expobs_scale = expobs_scale[vec(dat_expobs'),:]
#expobs_scale[:,end] -= repeat(cal_delta,size(temp)[1])

simobs_scale = simobs_scale[:,select]
simobs_scale = Vector(vec(simobs_scale'))

#mdl = "test"
#design_scale = 0.0:10.0:140.0
#simobs_scale = 0.0:10.0:140.0
#expobs_scale = 25.0 .+ rand(Normal(0,0.5),200)
=#
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
################################# End of Inputs ###################################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

#=
open(mdl*"design_true1.txt","w") do f
    for i in 1:size(design_scale)[1]
        for j in 1:size(expobs_scale)[1]
            temp = vec(hcat(expobs_scale[j,1:3]',design_scale[i,:]'))
            println(f,join(temp,","))
        end
    end
end
=#
