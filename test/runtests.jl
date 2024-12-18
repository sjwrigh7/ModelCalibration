path = "/home/stephenw/.julia/dev/ModelCalibration"
cd(path)
using Pkg
Pkg.activate(".")
Pkg.develop(path=path)
using Revise
using ModelCalibration
using Distributions
using LinearAlgebra
#using GaussianProcesses
#using BlackBoxOptim
using Test

function format_exp(expobs,gauss)
    out = repeat(expobs,size(gauss)[2])
    gauss = Vector(vec(gauss))
    out[:,end] .= out[:,end] .+ gauss
    return out
end

function true_fn(theta::Vector{Float64})
    y = prod(theta .* sin.(theta))
    return y
end

function true_fn(x::Vector{Float64},theta::Vector{Float64})
    y = sum(theta .* sin.(x))
    return y
end

@testset "griddy Gibbs" begin
    include("test.jl")
end

@testset "continuous" begin
    include("test2.jl")
end
#TODO add grid responses to output of this function

#TODO add tests for sampling scheme

#TODO add tests for continuous method
# initializing variables at theta optimum
# step size algorithm
# sampling scheme

#TODO add tests for post processing
