cd(@__DIR__)
using Pkg
Pkg.activate(".")

using ModelCalibration
using Documenter

DocMeta.setdocmeta!(ModelCalibration, :DocTestSetup, :(using ModelCalibration); recursive=true)

makedocs(;
    modules=[ModelCalibration],
    authors="sjwrigh7",
    sitename="ModelCalibration",
    #repo="/home/stephenw/.julia/dev/ModelCalibration",
    #remotes=nothing,
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Setup and General Functions" => "setup.md",
        "Griddy Gibbs Calibration" => "griddy.md",
        "Traditional Bayesian Calibration" => "traditional.md"
    ],
)
