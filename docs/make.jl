using ModelCalibration
using Documenter

DocMeta.setdocmeta!(ModelCalibration, :DocTestSetup, :(using ModelCalibration); recursive=true)

makedocs(;
    modules=[ModelCalibration],
    authors="Stephen Wright",
    sitename="ModelCalibration.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
