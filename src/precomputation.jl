#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
####################### Define Precomputation Functions ###########################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

"""
    scaled_sse(y_hat::Vector{Float64},sigma_inv::Array{Float64},y_bar::Vector{Float64},nobs::Int)
Function to calculate the exponent in the likelihood function.

---
Keyword arguments
* `y_hat::Vector{Float64}` A Vector of length n containing the computer model's estimates of the experimental responses for a setting of θ.
* `sigma_inv::Array{Float64}` A n by n Array describing the covariance structure of the data model.
* `y_exp::Array{Float64}` A Vector of length n containing the mean experimental observations, averaged of all m independent observations.
* `nobs::Int` A scalar integer describing the number of independent observations of the data (m).

---
Returns
* `sse::Float64` A scalar Float describing the sum of squared error between `y_hat` and `y_exp`, scaled by the covariance matrix.
"""
function scaled_sse(y_hat::Vector{Float64},sigma_inv::Array{Float64},y_bar::Vector{Float64},nobs::Int)
    sse = nobs*(y_hat-y_bar)'*sigma_inv*(y_hat-y_bar)
    return sse
end

"""
    preallocate(data::DataStr,grid::GridData)
Function to preallocate Arrays for precomputation in Griddy Gibbs approach.

---
Keyword arguments
* `data::DataStr` A data structure containing the computer and experimental data.
* `grid::GridData` A data structure specifying the bounds and number of grid points for σ*^2 and ρ.

---
Returns
* `c_sse::Array{Float64}` An initialized Array to store the precomputed values of the scaled SSE.
* `sig_det::Array{Float64}` An initialized Array to store the precomputed values of det(Σ).
* `sig_design::Array{Float64}` An initialized Array to store the ρ and σ*^2 inputs for Σ.
* `rho::Array{Float64}` An initialized Array to store the values of ρ for precomputation.
* `sig_star2::Vector{Float64}` An initialized Vector to store the values of σ*^2 for precomputation.
"""
function preallocate(data::DataStr,grid::GridData)
    nsim = size(unique(data.sim.theta,dims=1))[1]

    rho = Vector{Float64}(undef,grid.rho.density)
    sig_star2 = Vector{Float64}(undef,grid.sig_star2.density)

    sig_det = Array{Float64}(undef,grid.rho.density,grid.sig_star2.density)
    sig_design = Array{Float64}(undef,grid.rho.density,grid.sig_star2.density,2)

    c_sse = Array{Float64}(undef,nsim,grid.rho.density,grid.sig_star2.density)

    rho_step = (grid.rho.bounds.max - grid.rho.bounds.min)/(grid.rho.density-1)
    rho = (:)(grid.rho.bounds.min,rho_step,grid.rho.bounds.max)
    
    #sig_star2_step = (log(grid.sig_star2.bounds.max) - log(grid.sig_star2.bounds.min))/
    #(grid.sig_star2.density-1)
    #sig_star2 = exp.((:)(log(grid.sig_star2.bounds.min),sig_star2_step,log(grid.sig_star2.bounds.max)))
    sig_star2_step = (sqrt(grid.sig_star2.bounds.max) - sqrt(grid.sig_star2.bounds.min))/
    (grid.sig_star2.density-1)
    sig_star2 = ((:)(sqrt(grid.sig_star2.bounds.min),sig_star2_step,sqrt(grid.sig_star2.bounds.max))).^2
    #sig_star2_step = (grid.sig_star2.bounds.max - grid.sig_star2.bounds.min)/(grid.sig_star2.density - 1)
    #sig_star2 = (:)(grid.sig_star2.bounds.min,sig_star2_step,grid.sig_star2.bounds.max)

    for j in 1:length(sig_star2)
        for i in 1:size(rho)[1]
            sig_design[i,j,1] = rho[i]
            sig_design[i,j,2] = sig_star2[j]
        end
    end

    return c_sse,sig_det,sig_design
end

"""
    precompute!(data::DataStr,c_sse::Array{Float64},sig_inv::Float64,sig_det::Vector{Float64},sig_design::Array{Float64},nx::Int)
Function to precompute required values for Griddy Gibbs calibration.

---
Keyword arguments
* `data::DataStr` Struct containing computer simulator and experimental data.
* `c_sse::Array{Float64}` Array to store precomputed values of the SSE scaled by the covariance matrix.
* `sig_det::Vector{Float64}` Array to store the precomputed values for the determinant of the covariance matrix.
* `sig_design::Array{Float64}` Array to containing the ρ and σ*^2 input values for computing Σ.
"""
function precompute!(data::DataStr,c_sse::Array{Float64},
    sig_det::Array{Float64},sig_design::Array{Float64},nx::Int)
    nloc = size(data.exp.x)[1]
    sigma = Array{Float64}(undef,nloc,nloc)
    sig_inv = similar(sigma)
    ident = Matrix(1.0I,nloc,nloc)

    y_bar = vec(mean(data.exp.y,dims=2))
    nobs = size(data.exp.y)[2]

    @showprogress 1 "Precomputing..." for j in 1:size(sig_design)[2] #loop over σ*^2
        for i in 1:size(sig_design)[1] #loop over ρ
            rho_vec = repeat([sig_design[i,j,1]],nx)
            sigma .= ident .+ sig_design[i,j,2]*correlation_construct(rho_vec,data.exp.x,nx,nloc)
            sig_det[i,j] = log(det(sigma)^(-nobs/2))
            sig_inv .= inv(sigma)
            for k in 1:size(data.sim.y)[2]
                c_sse[k,i,j] = scaled_sse(data.sim.y[:,k],sig_inv,y_bar,nobs)
            end
        end
    end
end