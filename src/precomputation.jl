#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
####################### Define Precomputation Functions ###########################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

"""
    scaled_sse(y_hat::Vector{Float64},sigma_inv::Array{Float64,2},y_bar::Vector{Float64},nobs::Int)
Function to calculate the exponent in the likelihood function.
Implementation for a multivariate normal distribution data model.

---
Positional arguments
* `y_hat::Vector{Float64}` A Vector of length n containing the computer model's estimates of the experimental responses for a setting of θ.
* `sigma_inv::Array{Float64}` A n by n Array describing the covariance structure of the data model.
* `y_bar::Array{Float64}` A Vector of length n containing the mean experimental observations, averaged of all m independent observations.
* `nobs::Int` A scalar integer describing the number of independent observations of the data (m).

---
Returns
* `sse::Float64` A scalar Float describing the sum of squared error between `y_hat` and `y_bar`, scaled by the covariance matrix.
"""
function scaled_sse(y_hat::Vector{Float64},sigma_inv::Array{Float64},y_bar::Vector{Float64},
            nobs::Int)

    sse = nobs*(y_hat-y_bar)'*sigma_inv*(y_hat-y_bar)
    return sse
end

"""
    scaled_sse(y_hat::Vector{Float64},y::Array{Float64})
Function to calculate the exponent in the likelihood function.
Implementation for a univariate normal distribution data model.

---
Positional arguments
* `y_hat::Vector{Float64}` A Vector of length n containing the computer model's estimates of the experimental responses for a setting of θ.
* `y::Array{Float64}` A Vector of length n containing the experimental responses.

---
Returns
* `sse::Float64` A scalar Float describing the sum of squared error between `y_hat` and `y`.
"""
function scaled_sse(y_hat::Float64,y::Array{Float64})
    sse = sum((y_hat .- y).^2)
    return sse
end

"""
    preallocate(theta_grid::Array{Float64,2},sig_grid::GridData,nx::Int)
Function to preallocate Arrays for precomputation in Griddy Gibbs approach.

---
Positional arguments
* `theta_grid::Array{Float64,2}` Grid of theta values for the griddy Gibbs sampler.
* `sig_grid::GridData` A data structure specifying the bounds and number of grid points for ϕ and ρ.
* `nx::Int`

---
Returns
* `c_sse::Array{Float64}` An initialized Array to store the precomputed values of the scaled SSE.
* `sig_det::Array{Float64}` An initialized Array to store the precomputed values of det(Σ).
* `sig_design::Array{Float64}` An initialized Array to store the ρ and ϕ inputs for Σ.
* `rho::Array{Float64}` An initialized Array to store the values of ρ for precomputation.
* `phi::Vector{Float64}` An initialized Vector to store the values of ϕ for precomputation.

---
Details
Preallocates Arrays for the griddy Gibbs precomputation. If the data model is univariate, `nothing` is returned for the integrated discrepancy covariance marix variables.
"""
function preallocate(theta_grid::Array{Float64,2},sig_grid::GridData,nx::Int)
    nsim = size(unique(theta_grid,dims=1))[1]
    
    if nx > 0
        rho = Vector{Float64}(undef,sig_grid.rho.density)
        phi = Vector{Float64}(undef,sig_grid.phi.density)

        sig_det = Array{Float64}(undef,sig_grid.rho.density,sig_grid.phi.density)
        sig_design = Array{Float64}(undef,sig_grid.rho.density,sig_grid.phi.density,2)

        c_sse = Array{Float64}(undef,nsim,sig_grid.rho.density,sig_grid.phi.density)

        rho_step = (sig_grid.rho.bounds.max - sig_grid.rho.bounds.min)/(sig_grid.rho.density-1)
        rho = (:)(sig_grid.rho.bounds.min,rho_step,sig_grid.rho.bounds.max)
        
        #phi_step = (log(grid.phi.bounds.max) - log(grid.phi.bounds.min))/
        #(grid.phi.density-1)
        #phi = exp.((:)(log(grid.phi.bounds.min),phi_step,log(grid.phi.bounds.max)))
        phi_step = (sqrt(sig_grid.phi.bounds.max) - sqrt(sig_grid.phi.bounds.min))/
        (sig_grid.phi.density-1)
        phi = ((:)(sqrt(sig_grid.phi.bounds.min),phi_step,sqrt(sig_grid.phi.bounds.max))).^2
        #phi_step = (grid.phi.bounds.max - grid.phi.bounds.min)/(grid.phi.density - 1)
        #phi = (:)(grid.phi.bounds.min,phi_step,grid.phi.bounds.max)

        for j in eachindex(phi)
            for i in axes(rho)[1]
                sig_design[i,j,1] = rho[i]
                sig_design[i,j,2] = phi[j]
            end
        end

        return c_sse,sig_det,sig_design
    else
        c_sse = Vector{Float64}(undef,nsim)
        return c_sse,nothing,nothing
    end
end

"""
    precompute!(grid_response::Array{Float64,2},data::DataStr,c_sse::Array{Float64,3},
    sig_det::Array{Float64,2},sig_design::Array{Float64,3},nx::Int)
Function to precompute required values for Griddy Gibbs calibration.
Implementation for a multivariate normal distribution data model.
---
Positional arguments
* `grid_response::Array{Float64,2}` Response grid from surrogate model.
* `data::DataStr` Struct containing computer simulator and experimental data.
* `c_sse::Array{Float64,3}` Array to store precomputed values of the SSE scaled by the covariance matrix.
* `sig_det::Array{Float64,2}` Array to store the precomputed values for the determinant of the covariance matrix.
* `sig_design::Array{Float64,3}` Array to containing the ρ and ϕ input values for computing Σ.
* `nx::Int` The number of x variables in the data.
"""
function precompute!(grid_response::Array{Float64,2},data::DataStr,c_sse::Array{Float64,3},
            sig_det::Array{Float64,2},sig_design::Array{Float64,3},nx::Int)

    nloc = size(data.exp.x)[1]
    sigma = Array{Float64}(undef,nloc,nloc)
    sig_inv = similar(sigma)
    ident = Matrix(1.0I,nloc,nloc)

    y_bar = vec(mean(data.exp.y,dims=2))
    nreps = size(data.exp.y)[2]

    @showprogress 1 "Precomputing..." for j in 1:size(sig_design)[2] #loop over ϕ
        for i in 1:size(sig_design)[1] #loop over ρ
            rho_vec = repeat([sig_design[i,j,1]],nx)
            sigma .= ident .+ sig_design[i,j,2]*correlation_construct(rho_vec,data.exp.x,nx,nloc)
            sig_det[i,j] = log(det(sigma)^(-nloc/2))
            sig_inv .= inv(sigma)
            for k in 1:size(grid_response)[1]
                c_sse[k,i,j] = scaled_sse(grid_response[k,:],sig_inv,y_bar,nreps)
            end
        end
    end
end

"""
    precompute!(grid_response::Array{Float64,2},data::DataStr,c_sse::Vector{Float64},
    sig_det::Nothing,sig_design::Nothing,nx::Int)
Function to precompute required values for Griddy Gibbs calibration.
Implementation for a univariate normal distribution data model.
This uses the same input arguments as the MVN implementation to allow using a similar function call.
The key diffence is that this accepts a Vector for c_sse instead of an `Array{Float64,3}` and the variables for the corrleation structure are `Nothing`.
These types are automatically determined by and returned from the `preallocate()` function, allowing for identical function calls.


---
Positional arguments
* `grid_response::Array{Float64,2}` Response grid from surrogate model.
* `data::DataStr` Struct containing computer simulator and experimental data.
* `c_sse::Vector{Float64}` Array to store precomputed values of the SSE scaled by the covariance matrix.
* `sig_det::Nothing` Dummy argument for function call.
* `sig_design::Nothing` Dummy variable for function call.
* `nx::Int` Dummy variable for function call.
"""
function precompute!(grid_response::Array{Float64,2},data::DataStr,c_sse::Vector{Float64},
    sig_det::Nothing,sig_design::Nothing,nx::Int)

    @showprogress 1 "Precomputing..." for k in 1:size(grid_response)[1]
        c_sse[k] = scaled_sse(grid_response[k,1],data.exp.y)
    end
end