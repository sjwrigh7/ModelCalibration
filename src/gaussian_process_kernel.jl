#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
########################## Define GP Kernel Functions #############################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

"""
    correlation_kernel(x1::Vector{Float64},x2::Vector{Float64},rho::Vector{Float64},nx::Int64)
Function to calculate the correlation value between two locations, `x1` and `x2`, given the correlation parameters, `rho`.

---
Keyword arguments
* `x1::Vector{Float64}` Vector of settings at the first location, length nx.
* `x2::Vector{Float64}` Vector of settings at the second location, length nx.
* `rho::Vector{Float64}` Vector of correlation parameters (ρ) across the different dimensions of x, length nx.
* `nx::Int64` Integer specifying the number of x dimensions.

---
Returns
* `correlation::Float64` A float describing the correlation between the two locations `x1` and `x2` given `rho`.

---
Details
Correlation is calculated as ∏[ρ^(4(x1-x2)^2)], where the product is taken over the number of x dimensions.
"""
function correlation_kernel(x1::Vector{Float64},x2::Vector{Float64},
    rho::Vector{Float64},nx::Int64)
    correlations = Vector{Float64}(undef,nx) #allocate vector to store correlation 
    #                                           in each dimension of x
    @inbounds for i in 1:nx                 #loop over x dimensions
        correlations[i] = rho[i]^(4*(x1[i]-x2[i])^2) #calculate the correlation in
        #                                               dimension
    end
    correlation = prod(correlations)
    return correlation                       #effective correlation is the product
end

"""
    correlation_construct(rho::Vector{Float64},x::Array{Float64},nx::Int64,nobs::Int64)
Function to construct the correlation Matrix for a set of x locations.

---
Keyword arguments
* `rho::Vector{Float64}` Vector of correlation parameters (ρ) across the different dimensions of x, length nx.
* `x::Array{Float64}` Array of x locations for calculating the correlation Matrix, dimensions nobs and nx.
* `nx::Int64` The number of x dimensions.
* `nobs::Int64` The number of data points for calculating the correlation Matrix.

---
Returns
* `correlation::Array{Float64}` The correlation matrix of x, given ρ.

---
Details
An nobs by nobs Matrix is initialized and, at each [i,j] index in the matrix, `correlation_kernel` is called to calculate the correlation given ρ, x[i,:], and x[j,:].
The square root of the machine precision for a Float64 is added to the diagonal elements of the Matrix for stability purposes.
"""
function correlation_construct(rho::Array{Float64},x::Array{Float64},
    nx::Int64,nobs::Int64)
    correlation = Matrix(1.0I,nobs,nobs)     #allocate array to store correlation
    @inbounds for i in 1:nobs                #loop over number of obs
        @inbounds for j in 1:nobs            #loop over number of obs
            #                                send to kernel function
            correlation[i,j] = correlation_kernel(x[i,:],x[j,:],rho,nx) 
        end
    end
    epsilon = Matrix(sqrt(eps(Float64))*I,nobs,nobs) # add machine precision for 
    #                                                  stability
    correlation .= correlation .+ epsilon 
    return correlation
end