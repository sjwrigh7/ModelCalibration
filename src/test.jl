function alloc_test(mat::Array{Float64},val::Float64)
    return val .* mat
end

function alloc_test!(mat::Array{Float64},val::Float64,ret::Array{Float64})
    ret .= mat .* val
end

function alloc_test2!(mat::Array{Float64},val::Float64,ret::Array{Float64})
    ret = mat .* val
end

temp = similar(c_sse[:,1,1])
pass = c_sse[:,1,1]

@btime alloc_test(c_sse[:,1,1],10.0)
@code_warntype alloc_test(c_sse[:,1,1],10.0)

@btime alloc_test!(c_sse[:,1,1],10.0,temp[:,1,1])
@btime alloc_test!(pass,10.0,temp)

@btime alloc_test2!(c_sse[:,1,1],10.0,temp[:,1,1])

function func_alloc_test(mat::Array{Float64},val::Float64)
    res = similar(mat[:,1,1:10])
    for i in 1:10
        res[:,i] .= alloc_test(mat[:,i,1],val)
    end
end

function func_alloc_test!(mat::Array{Float64},val::Float64)
    res = similar(mat[:,1,1:10])
    for i in 1:10
        alloc_test!(mat[:,i,1],val,res[:,i])
    end
end

function func_alloc_test2!(mat::Array{Float64},val::Float64)
    res = similar(mat[:,1,1:10])
    temp = similar(mat[:,1,1])
    for i in 1:10
        alloc_test!(mat[:,i,1],val,temp)
        res[:,i] .= temp
    end
end

function func_alloc_test3!(mat::Array{Float64},val::Float64)
    res = similar(mat[:,1,1:10])
    temp = similar(mat[:,1,1])
    pass = similar(temp)
    for i in 1:10
        pass .= res[:,i]
        alloc_test!(pass,val,temp)
        res[:,i] .= temp
    end
end

@btime func_alloc_test(c_sse,10.0)
metrics, events = LIKWID.@perfmon "FLOPS_DP" func_alloc_test(c_sse,10.0)

@btime func_alloc_test!(c_sse,10.0)

@btime func_alloc_test2!(c_sse,10.0)

@btime func_alloc_test2!(c_sse,10.0)

function loglik_test0(mat::Array{Float64},val::Float64)
    for i in 1:10
        vals = loglik_theta(mat[:,i,1],val)
    end
end

function loglik_test(mat::Array{Float64},val::Float64)
    vals = similar(mat[:,1,1])
    for i in 1:10
        loglik_theta!(mat[:,i,1],val,vals)
    end
end

@btime loglik_test0(c_sse,10.0)

@btime loglik_test(c_sse,10.0)

########################################
#for no sampling or calcs, only allocating matrices
@btime griddy_gibbs!(data,sampling_vars,c_sse,log_det_sig,sig_design,priors)
# 3.915 ms (149066 allocations: 10.22 MiB)

#for calculating theta likelihood
@btime griddy_gibbs!(data,sampling_vars,c_sse,log_det_sig,sig_design,priors)
# 7.100 s (299551 allocations: 43.84 GiB)

#for calculating theta likelihood while using a temporary pass variable to store vector from c_sse
@btime griddy_gibbs!(data,sampling_vars,c_sse,log_det_sig,sig_design,priors)
# 9.207 s (299735 allocations: 43.84 GiB)

#for calculating theta likelihood and sampling theta
@btime griddy_gibbs!(data,sampling_vars,c_sse,log_det_sig,sig_design,priors)

############################################
function griddy_sample(posterior_vals::Vector{Float64})
    #posterior_vals .= exp.(posterior_vals)
    #println(posterior_vals)
    stability_const = maximum(posterior_vals)
    #println(stability_const)
    stable_vals = exp.(posterior_vals .+ stability_const)
    #println(stable_vals)
    norm_vals = stable_vals ./sum(stable_vals)
    #println(stable_vals)
    cumsum_vals = cumsum(norm_vals)
    #println(cumsum_vals)
    uchance = rand(Uniform(0,1))
    #println("minimum = ",minimum(cumsum_vals),", max = ",maximum(cumsum_vals))
    #println("sample = $uchance")
    sample_idx = findfirst(x->x==true,cumsum_vals .> uchance)
    return sample_idx
end

function griddy_sample2(vals::Vector{Float64})
    stable_vals = exp.(vals .+ maximum(vals))
    cumsum_vals = cumsum(stable_vals ./ sum(stable_vals))
    uchance = rand(Uniform(0,1))
    sample_idx = findfirst(x->x==true,cumsum_vals .> uchance)
    return sample_idx
end

function griddy_sample2!(vals::Vector{Float64},cumsum_vals::Vector{Float64})
    cumsum_vals .= collect(0:1/99:1)
    uchance = rand(Uniform(0,1))
    sample_idx = findfirst(x->x==true,cumsum_vals .> uchance)
    return sample_idx
end

function sample_test(mat::Array{Float64})
    stable = similar(mat[:,1,1])
    norm = similar(stable)
    cumul = similar(norm)
    samples = Array{Float64}(undef,100)
    for i in 1:100
        sample = griddy_sample!(mat[:,1,1],stable,norm,cumul)
        samples[i] = sample
    end
end

function sample_test2(mat::Array{Float64})
    samples = Array{Float64}(undef,100)
    for i in 1:100
        sample = griddy_sample(mat[:,1,1])
        samples[i] = sample
    end
end

function sample_test3(mat::Array{Float64})
    samples = Array{Float64}(undef,100)
    for i in 1:100
        sample = griddy_sample2(mat[:,1,1])
        samples[i] = sample
    end
end

function sample_test4(mat::Array{Float64})
    samples = Array{Float64}(undef,100)
    temp = collect(0:1/99:1)
    for i in 1:100
        sample = griddy_sample2!(mat[:,1,1],temp)
        samples[i] = sample
    end
end

@btime sample_test(c_sse)
# 14.189 ms (113 allocations: 21.69 MiB)
# 155.771 ms (1013 allocations: 184.55 MiB)

@btime sample_test2(c_sse)
# 13.690 ms (162 allocations: 36.05 MiB)
# 157.688 ms (1602 allocations: 360.50 MiB)
# 148.243 ms (1602 allocations: 360.50 MiB)

@btime sample_test3(c_sse)
# 259.572 ms (1302 allocations: 270.73 MiB)
# 320.598 ms (1602 allocations: 360.50 MiB)
# 152.513 ms (1602 allocations: 360.50 MiB)

@btime sample_test4(c_sse)
# 332.130 ms (1307 allocations: 272.52 MiB)
# 334.109 ms (1607 allocations: 362.29 MiB)
# 6.881 ms (804 allocations: 89.87 MiB)