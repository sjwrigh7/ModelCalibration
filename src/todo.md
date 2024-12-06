### Tasks
* finish base code
    * check for `length()` and `size()` calls for iterators and replace with `eachindex()` and `axes()`
    * look into adding functionality for different prior distributions
    * add `MCMC constructor` data struct to inform how to do the MCMC sampling for different settings
* look into implementing other MCMC samplers
    * slice sampling (ellliptical)
    * hamiltonian
    * exact samlping
    * no u turn sampling
    *look at existing code for possible other ideas (mamba.jl ?)
* work on parallelization of main MCMC sampling
    * possibly also parallelization of other parts of the code? step size algorithm and surrogate model training
* Once parallelization is finished, look into markov chain analysis metrics (mamba.jl ?) and other analysis criteria
* autocorrelation based thinning of posterior samples

Fragility analysis of misalignment of CFD plumes and soot witness plumes
* 