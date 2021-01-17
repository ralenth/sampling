from .autocorrelation import *
from .convergence import *
from .gibbs_sampler import *
from .rain_bayesian_network import *


# define variables of interest (probability of which you want to sample)
variables = ['cloudy', 'rain']

# sample for two independent runs, 50000 times each run
chains = []
for _ in range(2):
    sampler = GibbsSampler()
    network = RainBayesianNetwork()
    samples = sampler.sample(network=network.network,
                             variables=variables,
                             evidence={'sprinkler': 'T', 'wet_grass': 'T'},
                             steps=50000)
    chains.append(samples)

# plot frequency of variable cloudy=True during first run
freq_plot(variables_dict={'cloudy': 'T'}, sampling_history=chains[0])

# prepare sampled data for computing sample diagnostics for both variables
chains = reorder_data(sampling_runs=chains, variables=variables)

# compute Gelman-Rubin shrink factors and their medians for both variables
bin_width = 50
shrink_factors, medians = gelman_plot_data(sampling_runs=chains, variables=variables, bin_width=bin_width)
# plot computed shrink factors for first variable
gelman_plot(shrink_factors=shrink_factors, medians=medians, variable=variables[0], bin_width=bin_width)
# since previous plot is fuzzy, zoom in on 100 first shrink factors
gelman_plot(shrink_factors=shrink_factors, medians=medians, variable=variables[0], bin_width=bin_width, zoom=100)

# perform autocorrelation diagnostics for all runs and all random variables
autocorr_diagnostics(sampling_runs=chains, variables=variables)
