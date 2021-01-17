from sampling.inference.rain_bayesian_network import *

import arviz as az
from collections import defaultdict
from itertools import accumulate
import matplotlib as mpl


def freq_plot(variables_dict, sampling_history):
    """Plot changes of given variable values during one sampling run.

    Parameters
    ----------
    variables_dict : dict
        Keys are variable names, values - their values of interest, for example True or 'T'.
    sampling_history : list of dict
        Sampling history returned when sampling using GibbsSampler class.
    """
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['deepskyblue', 'rebeccapurple', 'gold', 'magenta'])
    ax = plt.subplot()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for variable, value in variables_dict.items():
        freqs = [int(state[variable] == value) for state in sampling_history]
        # we accumulate computed values, so we can plot evolution of their frequency
        freqs = list(accumulate(freqs))

        for i in range(len(freqs)):
            # compute mean frequency
            freqs[i] /= (i + 1)

        ax.plot(freqs, label=f'{variable}={value}')

    plt.legend(fontsize='medium')
    plt.title('Change in variable frequencies during sampling')
    plt.show()


def reorder_data(sampling_runs, variables):
    """Prepare data for computing Gelman-Rubin shrink factor.
    Parameters
    ----------
    sampling_runs : list
        List of sampling history returned by GibbsSampler.
    variables : list of str
        Names of variables, same as ones defined before running the sampler.
    """
    data = defaultdict(list)
    for sampling_run in sampling_runs:
        for variable in variables:
            # we need numeric data to compute shrink factor
            sampled_values = [1 if sampling_run[i][variable] == 'T' else 0 for i in range(len(sampling_run))]
            data[variable].append(sampled_values)

    return data


def gelman_plot_data(sampling_runs, variables, bin_width):
    """Prepare sampled data for plotting evolution of Gelman-Rubin shrink factor.

    Parameters
    ----------
    sampling_runs : dict of list
        Sampled data prepared by function reorder_data.
    variables : list of str
        Names of variables of interest.
    bin_width : int
        Number of observations per segment.
    """
    shrink_factors = defaultdict(list)
    medians = defaultdict(list)

    for variable in variables:
        sampled_runs = sampling_runs[variable]
        length = len(sampled_runs[0])
        for end in range(bin_width, length + 1, bin_width):
            sampled_data = [run[:end] for run in sampled_runs]
            # we create special arviz object to compute shrink factor using rhat function
            sampled_data = az.convert_to_dataset({variable: sampled_data})
            shrink_factors[variable].append(float(az.rhat(sampled_data)[variable]))
            medians[variable].append(np.median(shrink_factors[variable]))

    return shrink_factors, medians


def gelman_plot(shrink_factors, medians, variable, bin_width, zoom=None):
    """Plot exact and median values of Gelman-Rubin shrink factor during sampling.
    Parameters
    ----------
    shrink_factors : dict of list
        Keys are variable names, values - list of computed shrink factor values.
    medians : dict of list
        Keys are variable names, values - list of computed shrink factor medians.
    variable : str
    bin_width : int
        Number of observations per segment.
    zoom : int
        Number of shrink factors to consider. If your sampling run took quite long, you may want to zoom results to
        take a closer look at the beginning of the simulation. If None, all shrink factors will be shown.
    """
    if zoom is None:
        zoom = len(medians[variable])

    ax = plt.subplot()
    xs = [i * bin_width for i in range(1, len(medians[variable][:zoom]) + 1)]
    ax.plot(xs, medians[variable][:zoom], label='median', c='rebeccapurple')
    ax.plot(xs, shrink_factors[variable][:zoom], '--', label='exact', c='deepskyblue', alpha=0.7)

    plt.title(f'Change of Gelman-Rubin shrink factor for variable {variable}')
    plt.xlabel('sample')
    plt.ylabel('shrink factor')
    plt.legend()
    plt.show()
