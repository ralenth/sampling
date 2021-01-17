import math
import statsmodels.api as sm

from convergence import *


def autocorr_diagnostics(sampling_runs, variables, **kwargs):
    """Run diagnostics on autocorrelation of sampled runs.

    Firstly, just plot autocorrelation, so one can see, which lags correspond to low autocorrelation value.
    Secondly, compute autocorrelation again and check computationally, which lags are satisfying for each samples (so
    for each variable in all runs).

    Parameters
    ----------
    sampling_runs : dict of list
        Sampled data prepared by function reorder_data.
    variables : list of str
        Names of variables of interest.
    kwargs
        Aditionally, you can specify any parameter of plot_acf function.
    """
    autocorrs = []

    for variable in variables:
        n_runs = len(sampling_runs[variable])

        for run in range(n_runs):
            # plot autocorrelation for each variable and each run
            sm.graphics.tsa.plot_acf(sampling_runs[variable][run], lags=40, zero=False,
                                     title=f'Autocorrelation for variable {variable} in {run + 1} run', **kwargs)
            plt.show()

            # now just compute autocorrelations without plotting them
            autocorrs.append(sm.tsa.acf(sampling_runs[variable][run]))

    # now we define maximum allowed autocorrelation value, as suggested in Box et al., 1994
    k = 2 / math.sqrt(len(sampling_runs[variables[0]][0]))
    # and we search lags for which each correlation value is <= k and we choose first one
    lag = [i for i in range(len(autocorrs[0])) if all(abs(autocorrs[j][i]) <= k for j in range(len(autocorrs)))][0]
    print(f'First lag satisfying condition: {lag}')
    return lag
