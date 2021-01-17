import copy
import random

import numpy as np
from pomegranate import DiscreteDistribution


class GibbsSampler:
    """Implementation of Gibbs sampling algorithm."""

    def __init__(self):
        self.history = []

    def sample(self, network, variables, evidence, steps):
        """Perform sampling algorithm.

        Parameters
        ----------
        network : pomegranate BayesianNetwork
        variables : list of str
            List of names of variables that will be sampled during simulation.
        evidence : dict
            Dict of observed variables (aka evidence). Dict keys should be variable names, values - observed values.
        steps : int
        """

        self.history = []
        # randomly initialise variables
        state = {var: random.choice(['T', 'F']) for var in variables}

        for _ in range(steps):
            variable = random.choice(variables)
            constant = {k: value for k, value in state.items() if k != variable}

            belief = network.predict_proba({**evidence, **constant})
            # now belief consists of different instances, so we need to filter one we're interested in
            belief = next(filter(lambda x: isinstance(x, DiscreteDistribution), belief))

            probs = belief.parameters[0]
            state[variable] = np.random.choice(list(probs.keys()), p=list(probs.values()))

            self.history.append(copy.deepcopy(state))

        return self.history

    def estimate_marginal(self, variable, value):
        """Estimate marginal probability for a given variable.

        Warning
        -------
        Please keep in mind that this function doesn't allow you to specify observed variables. They are set in sample
        function, so in order to change them for estimating marginal, you need to perform sampling as well.
        """

        return sum(state[variable] == value for state in self.history) / len(self.history)
