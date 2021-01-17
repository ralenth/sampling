from pomegranate import *
import matplotlib.pyplot as plt


class RainBayesianNetwork:
    """Implementation of a simple Bayesian network predicting whether is raining based on environment."""

    def __init__(self):
        self._initialize_network()

    def _initialize_network(self):
        """Create a network structure."""

        cloudy = DiscreteDistribution({'T': 0.5, 'F': 0.5})

        sprinkler = ConditionalProbabilityTable(
            [['T', 'T', 0.1],  # probability that sprinkler is on given it's cloudy is 0.1
             ['T', 'F', 0.9],  # probability that sprinkler is off given it's cloudy is 0.9
             ['F', 'T', 0.5],
             ['F', 'F', 0.5]], [cloudy]
        )

        rain = ConditionalProbabilityTable(
            [['T', 'T', 0.8],
             ['T', 'F', 0.2],
             ['F', 'T', 0.2],
             ['F', 'F', 0.8]], [cloudy]
        )

        wet_grass = ConditionalProbabilityTable(
            [['T', 'T', 'T', 0.99],
             ['T', 'T', 'F', 0.01],
             ['T', 'F', 'T', 0.9],
             ['T', 'F', 'F', 0.1],
             ['F', 'T', 'T', 0.9],
             ['F', 'T', 'F', 0.1],
             ['F', 'F', 'T', 0.01],
             ['F', 'F', 'F', 0.99]], [sprinkler, rain]
        )

        node1 = State(cloudy, name='cloudy')
        node2 = State(sprinkler, name='sprinkler')
        node3 = State(rain, name='rain')
        node4 = State(wet_grass, name='wet_grass')

        # building the Bayesian Network
        network = BayesianNetwork("Predicting whether is raining based on environment")
        network.add_states(node1, node2, node3, node4)
        network.add_edge(node1, node2)
        network.add_edge(node1, node3)
        network.add_edge(node3, node4)
        network.add_edge(node2, node4)
        network.bake()

        self.network = network

    def plot(self):
        """Plot network structure."""

        self.network.plot()
        plt.show()
