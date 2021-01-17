# sampling

Scripts for sampling diagnostics created for Statistical Data Analysis 2 course. Following scripts and functionalities 
included:
* `rain_bayesian_network.py` - example of Bayesian network, from which one can sample probabilities, using this
  repository,
* `gibbs_sampler.py` - Gibbs sampler used for sampling probabilities,
* `convergence.py` - tools for convergence diagnostics, among which one is mimicking R visualization function - 
  `gelman_plot` (from `coda` package),
* `autocorrelation.py` - autocorrelation diagnostics tool, mimicking R visualization function - `acf`,
* `usage_example.py`

<br/>

#### Rain Bayesian Network
Bayesian network implemented according to SDA2 project specification. Network has following structure:
![RainBayesianNetwork](../bn.png)