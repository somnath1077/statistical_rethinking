# The Howell data set for Kalahari foragers consists of the
# height, weight, age and sex of 544 individuals.
# We are modeling the height as a normally distributed variable with mean mu and
# sd sigma, where mu ~ N(178, 20) and sigma ~ Uniform(0, 50).

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform

sns.set_style('whitegrid')
NUM_SAMPLES = 10000

sample_mu = norm.rvs(loc=178, scale=20, size=NUM_SAMPLES)
sample_sigma = uniform.rvs(loc=0, scale=50, size=NUM_SAMPLES)

# prior heights
prior_h = norm.rvs(sample_mu, sample_sigma, NUM_SAMPLES)

sns.kdeplot(prior_h)
plt.show()
