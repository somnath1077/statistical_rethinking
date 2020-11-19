# We first generate the posterior as in s02_grid_compute.py and then
# approximate a normal distribution to the data

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, norm, beta

NUM_PTS = 20

p_grid = np.linspace(start=0, stop=1, num=NUM_PTS)
# Uniform prior is a special case of the Beta distribution
prior = [1] * NUM_PTS

# The likelihood is Binomial(n, p). Note that the Beta-Binomial is a conjugate pair.
likelihood = binom.pmf(k=6, n=9, p=p_grid)

# Compute the un-normalized posterior
unnormalized_posterior = likelihood * prior

# Normalized posterior
posterior = unnormalized_posterior / sum(unnormalized_posterior)

# Normal approximation
# the mean of the posterior distribution is
mu = np.sum(p_grid * posterior)

# The variance = E[X**2] - (E[X])**2
exp_x_squared = np.sum(np.square(p_grid) * posterior)
std = np.sqrt(exp_x_squared - mu ** 2)

print(f'posterior mean = {mu}, posterior standard deviation = {std}')
norm_approx_posterior = norm.pdf(p_grid, loc=mu, scale=std)

# The Beta dist. is a conjugate pair of the binomial dist
# More specifically, if X_1, ..., X_n are iid random variables from a Binomial dist.
# with parameter p, and p ~ Beta(a, b), then the posterior distribution of p
# given X_1 = x_1, ..., X_n = x_n is Beta(a + sum(x_1, ..., x_n), b + n - sum(x_1, ..., x_n))
# Since Uniform(0, 1) = Beta(1, 1), the parameter update rule after observing water W times
# and land L times is a = W + 1 and b = L + 1
W = 6
L = 3
beta_data = beta.pdf(p_grid, W + 1, L + 1)
beta_mu = beta.mean(W + 1, L + 1)
beta_std = beta.std(W + 1, L + 1)

norm_approx = norm.pdf(p_grid, beta_mu, beta_std)
# Plot both the analytically obtained posterior and the normal approximation
plt.plot(p_grid, beta_data, 'bo-', label='beta')
plt.plot(p_grid, norm_approx, 'ro-', label='normal')

plt.xlabel('Fraction of water')
plt.ylabel('Beta(W=6, L=3)')
plt.title(f'Sample= WLWWWLWLW; number of grid points = {NUM_PTS}')
plt.legend()

plt.show()
