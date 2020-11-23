import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

NUM_PTS = 20

p_grid = np.linspace(start=0, stop=1, num=NUM_PTS)

# The prior is a uniform distribution Uniform(0, 1)
prior_1 = [1] * NUM_PTS

# Suppose that we know that water covers at least 50% of the planet
prior_2 = np.where(p_grid >= 0.5, 1, 0)

# Third prior
prior_3 = np.exp(-5 * np.abs(p_grid - 0.5))

# set the actual prior
prior = prior_3

# The likelihood is Binomial(n, p)
likelihood = binom.pmf(k=6, n=9, p=p_grid)

# Compute the un-normalized posterior
unnormalized_posterior = likelihood * prior

# Normalized posterior
posterior = unnormalized_posterior / sum(unnormalized_posterior)
print(f'Sum of posterior values = {sum(posterior)}')

plt.plot(p_grid, posterior, 'o-')

plt.xlabel('Fraction of water')
plt.ylabel('Posterior probability')
plt.title(f'Sample= WLWWWLWLW; number of grid points = {NUM_PTS}')

plt.show()
