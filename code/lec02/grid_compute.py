import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

NUM_PTS = 20

p_grid = np.linspace(start=0, stop=1, num=NUM_PTS)

# The prior is a uniform distribution Uniform(0, 1)
prior = [1] * NUM_PTS

# The likelihood is Binomial(n, p)
likelihood = binom.pmf(k=6, n=9, p=p_grid)

# Compute the un-normalized posterior
unnormal_posterior = likelihood * prior

# Normalized posterior
posterior = unnormal_posterior / sum(unnormal_posterior)
print(f'Sum of posterior values = {sum(posterior)}')

plt.plot(p_grid, posterior, 'o-')
plt.xlabel('Probability of water')
plt.ylabel('Posterior probability')
plt.title(f'Sample= WLWWWLWLW; number of grid points = {NUM_PTS}')
plt.show()
