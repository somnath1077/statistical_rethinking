import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

NUM_PTS = 1000
# The sample is WLWWWLWLW. We use 1's for W and 0's for L
SAMPLE = [1, 0, 1, 1, 1, 0, 1, 0, 1]

p_grid = np.linspace(start=0, stop=1, num=NUM_PTS)

# The initial prior is a uniform distribution Uniform(0, 1)
prior = [1] * NUM_PTS

count = 1

for s in SAMPLE:
    # what is the likelihood given the next sample point
    likelihood = binom.pmf(k=s, n=1, p=p_grid)
    # Compute the un-normalized posterior
    unnormalized_posterior = likelihood * prior
    # Normalized posterior becomes the new prior for the next iteration
    posterior = unnormalized_posterior / sum(unnormalized_posterior)

    plt.plot(p_grid, posterior, label=f'{count}')
    prior = posterior
    count += 1

plt.xlabel('Probability of water')
plt.ylabel('Posterior probability')
plt.legend()
plt.title(f'number of grid points = {NUM_PTS}')
plt.show()
