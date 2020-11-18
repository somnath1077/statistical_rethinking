import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

NUM_PTS = 20
# The sample is WLWWWLWLW. We use 1's for W and 0's for L
SAMPLE = [1, 0, 1, 1, 1, 0, 1, 0, 1]

p_grid = np.linspace(start=0, stop=1, num=NUM_PTS)

# The initial prior is a uniform distribution Uniform(0, 1)
prior = [1] * NUM_PTS

plt.plot(p_grid, prior)

for l in SAMPLE:
    # what is the likelihood given the next sample point
    likelihood = binom.pmf(k=l, n=1, p=p_grid)
    # Compute the un-normalized posterior
    unnormal_posterior = likelihood * prior
    # Normalized posterior becomes the new prior for the next iteration
    posterior = unnormal_posterior / sum(unnormal_posterior)
    print(f'Sum of posterior = {sum(posterior)}')
    plt.plot(p_grid, posterior)
    prior = posterior


plt.xlabel('Probability of water')
plt.ylabel('Posterior probability')
plt.title(f'number of grid points = {NUM_PTS}')
plt.show()
