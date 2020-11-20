import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm, binom, uniform, beta

sns.set_style('whitegrid')
NUM_PTS = 2000

p_grid = np.linspace(start=0, stop=1, num=NUM_PTS)
p = np.zeros(NUM_PTS)

p[0] = 0.5
W = 6
L = 3

for i in range(1, NUM_PTS):
    p_new = norm.rvs(loc=p[i - 1], scale=0.1, size=1)

    if p_new < 0:
        p_new = abs(p_new)
    if p_new > 1:
        p_new = 2 - p_new

    q0 = binom.pmf(k=W, n=W + L, p=p[i - 1])
    q1 = binom.pmf(k=W, n=W + L, p=p_new)
    
    p[i] = np.where(uniform.rvs(loc=0, scale=1, size=1) < q1 / q0, p_new, p[i - 1])

analytical_posterior = beta.pdf(p_grid, W + 1, L + 1)

sns.kdeplot(p, label='MCMC')
plt.plot(p_grid, analytical_posterior, label='Analytical Posterior')

plt.xlabel('Fraction of water')
plt.ylabel('Posterior Probability')
plt.title(f'Number of samples in MCMC = {NUM_PTS}')
plt.legend()

plt.show()
