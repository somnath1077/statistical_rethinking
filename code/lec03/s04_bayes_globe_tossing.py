import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import binom, uniform

sns.set_style('whitegrid')

NUM_PROB_VALUES = 1000
W = 6
N = 9

p_grid = np.linspace(start=0, stop=1, num=NUM_PROB_VALUES)

posterior = binom.pmf(k=W, n=N, p=p_grid) * uniform.pdf(p_grid, loc=0, scale=1)
norm_posterior = posterior / sum(posterior)

plt.plot(p_grid, norm_posterior)
plt.show()
