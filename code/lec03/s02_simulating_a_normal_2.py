import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.stats import norm, uniform

sns.set_style('whitegrid')

REPETITIONS = 10000
NUM_SAMPLES = 12

growth_rates = [np.prod(uniform.rvs(loc=1.0, scale=0.1, size=NUM_SAMPLES))
                for _ in range(REPETITIONS)]

big = [np.prod(uniform.rvs(loc=1.0, scale=0.5, size=NUM_SAMPLES))
       for _ in range(REPETITIONS)]

small = [np.prod(uniform.rvs(loc=1.0, scale=0.01, size=NUM_SAMPLES))
         for _ in range(REPETITIONS)]


def dist_plot(vals, norm_compare=False):
    sns.distplot(vals, label='Distribution plot')
    if norm_compare:
        sample_mu = np.mean(vals)
        sample_sd = np.std(vals)

        min_val = np.min(vals)
        max_val = np.max(vals)

        d_range = np.linspace(min_val, max_val, num=1000)
        normal_data = norm.pdf(x=d_range, loc=sample_mu, scale=sample_sd)
        plt.plot(d_range, normal_data, linestyle='--', color='black', label='Normal approximation')
    plt.legend()
    plt.show()


dist_plot(growth_rates, norm_compare=True)
dist_plot(big, norm_compare=True)
dist_plot(small, norm_compare=True)
dist_plot(np.log(big), norm_compare=True)
