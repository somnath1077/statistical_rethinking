import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.stats import norm, uniform

sns.set_style('whitegrid')

NUM_PEOPLE = 1000
NUM_TOSSES = 200

dist_from_center_line = [sum(uniform.rvs(loc=-1, scale=2, size=NUM_TOSSES)) for _ in range(NUM_PEOPLE)]

sample_mu = np.mean(dist_from_center_line)
sample_sd = np.std(dist_from_center_line)

min_dist = np.min(dist_from_center_line)
max_dist = np.max(dist_from_center_line)

d_range = np.linspace(min_dist, max_dist, NUM_PEOPLE)
normal_data = norm.pdf(x=d_range, loc=sample_mu, scale=sample_sd)

sns.kdeplot(dist_from_center_line)

plt.xlabel('Distance from center line')
plt.ylabel('Number of people')
plt.title(f'Number of people = {NUM_PEOPLE}, number of tosses = {NUM_TOSSES}')

plt.axvline(x=sample_mu, linestyle='--', color='b', label='mean')
plt.axvline(x=sample_mu + sample_sd, linestyle='--', color='r', label='mean + sd')
plt.axvline(x=sample_mu - sample_sd, linestyle='--', color='g', label='mean - sd')

plt.plot(d_range, normal_data, linestyle='--', color='black')
plt.legend()
plt.show()
