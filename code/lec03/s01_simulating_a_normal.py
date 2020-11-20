import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.stats import binom

sns.set_style('whitegrid')

NUM_PEOPLE = 1000
NUM_TOSSES = 20

dist_from_center_line = np.zeros(NUM_PEOPLE)

for i in range(NUM_TOSSES):
    tosses = binom.rvs(n=1, p=0.5, size=NUM_PEOPLE)
    # convert 0s to -1
    steps_to_move = np.where(tosses == 0, -1, 1)
    dist_from_center_line += steps_to_move

sns.kdeplot(dist_from_center_line)
plt.show()
