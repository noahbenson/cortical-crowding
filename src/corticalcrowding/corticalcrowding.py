"""Code related to cortical crowding distance for the corticalcrowding library.
"""

import numpy as np

# bootstrap on coefficient of variation of cortical crowding distance
def bootstrap_cv(data, num_samples=1000):
    bootstrapped_cv = []
    n = len(data)
    for _ in range(num_samples):
        sample_indices = np.random.choice(n, size=n, replace=True)
        bootstrapped_sample = data[sample_indices]
        mean_sample = np.mean(bootstrapped_sample)
        std_sample = np.std(bootstrapped_sample)
        cv_sample = std_sample / mean_sample
        bootstrapped_cv.append(cv_sample)
    return np.array(bootstrapped_cv)
