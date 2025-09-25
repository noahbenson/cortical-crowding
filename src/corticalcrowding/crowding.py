"""Code related to crowding distance for the corticalcrowding library.
"""

import numpy as np
import scipy.optimize
from scipy.optimize import curve_fit

def Kurzawski2023_cd(x, b=0.15):
    """Estimate the crowding distance on cortex for the eccentricity `x`.
    
    This function uses the formula for crowding distance by Kurzawski et
    al. (2023) J. Vis.

    The parameter ``b`` is a gain factor for the curve. The default value is
    0.15, as described in the paper.
    """
    # fitting function for crowding distance
    return (0.43 + x + 0.06*(x**2)) * b

# We wrap the Kurzawski2023_cd with a log10 to fit using log errors.
def log_Kurzawski2023_cd(x, b):
    return np.log10(Kurzawski2023_cd(x, b))

def bootstrap_fit(sids, xdata, ydata, x, num_bootstrap_samples):
    # extracts the list of unique subjects
    unique_sids = np.unique(sids)
    bootstrapped_parameters = []
    # run the bootstrap how many times
    for _ in range(num_bootstrap_samples):
        indices = np.random.choice(unique_sids, size=len(unique_sids), replace=True)
        # for each sampled subject ID, find the list of row indices in sids where that subject appears.
        indices = [np.where(sids == sid)[0] for sid in indices]
        # flatten it
        indices = np.concatenate(indices)
        x_boot = xdata[indices]
        y_boot = ydata[indices]
        # Fit the curve to the bootstrapped sample
        b, _ = curve_fit(log_Kurzawski2023_cd, x_boot, np.log10(y_boot), p0=0.15)
        y = (0.43 + x + 0.06*(x**2)) * b
        bootstrapped_parameters.append(y) 
    return bootstrapped_parameters