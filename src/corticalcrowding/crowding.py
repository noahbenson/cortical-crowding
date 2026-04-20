"""Code related to crowding distance for the corticalcrowding library.
"""

import numpy as np
import scipy.optimize
from scipy.optimize import curve_fit
from scipy.stats import gmean

def Kurzawski2023_cd_3param(r, phi0, c, b):
    return (phi0 + r + c * r**2) * b

def log_Kurzawski2023_cd_3param(r, phi0, c, b):
    return np.log10(Kurzawski2023_cd_3param(r, phi0, c, b))

def fit_scale_logspace(inv_cmag, crowding):
    """
    Find scalar s that minimizes squared error in log space:
    log(crowding) ≈ log(s * inv_cmag)
    """
    log_ratio = np.log10(crowding / inv_cmag)
    s = 10 ** np.mean(log_ratio)
    return s

def bootstrap_fit_subject_avg(sids, xdata, ydata, x, num_bootstrap_samples):
    sids  = np.asarray(sids)
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    unique_sids = np.unique(sids)
    unique_ecc  = np.unique(xdata)

    boot_curves = []
    boot_params = []

    for _ in range(num_bootstrap_samples):

        # resample subjects with replacement
        sampled_sids = np.random.choice(
            unique_sids,
            size=len(unique_sids),
            replace=True
        )

        # compute subject-averaged data per eccentricity
        x_mean = []
        y_mean = []

        for ecc in unique_ecc:
            vals = []

            for sid in sampled_sids:
                mask = (sids == sid) & (xdata == ecc)
                if np.any(mask):
                    vals.append(gmean(ydata[mask]))

            if len(vals) > 0:
                x_mean.append(ecc)
                y_mean.append(gmean(vals))

        
        x_mean = np.asarray(x_mean)
        y_mean = np.asarray(y_mean)

        if len(y_mean) < 3:   # need >=3 points for 3 params
            continue

        # fit model to averaged data
        try:
            popt, _ = curve_fit(
                log_Kurzawski2023_cd_3param,
                x_mean,
                np.log10(y_mean),
                p0=[0.43, 0.06, 0.15]
            )
        except RuntimeError:
            continue

        phi0, c, b = popt

        # store fitted curve 
        y_fit = Kurzawski2023_cd_3param(x, phi0, c, b)

        boot_params.append(popt)
        boot_curves.append(y_fit)

    return np.array(boot_curves), np.array(boot_params)




######### functions not used  ##################
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