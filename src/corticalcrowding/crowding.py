"""Code related to crowding distance for the corticalcrowding library.
"""

import numpy as np
import scipy.optimize
from scipy.optimize import curve_fit
from scipy.stats import gmean

def Kurzawski2023_cd(r, phi0, c, b):
    return (phi0 + r + c * r**2) * b

def log_Kurzawski2023_cd(r, phi0, c, b):
    return np.log10(Kurzawski2023_cd(r, phi0, c, b))

def fit_scale_logspace(inv_cmag, crowding):
    """
    Find scalar s that minimizes squared error in log space:
    log(crowding) ≈ log(s * inv_cmag)
    """
    log_ratio = np.log10(crowding / inv_cmag)
    s = 10 ** np.mean(log_ratio)
    return s

def bootstrap_fit(sids, xdata, ydata, x, num_bootstrap_samples):
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
                log_Kurzawski2023_cd,
                x_mean,
                np.log10(y_mean),
                p0=[0.43, 0.06, 0.15]
            )
        except RuntimeError:
            continue

        phi0, c, b = popt

        # store fitted curve 
        y_fit = Kurzawski2023_cd(x, phi0, c, b)

        boot_params.append(popt)
        boot_curves.append(y_fit)

    return np.array(boot_curves), np.array(boot_params)


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
