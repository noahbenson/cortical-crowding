"""Code related to cortical magnification for the corticalcrowding library.
"""

# Dependencies #################################################################

import numpy as np

import neuropythy as ns


# Horton & Hoyt, 1991 ##########################################################

def hh91(eccen, c1=17.3, c2=0.75):
    """Calculates the areal cortical magniifcation according to Horton and Hoyt
    (1991).
    """
    eccen = np.asarray(eccen)
    return (c1 / (c2 + eccen))**2


