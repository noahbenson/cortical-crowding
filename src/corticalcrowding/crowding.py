"""Code related to crowding distance for the corticalcrowding library.
"""

import numpy as np

def Kurzawski2023_cd(x, b=0.15):
    """Estimate the crowding distance on cortex for the eccentricity `x`.
    
    This function uses the formula for crowding distance by Kurzawski et
    al. (2023) J. Vis.

    The parameter ``b`` is a gain factor for the curve. The default value is
    0.15, as described in the paper.
    """
    # fitting function for crowding distance
    return (0.43 + x + 0.06*(x**2)) * b

