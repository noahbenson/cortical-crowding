"""Code related to crowding distance for the corticalcrowding library.
"""

import numpy as np

def func_cd(x, b):
    # fitting function for crowding distance
    return np.log10((0.43 + x + 0.06*(x**2)) * b)

