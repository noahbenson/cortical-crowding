"""The corticalcrowding library is a set of tools written for the analysis of
crowding distances in the visual field and cortical magnification.
"""

from .cmag import (
    load_subject,
    HH91, HH91_integral, HH91_gain, HH91_fit, HH91_fit_cumarea,
    invsuplin, invsuplin_integral, invsuplin_fit, invsuplin_fit_cumarea,
    cmag_basics,
    fit_cumarea,
    signed_bounds_from_abs_ranking,
    ring_area_deg2,
    ring_cmag)

from .crowding import (
    Kurzawski2023_cd, log_Kurzawski2023_cd,
    bootstrap_fit, bootstrap_cv)

from . import config
