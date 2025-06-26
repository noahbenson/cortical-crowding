"""The corticalcrowding library is a set of tools written for the analysis of
crowding distances in the visual field and cortical magnification.
"""

from .cmag import load_subject
from .cmag import HH91, HH91_integral, HH91_gain
from .cmag import HH91_fit, HH91_fit_cumarea
from .cmag import ring_area_deg2
from .cmag import ring_cmag
from .cmag import bilateral_areal_cmag
from .cmag import calculate_cortical_magnification
from .regression import fit_and_evaluate
from .crowding import Kurzawski2023_cd


