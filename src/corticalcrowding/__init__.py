"""The corticalcrowding library is a set of tools written for the analysis of
crowding distances in the visual field and cortical magnification.
"""

from .cmag import load_subject
from .cmag import HH91
from .cmag import HH91_for_fit
from .cmag import fit_cmag
from .cmag import ring_area_deg2
from .cmag import ring_cmag
from .cmag import bilateral_areal_cmag
from .cmag import calculate_cortical_magnification
from .regression import fit_and_evaluate
from .crowding import func_cd


