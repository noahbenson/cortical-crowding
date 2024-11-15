"""Code related to cortical magnification for the corticalcrowding library.
"""

# Dependencies #################################################################

import numpy as np
import neuropythy as ny
from scipy.spatial import KDTree
from scipy.optimize import minimize
from .config import proc_path

# Load subjects ###############################################################
def load_subject(sid, proc_path = proc_path, rater = 'Linda'):
    freesurfer_path = proc_path/'freesurfer'/sid
    sub = ny.freesurfer_subject(str(freesurfer_path))
    roi_path = proc_path/'rois'
    prf_path = proc_path/'prfvista'/sid/'ses-nyu3t01'
    prf_data = {}
    for h in ('lh','rh'):
        hem = sub.hemis[h]
        tmp = {}
        tmp['polar_angle'] = ny.load(str(prf_path/f'{h}.angle_adj.mgz'))
        tmp['eccentricity'] = ny.load(str(prf_path/f'{h}.eccen.mgz'))
        tmp['radius'] = ny.load(str(prf_path/f'{h}.sigma.mgz'))
        tmp['variance_explained'] = ny.load(str(prf_path/f'{h}.vexpl.mgz'))
        tmp['visual_area'] = ny.load(str(roi_path/f'{rater}_{h}.{sid}.ROIs_V1-4.mgz'))
        prf_data[h] = hem.with_prop(tmp)
    sub = sub.with_hemi(prf_data) 
    return sub   

# Horton & Hoyt, 1991 ##########################################################

def HH91(x, a=17.3, b=0.75, c=2):
    """Calculates the areal cortical magniifcation according to Horton and Hoyt
    (1991).
    """
    x = np.asarray(x, dtype=np.float64)
    return (a / (b + x))**c

def HH91_for_fit(x, a=17.3, b_sqrt=np.sqrt(0.75), c=2):
    return HH91(x, a, b_sqrt**2, c)

def fit_cmag(ecc, cmag, p0=[17.3, 0.75], method=None):
    ecc = ecc.astype(np.float64)
    cmag = cmag.astype(np.float64)
    def func(params):
        pred = HH91_for_fit(ecc, *params)
        error = (cmag/pred - 1)**2 + (pred/cmag - 1)**2
        return np.sum(error)
    p0 = list(p0)
    p0[1] = np.sqrt(p0[1])
    result = minimize(func, p0, method=method)
    params = list(result.x)
    params[1] = params[1]**2
    return params

# Calculate cortical magnification ##########################################################

def ring_area_deg2(min_eccen, max_eccen, hemifield=False):
    """Computes the area (in square degrees) of a ring in the visual field."""
    if hemifield:
        return (np.pi * max_eccen**2 - np.pi * min_eccen**2) / 2
    else:
        return np.pi * max_eccen**2 - np.pi * min_eccen**2

def ring_cmag(sub, eccen = None, hemi=('lh','rh'), ring_size=None, retinotopy='any', mask=None):
    sub_ecc = []
    sub_sarea = []
    if isinstance(hemi, str):
        hemi = [hemi]
    if not isinstance(mask, dict) or not np.isin(mask.keys(), ['lh', 'rh']).all():
        mask = {h: mask for h in hemi}
    for h in hemi:
        hem = sub.hemis[h]
        # ret : a dict
        ret = ny.retinotopy_data(hem, retinotopy)
        m = mask[h]
        if m is not None:
            m = hem.mask(m)
        else:
            m = slice(0,None)
        sub_ecc.append(ret['eccentricity'][m])
        sub_sarea.append(hem.prop('midgray_surface_area')[m])
    sub_ecc = np.concatenate(sub_ecc)
    sub_sarea = np.concatenate(sub_sarea)
    if eccen is None:
        eccen = np.sort(sub_ecc)
    # ring size: the fraction of points in the visual area that should be included in the window.
    if ring_size is None:
        ring_size = int(len(sub_ecc) * 0.2)
    # if the given ring_size is a percentage
    elif isinstance(ring_size, float):
        ring_size = int(len(sub_ecc) * ring_size)
    else:
        ring_size = int(ring_size)
    
    shash = KDTree(sub_ecc[:,None]) 
    (d, ii) = shash.query(np.reshape(eccen, (-1,1)), ring_size)
    
    near_ecc = sub_ecc[ii]
    near_sarea = sub_sarea[ii]
    
    min_eccen = np.min(near_ecc, axis=1)
    max_eccen = np.max(near_ecc, axis=1)
    
    total_surface_area = np.sum(near_sarea, axis=1)
    area_vis = ring_area_deg2(min_eccen, max_eccen, hemifield=(len(hemi)==1))
    
    return (eccen,total_surface_area / area_vis)


def bilateral_areal_cmag(sub, retinotopy='any', mask=None, hemi=('lh','rh'),
                         surface_area=None, nnearest=None):
    """Similar to neuropythy's areal_cmag function but operates bilaterally.
    
    See also `neuropythy.vision.areal_cmag`.
    """
    from collections.abc import Mapping
    from neuropythy.vision import retinotopy_data, as_retinotopy
    from neuropythy.vision.cmag import ArealCorticalMagnification
    # The mask and surface area arguments are tricky, so we parse it ahead of time:
    if isinstance(mask, list):
        mask = {h:m for (h,m) in zip(hemi, mask)}
    elif not isinstance(mask, Mapping):
        mask = {h: mask for h in hemi}
    if isinstance(surface_area, list):
        surface_area = {h:m for (h,m) in zip(hemi, surface_area)}
    elif not isinstance(surface_area, Mapping):
        surface_area = {h:surface_area for h in hemi}
    # First, find the retino data
    (angs,eccs,sars) = ([],[],[])
    for h in hemi:
        hem = sub.hemis[h]
        retino = retinotopy_data(hem, retinotopy)
        # Convert from polar angle/eccen to longitude/latitude
        (ang,ecc) = as_retinotopy(retino, 'visual')
        # get the surface area
        s = surface_area[h]
        if s is None:
            s = 'midgray_surface_area'
        if isinstance(s, str):
            s = hem.prop(s)
        # Get the indices we care about
        m = mask[h]
        if m is None:
            ii = hem.indices
        else:
            ii = hem.mask(m, indices=True)
        ii = ii[np.isfinite(ang[ii]) & np.isfinite(ecc[ii])]
        # Append all our data:
        angs.append(ang[ii])
        eccs.append(ecc[ii])
        sars.append(s[ii])
    angs = np.concatenate(angs)
    eccs = np.concatenate(eccs)
    sars = np.concatenate(sars)
    # get our nnearest
    if nnearest is None:
        nnearest = int(np.ceil(np.sqrt(len(angs)) * np.log(len(angs))))
    return ArealCorticalMagnification(
        angs, eccs, sars,
        nnearest=nnearest,
        weight=None)

# calculate cortical magnification based on x,y eccentricity given in a df
def calculate_cortical_magnification(df):
    cortical_magnifications = {1: [], 2: [], 3: [], 4: []}
    for index, row in df.iterrows():
        observer_id = row['ID']
        try:
            sub = load_subject(observer_id)
            for mask_value in [1, 2, 3, 4]:
                cm = bilateral_areal_cmag(sub, mask=('visual_area', mask_value))
                cm_value = cm(row['Eccen_X'], row['Eccen_Y'])
                cortical_magnifications[mask_value].append(cm_value)
        except Exception as e:
            for mask_value in [1, 2, 3, 4]:
                cortical_magnifications[mask_value].append(np.nan)  # Assign NaN if an exception occurs
    return cortical_magnifications