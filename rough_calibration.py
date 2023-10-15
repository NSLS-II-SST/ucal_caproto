import numpy as np
from scipy.signal import find_peaks
import pickle

cal_line_master = {'ck': 278.21, 'nk': 392.25, 'tila': 452, 'ok': 524.45, 'fell': 614.84,
                   'coll': 675.98, 'fk': 677, 'fela': 705.01, 'felb': 717.45,
                   'cola': 775.31, 'colb': 790.21, 'nill': 742.3, 'nila': 848.85,
                   'nilb': 866.11, 'cula': 926.98, 'culb': 947.52,
                   'znla': 1009.39, 'znlb': 1032.46}

def get_line_energies(line_names):
    """
    Takes a list of strings or floats, and returns the line energies in
    cal_line_master.
    """
    line_energies = [cal_line_master.get(n, n) for n in line_names]
    return line_energies

def simple_calibration(data, npeaks):
    c, x = np.histogram(data, bins=100)
    peak_loc, info = find_peaks(c, height=10)
    peak_heights = info['peak_heights']
    if len(peak_heights) < npeaks:
        return None
    else:
        peak_idx = np.argsort(peak_heights)[-npeaks:]
        if np.any(peak_heights[peak_idx] < 0):
            return None
        return np.sort(x[peak_loc[peak_idx]])

def make_cal_dict(data, npeaks):
    cal_dict = {}
    for channum, values in data.items():
        peaks = simple_calibration(values, npeaks)
        if peaks is None:
            continue
        else:
            cal_dict[channum] = peaks
    return cal_dict

def make_poly_dict(data):
    poly_dict = {}
    for k, v in cal_dict.items():
        if v is None:
            continue
        x = np.insert(v, 0, 0.0)
        y = np.array([0, 278.2, 392, 524])
        poly_dict[k] = np.poly1d(np.polyfit(x, y, 2))

    
