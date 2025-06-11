# cConv -- coordinates conversion
import numpy as np
from numpy.linalg import inv

def cc_to_fc(coords, lat_vec=None, ilat_vec=None):
    if ilat_vec is None:
        ilat_vec = inv(lat_vec)
    return np.matmul(coords, ilat_vec)

def fc_to_cc(coords, lat_vec=None):
    return np.matmul(coords, lat_vec)

