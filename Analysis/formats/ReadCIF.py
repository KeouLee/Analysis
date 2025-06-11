import numpy as np
import copy
import re
#from .base import FormatBase
from .ReadXYZL import XYZL

"""currently for P1 space group only"""
pat_len = re.compile(r"_cell_length_([a-c])\s+([\d\.]+)")
pat_angle = re.compile(r"_cell_angle_([a-z]+)\s+([\d\.]+)")
pat_eecc = re.compile(r"([A-Z][a-z]*)\s+\1\s+[-\d\.]+[-\d\.]+[-\d\.]+[-\d\.]+") # pattern element  element  coordinates  charge

class CIF:
    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        self.f = open(self.fn, 'r')
        lat_param,frac_coords,charges,atom_lt = [], [], [], []
        for line in self.f:
            if pat_len.match(line):
                length = pat_len.match(line).groups(0)[-1]
                lat_param.append(length) # a b c alpha beta gamma
            elif pat_angle.match(line):
                angle = pat_angle.match(line).groups(0)[-1]
                lat_param.append(angle)
            elif pat_eecc.match(line):
                lt = line.split()
                atom_lt.append(lt[0])
                frac_coords.append([lt[2],lt[3],lt[4]])
                charges.append(lt[5])
        self.lat_param=np.array(lat_param,dtype=np.float64)
        self.atom_lt = atom_lt
        self.frac_coords = np.array(frac_coords,dtype=np.float64)

        for i in range(len(self.frac_coords)):
            if self.frac_coords[i][2] < .1:
                self.frac_coords[i][2] += 1.

        self.charges = np.array(charges, dtype=np.float64)
        self.AtomNum = len(self.atom_lt)
        self._get_cart_coords()
        return self

    def __exit__(self, exe_type, exe_value, traceback):
        self.f.close()

    def _get_cart_coords(self):
        xyzl=XYZL(lat_param=self.lat_param,from_stream=True,AtomNum=self.AtomNum, FrameNum=1, coords=self.frac_coords,atom_lt=self.atom_lt,Cartesian=False)
        self.coords = copy.deepcopy(xyzl.coords)
        self.lat_vec = copy.deepcopy(xyzl.lat_vec)

    def write_mol2(self):
        """write mol2 file based on"""
