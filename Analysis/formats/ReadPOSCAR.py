import numpy as np
from numpy.linalg import norm
import re
from pathlib import Path
from .ReadXYZL import XYZL
from .ReadXYZL import vec_to_param#, XYZL
from .ReadLAMMPS import LAMMPS
from ..utils.cConv import cc_to_fc, fc_to_cc


class POSCAR:
    def __init__(self, path=None,):
        self.path = path
        self._memmap=False
        self._encoding = 'utf-8'
        if path is None:
            self.path = 'POSCAR'   # under current file
        if not Path(self.path).exists():
            raise FileNotFoundError('no POSCAR found')
        self.file_suffix = 'POSCAR'
        self.FrameNum = 1
        self.f = open(self.path, 'r')
        self._read_info()
        self._get_one_frame()
        self.f.close()

    def _read_info(self):
        #while (lines := self.f.readline()):
        self._offsets = list()
        self.name = self.f.readline()
        self.poscar_scaling_factor = float(self.f.readline())
        self.lat_vec = np.zeros((3,3))
        self.Gmat = np.zeros((3,3))
        for i in range(3):
            self.lat_vec[i] = np.array(self.f.readline().split(),dtype=np.float64)
        self.lat_param = vec_to_param(self.lat_vec)
        for i in range(3):
            for j in range(3):
                self.Gmat[i][j] = np.dot(self.lat_vec[i], self.lat_vec[j])
        
        lt1 = self.f.readline().split()
        lt2 = self.f.readline().split()
        assert len(lt1) == len(lt2), "wrong in POSCAR Nr. atom and atom type info"
        self.atom_lt = [] 
        for i in range(len(lt1)):
            self.atom_lt+=([lt1[i]] * int(lt2[i]))
        self.AtomNum = sum([int(num) for num in lt2])
        # first `Cartesian` or `Direct` label 
        # generation of `Cartesian` or `Direct` flag
        self.cartesian=True if self.f.readline().strip().lower() == 'cartesian' else False
        self._offsets.append(self.f.tell())
        self.LinesPerFrame = self.AtomNum + 1

    def _get_one_frame(self):
        c = list()
        self.f.seek(self._offsets[-1])
        while True:
            line = self.f.readline()
            if (re.match(r'\s+$', line) is not None or
                line == ''):
                break
            if re.search(r'[a-zA-Z]', line):
                c.append(np.array(line.split()[:-1], dtype=np.float64))
            else:
                c.append(np.array(line.split(), dtype=np.float64))
        if self.cartesian:
            self.c_coords = np.array(c)
            self.f_coords = cc_to_fc(self.c_coords,lat_vec=self.lat_vec)
        else:
            self.f_coords = np.array(c)
            self.c_coords = fc_to_cc(self.f_coords,self.lat_vec)
        self.coords = self.c_coords


    def get_neighbours(self, atom_symbol, threshold, aver=True):
        """cos` POSCAR is single-frame, we directly return a single-frame XYZL object."""
        xyzl=XYZL(from_stream=True, 
                  AtomNum=self.AtomNum, 
                  FrameNum=1, 
                  coords=self.coords, 
                  atom_lt=self.atom_lt,
                  lat_vec=self.lat_vec) 
        xyzl.get_neighbours(atom_symbol, threshold, aver)
            
            #trajl.get_neighbours(atom_symbol, threshold, aver)

    def write_to_xyz(self, idx_lt, filename, order=True, threshold=None):
        """write to .xyz file and group them based on molecule."""
        if order:
            assert threshold is not None, "when you want ordered xyz table, you must set threshold."
        xyzl=XYZL(from_stream=True,
                  AtomNum=self.AtomNum,
                  FrameNum=1,
                  coords=self.coords,
                  atom_lt=self.atom_lt,
                  lat_vec=self.lat_vec)
        if order:
            xyzl.set_mol(threshold)
            new_lt = build_new_order(xyzl.AtomNum, xyzl.mol_lt)
            for mol in xyzl.mol_lt:
                xyzl.coords[mol] = xyzl._get_mol_unwrap(mol)

            xyzl.coords = xyzl.coords[new_lt]
            xyzl.atom_lt = np.array(xyzl.atom_lt)[new_lt]
        xyzl.coords = xyzl.coords.reshape((1,*xyzl.coords.shape ))
        xyzl.write_to_xyz(idx_lt, filename)

    def to_lammps(self, low=None):
        # convert to lammps object
        # based on lammps manual first  we calculate the tilt factor by lattice parameters
        a,b,c,alpha,beta,gamma=vec_to_param(self.lat_vec)
        xy = b*np.cos(np.deg2rad(gamma))
        xz = c*np.cos(np.deg2rad(beta))
        
        denom=(b**2-xy**2)**.5
        yz = (b*c*np.cos(np.deg2rad(alpha))-xy*xz) / denom
        
        lx=a
        ly=denom
        lz=(c**2-xz**2-yz**2)**.5


        xlo, ylo, zlo = 0., 0., 0.
        if low is not None:
            xlo, ylo, zlo = low

        xhi=xlo+lx
        yhi=ylo+ly
        zhi=zlo+lz
        return LAMMPS(AtomNum=self.AtomNum, coords=self.coords, type_lt=self.type_lt, atom_lt=self.atom_lt, df=None, restricted_lat=[xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz])


def build_new_order(AtomNum, MolLt):
    MolLt1d = [ele for lt in MolLt for ele in lt]

    rest=np.setdiff1d(np.arange(AtomNum), MolLt1d)
    new_order=list(rest)+list(MolLt1d)
    return new_order
