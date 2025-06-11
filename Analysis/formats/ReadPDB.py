import re
import copy
import numpy as np
from numpy.linalg import norm
import sys
#from numpy.linalg import inv
#from .ReadXYZL import param_to_vec

np.set_printoptions(threshold=sys.maxsize)

class PDB:
    cryst_pat=r'CRYST1\s+([\.\d]+\s+)([\.\d]+\s+)([\.\d]+\s+)([\.\d]+\s+)([\.\d]+\s+)([\.\d]+\s+)'
    coord_pat=r'(ATOM|HETATM)\s+\d+\s+[A-Z][a-z]*\s+[A-Za-z]+\s+[A-Za-z]+\s+\d+\s+([-\.\d]+)\s+([-\.\d]+)\s+([-\.\d]+)\s+[-\.\d]+\s+[-\.\d]+\s+[A-Z][a-z]*'
    def __init__(self, pdb, mo, mol2, size, convenient=False):
        """
        arguments
        -----------------
        pdb :: file name of supercell pdb file
        mo :: file name of output mol2 file
        mol2 :: template mol2 object based on unit cell
        size :: supercell size (ix, iy, iz)
        convenient :: check the correspondence or not 
        """
        f=open(pdb,'r') # open newly created spcl pdb file
        coords = []
        for line in f:
            m = re.match(self.cryst_pat, line)
            if m:
                self.lat_param=np.array(m.groups(),dtype=np.float64) # self.lat_param -- supercell
            m = re.match(self.coord_pat, line)
            if m:
                coords.append(m.groups()[1:])
                #extract the coordinates
        self.coords = np.array(coords, dtype=np.float64)
        f.close()

        if convenient:
            self._write_spcl_mol2(mol2, size, mo)
            return 

        self._search(mol2, size)
        self._write_spcl_mol2(mol2, size, mo)

    def _write_spcl_mol2(self,mol2,size, mo):
        f=open(mo,'w')
        self._construct_bond_list(mol2, size)
        f.close()

    def _construct_bond_list(self,mol2,size):
        #mol2.bond -> (AtomIdx AtomIdx BondType)
        lt = []
        multiple=int(size[0])*int(size[1])*int(size[2])
        for i in range(multiple):
            for bond in mol2.bond:
                base=mol2.AtomNum*i
                lt.append([base+int(bond[0]), base+int(bond[1]), bond[2]])
        for i, bond in enumerate(lt):
            print(f'{i}  {bond[0]}  {bond[1]}  {bond[2]}')  

    def _search(self, mol2, size):
        # c is the cartesian coordinates of passed in atom
        # we try to wrap back into the unit cell
        
        # first we inspect the mol2 coords to make sure 0<=coord<=1 
        # and store the range-modified fractional coordinates of the 
        # unit cell to `mol2_fc` variable
        mol2_fc = np.empty(mol2.frac_coords.shape)
        for i in range(len(mol2.frac_coords)):
            tmp = mol2.frac_coords[i] - np.array(mol2.frac_coords[i], dtype=np.int32)
            for j in range(len(tmp)):
                if tmp[j] < 0:
                    tmp[j] += 1
            mol2_fc[i] = copy.deepcopy(tmp)

        idx=[]
        #abc = np.empty(self.coords.shape)
        # now compare each coordinates in the supercell to `mol2_fc` we just constructed            
        for i in range(len(self.coords)):
            fc1=np.matmul(self.coords[i],mol2.ilat_vec)
            tmp = fc1 - np.array(fc1, dtype=np.int32)
            for j in range(len(tmp)):
                if tmp[j] < 0:
                    tmp[j] += 1
            #abc[i] = copy.deepcopy(tmp)

            q=0
            for k in range(len(mol2_fc)):
                for l in range(3):
                    if abs(mol2_fc[k][l] - tmp[l]) > .999:
                        tmp[l] = mol2_fc[k][l]
                d=norm(mol2_fc[k] - tmp)
                if q > 1:
                    raise Exception
                if d<1e-3:
                    q+=1
                    idx.append(k)
            if q == 0:
                #print(i)
                #print(self.coords[i])
                #print(tmp)
                #print(fc1)
                #print(d)
                raise Exception
        self._chk_idx(idx, size, mol2.AtomNum)

    def _chk_idx(self, idx, size, AtomNum):
        #AtomNum == Nr. atom per unit cell
        multiple=int(size[0])*int(size[1])*int(size[2])
        assert (len(idx)%multiple==0)
        arr = np.array(idx).reshape((multiple,AtomNum))
        for sa in arr:
            self._chk_idx_low(sa)

    def _chk_idx_low(self, idx):
        for i in range(len(idx)):
            assert (i == idx[i])
