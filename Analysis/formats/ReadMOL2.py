import numpy as np
import copy
from numpy.linalg import inv
import re
from .ReadXYZL import param_to_vec, vec_to_param
from ..utils.cat import cat_to_mol

class MOL2:
    pdb_coord_pat=r'(ATOM|HETATM)\s+\d+\s+[A-Z][a-z]*\s+[A-Za-z]+\s+[A-Za-z]+\s+\d+\s+([-\.\d]+)\s+([-\.\d]+)\s+([-\.\d]+)\s+[-\.\d]+\s+[-\.\d]+\s+([A-Z][a-z]*)'
    cryst_pat=r'CRYST1\s+([\.\d]+\s+)([\.\d]+\s+)([\.\d]+\s+)([\.\d]+\s+)([\.\d]+\s+)([\.\d]+\s+)'
    mol2_atom_pat=r'@<TRIPOS>ATOM'
    mol2_num_pat=r'\s*(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+'
    mol2_bond_pat=r'@<TRIPOS>BOND'
    mol2_substrate_pat=r'@<TRIPOS>SUBSTRUCTURE'
    def __init__(self, pdb, mi, mo, size, boundary_idx1, boundary_idx2, boundary_idx3):
        """
        arguments
        ------------------
        pdb: .pdb to make supercell (MUST contain CRYST1 card) and atom_lt. Its atom order must accord with input .mol2
        mi: input mol2 file to define residue name, element symbol, coords, atom type* and charge*.
        mo: later output supercell mol2 file
        size: supercell size to make [2 2 2]
        boundary_idx*: [[23,32,'cp'], [5, 77,'qq']] the atoms at boundaries of three direction
        """
        f1,f2=open(pdb,'r'),open(mi,'r') 
        self.atom_lt = []
        for line in f1:
            m=re.match(self.cryst_pat, line)
            if m:
                self.lat_param=np.array(m.groups(),dtype=np.float64)
                continue
            m=re.match(self.pdb_coord_pat,line)
            if m:
                self.atom_lt.append(m.groups()[-1])

        f1.close()
        #for line in f2:
        #    if (m:=re.match(self.mol2_num_pat, line)):
        #        AtomNum, BondNum = m.groups()
        #        self.AtomNum, self.BondNum = int(AtomNum), int(BondNum)
        #    elif (m:=re.match(self.mol_atom_pat,line)):
        #        break  # next line is coords etc.
        
        i = 0
        for line in f2:
            if i == 0:
                m = re.match(self.mol2_num_pat, line)
                if m:
                    AtomNum, BondNum = m.groups()
                    self.AtomNum, self.BondNum = int(AtomNum), int(BondNum)
                    i+=1
            elif i == 1:
                m = re.match(self.mol2_atom_pat, line)
                if m:
                    #read next `self.AtomNum` line
                    break
        j=0
        self.atom_type_lt, self.charges = np.empty(self.AtomNum, dtype='U2'),  np.empty(self.AtomNum)
        self.coords=np.empty((self.AtomNum,3))
        while j < self.AtomNum:
            line = f2.readline().split()
            self.atom_type_lt[j], self.charges[j] = \
            line[5].strip(),      float(line[-1])
            self.coords[j] = line[2:5]
            if j == 0:
                self.residue = line[-2].strip()
            j+=1

        self.bond=[]
        for line in f2:
            m = re.match(self.mol2_bond_pat,line)
            if m:
                break
        j=0
        while j < self.BondNum:
            line = f2.readline().split()
            self.bond.append((int(line[1])-1, int(line[2])-1, line[3])) # !!!
            j+=1

        for line in f2:
            m = re.match(self.mol2_substrate_pat, line)
            if m:
                self.substrate_template=f2.readline()
                break
        f2.close()

        self.lat_vec = param_to_vec(self.lat_param)
        self.ilat_vec = inv(self.lat_vec)
        self.frac_coords=np.matmul(self.coords,self.ilat_vec)

        self.boundary = copy.deepcopy([boundary_idx1, boundary_idx2, boundary_idx3])
        self.size=[]
        for i in range(len(size)):
            self.size.append(int(size[i]))
        self.multiple = self.size[0] * self.size[1] * self.size[2]
        
        spcl_fcZ, spcl_chgs, spcl_atom_lt, spcl_atom_type_lt = self._make_supercell()
        spcl_bond_list=self._make_bond_list(spcl_fcZ)

        connectivity=[]
        for b in spcl_bond_list:
            connectivity.append(b[:2])
        mols = cat_to_mol(connectivity)
        
        spcl_fcZ, spcl_chgs, spcl_bond_list, spcl_atom_lt, spcl_atom_type_lt = self._order(spcl_fcZ, spcl_chgs, spcl_bond_list, spcl_atom_lt, spcl_atom_type_lt, mols) # use this to make things ordered

        self._write_to_pdb(spcl_fcZ, spcl_atom_lt, True) # write _test.pdb for visualization check
        self._write_mol2(spcl_fcZ, spcl_chgs, spcl_bond_list, spcl_atom_lt, spcl_atom_type_lt, mo, True)

    def _order(self, spcl_fcZ, spcl_chgs, spcl_bond_list, spcl_atom_lt, spcl_atom_type_lt, mols):
        shift1,shift2=self.AtomNum*self.size[0], self.AtomNum*self.size[0]*self.size[1]
        spcl_fcZ_1d = np.empty((self.AtomNum*self.multiple,3))

        for i in range(self.size[2]):
            for j in range(self.size[1]):
                for k in range(self.size[0]):
                    for l in range(self.AtomNum):
                        #print(l+k*self.AtomNum+j*shift1+i*shift2)
                        spcl_fcZ_1d[l+k*self.AtomNum+j*shift1+i*shift2]=spcl_fcZ[k,j,i,l]
        mols_1d=[]
        for mol in mols:
            for m in mol:
                mols_1d.append(m)
        spcl_fcZO_1d=spcl_fcZ_1d[mols_1d]
        spcl_chgsO=spcl_chgs[mols_1d]
        spcl_atom_ltO=spcl_atom_lt[mols_1d]
        spcl_atom_type_ltO=spcl_atom_type_lt[mols_1d]
        
        spcl_bond_listO=[]
        for bond in spcl_bond_list:
            t=(mols_1d.index(bond[0]), mols_1d.index(bond[1]), bond[2])
            spcl_bond_listO.append(t)
        return spcl_fcZO_1d, spcl_chgsO, spcl_bond_listO, spcl_atom_ltO, spcl_atom_type_ltO

    def _write_mol2(self, spcl_fc, spcl_chgs, spcl_bl, spcl_al, spcl_atl, mo, new=True):
        """
        spcl_fc [X-axis, Y-axis, Z-axis, AtomNum, 3]
        spcl_bl bond list built from making supercell first along x, then y, at last z-axis
        """
        f=open(mo, 'w')
        f.write('@<TRIPOS>MOLECULE\n')
        f.write(f'{self.residue}\n')
        f.write(f'  {self.AtomNum*self.multiple}    {len(spcl_bl)}      1     0     0\n')
        f.write('SMALL\n')
        f.write('DDEC\n\n\n')
        f.write('@<TRIPOS>ATOM\n')

        if new:
            for i in range(len(spcl_chgs)):
                c=np.round(np.matmul(spcl_fc[i], self.lat_vec),4)
                print('{:>7} {:>2s}{:>17}{:>11}{:>11} {:<2s}          1 {:s}{:>16}'.format(i+1,spcl_al[i],c[0],c[1],c[2],spcl_atl[i],self.residue,spcl_chgs[i]),file=f)

        else:
            shift1=self.size[0]*self.AtomNum
            shift2=shift1*self.size[1]
            for i in range(self.size[2]):
                for j in range(self.size[1]):
                    for k in range(self.size[0]):
                        coords=np.round(np.matmul(spcl_fc[k,j,i], self.lat_vec),4)
                        for l in range(len(coords)):
                            idx = l + k*self.AtomNum + j*shift1 + i*shift2 
                            print('{:>7} {:>2s}{:>17}{:>11}{:>11} {:<2s}          1 {:s}{:>16}'.format(idx+1,self.atom_lt[l],coords[l][0],coords[l][1],coords[l][2],self.atom_type_lt[l],self.residue,self.charges[l]),file=f)

            spcl_bl = sorted(spcl_bl)
        f.write('@<TRIPOS>BOND\n')
        for i, bond in enumerate(spcl_bl,start=1):
            print('{:>6}{:>5}{:>6} {:s}'.format(i, bond[0]+1, bond[1]+1, bond[2]), file=f)

        f.write('@<TRIPOS>SUBSTRUCTURE\n')
        f.write('     1 LIG         1 TEMP              0 ****  ****    0 ROOT')


        f.close()

    def _make_bond_list(self, spcl_fcZ):
        _boundary = copy.deepcopy(self.boundary)
        order=[]
        for i, boundary_alpha in enumerate(self.boundary):
            order.append([])
            for j, bond in enumerate(boundary_alpha):
                if self.coords[bond[1]][i] > self.coords[bond[0]][i]: 
                    order[-1].append(True)
                else:
                    self.boundary[i][j][0], self.boundary[i][j][1] = self.boundary[i][j][1], self.boundary[i][j][0]
                    order[-1].append(False)
        # exchange if it is not ordered
        if not _boundary == self.boundary:
            print('boundary atom sequence changed!')
            print(f'new self.boundary {self.boundary}')
            print(f'old boundary {_boundary}')
        for boundary_alpha in self.boundary:
            for bbond in boundary_alpha:
                for i in range(len(self.bond)):
                    if bbond != self.bond[i]:
                        tmp = (bbond[1], bbond[0], bbond[2])
                        if tmp == self.bond[i]:
                            self.bond[i] = copy.deepcopy(tuple(bbond))

        if self.size[0] != 1:
            ltx = [copy.deepcopy(self.bond)]
            for i in range(1, self.size[0]):
                ltx.append([])
                for b in self.bond:
                    if not _check_in(self.boundary[0], b):  # check if it is the boundary atoms
                        ltx[-1].append((b[0]+i*self.AtomNum, b[1]+i*self.AtomNum,b[2]))
                    else:
                        #if self.coords[b[1]][0] > self.coords[b[0]][0]:
                        ltx[-1].append((b[1] +self.AtomNum*(i-1), b[0]+self.AtomNum * i, b[2]))
                        #else: 
                        #ltx[-1].append((b[0] +self.AtomNum*(i-1), b[1]+self.AtomNum * i, b[2]))
            new_ltx=[]
            for cell in ltx:
                for bond in cell:
                    if not _check_in(self.boundary[0], bond):
                        new_ltx.append(bond)

            tmp = []
            for bbond in self.boundary[0]:  # loop over boundary bonds in x-direction
                #if self.coords[bbond[1]][0] > self.coords[bbond[0]][0]:
                new_ltx.append((bbond[0], bbond[1]+(self.size[0]-1)*self.AtomNum, bbond[2]))
                tmp.append((bbond[0], bbond[1]+(self.size[0]-1)*self.AtomNum, bbond[2]))
                #else: 
                #    new_ltx.append((bbond[1], bbond[0]+(self.size[0]-1)*self.AtomNum, bbond[2]))
                #    tmp.append((bbond[1], bbond[0]+(self.size[0]-1)*self.AtomNum, bbond[2]))

            # update new boundary
            for i in range(len(tmp)):
                self.boundary[0][i] = tmp[i]

            tmp = []
            if self.boundary[1]:
                for i in range(self.size[0]):
                    for ybbond in self.boundary[1]:
                        tmp.append((ybbond[0]+i*self.AtomNum, ybbond[1]+i*self.AtomNum, ybbond[2]))
                self.boundary[1] = copy.deepcopy(tmp)

            if self.boundary[2]:
                for i in range(self.size[0]):
                    for zbbond in self.boundary[2]:
                        tmp.append((zbbond[0]+i*self.AtomNum, zbbond[1]+i*self.AtomNum, zbbond[2]))
                self.boundary[2] = copy.deepcopy(tmp)
        else:
            new_ltx = copy.deepcopy(self.bond)

        #y
        if self.size[1] != 1:
            lty = [copy.deepcopy(new_ltx)]
            shift=self.AtomNum*self.size[0]
            # now make supercell along y-direction
            # doing this will alter the boundary connectivity thus you must re-generate it
            for i in range(1, self.size[1]):    
                lty.append([])
                for bond in new_ltx:
                    if not _check_in(self.boundary[1], bond): #?
                        lty[-1].append((bond[0]+shift*i, bond[1]+shift*i, bond[2]))
                    else:
                        #if (bond[1] +shift*(i-1), bond[0]+ shift* i, bond[2]) == (76,481,'1'):
                        #    print('yes')
                        lty[-1].append((bond[1] +shift*(i-1), bond[0]+ shift* i, bond[2]))
            new_lty=[]
            for cell in lty:
                for bond in cell:
                    if not _check_in(self.boundary[1], bond):
                        new_lty.append(bond)

            tmp = []
            for bbond in self.boundary[1]:  # loop over boundary bonds in x-direction
                new_lty.append((bbond[0], bbond[1]+(self.size[1]-1)*shift, bbond[2]))
                tmp.append((bbond[0], bbond[1]+(self.size[1]-1)*shift, bbond[2]))

            # update self.boundary
            for i in range(len(tmp)):
                self.boundary[1][i] = tmp[i]

            tmp = []
            if self.boundary[0]:
                for i in range(self.size[1]):
                    for xbbond in self.boundary[0]:
                        tmp.append((xbbond[0]+i*shift, xbbond[1]+i*shift, xbbond[2]))
                self.boundary[0] = copy.deepcopy(tmp)

            if self.boundary[2]:
                for i in range(self.size[1]):
                    for zbbond in self.boundary[2]:
                        tmp.append((zbbond[0]+i*shift, zbbond[1]+i*shift, zbbond[2]))
                self.boundary[2] = copy.deepcopy(tmp)
        else:
            new_lty = copy.deepcopy(new_ltx)
         
        #z
        if self.size[2] != 1:
            ltz = [copy.deepcopy(new_lty)]
            shift=self.AtomNum*self.size[0]*self.size[1]
            # now make supercell along y-direction
            # doing this will alter the boundary connectivity thus you must re-generate it
            for i in range(1, self.size[2]):    
                ltz.append([])
                for bond in new_lty:
                    if not _check_in(self.boundary[2], bond):
                        ltz[-1].append((bond[0]+shift*i, bond[1]+shift*i, bond[2]))
                    else:
                        ltz[-1].append((bond[1] +shift*(i-1), bond[0]+ shift* i, bond[2]))
            new_ltz=[]
            for cell in ltz:
                for bond in cell:
                    if not _check_in(self.boundary[2], bond):
                        new_ltz.append(bond)

            tmp = []
            for bbond in self.boundary[2]:  # loop over boundary bonds in x-direction
                new_ltz.append((bbond[0], bbond[1]+(self.size[2]-1)*shift, bbond[2]))
                tmp.append((bbond[0], bbond[1]+(self.size[2]-1)*shift, bbond[2]))

            # update self.boundary
            for i in range(len(tmp)):
                self.boundary[2][i] = tmp[i]

            tmp = []
            if self.boundary[0]:
                for i in range(self.size[2]):
                    for xbbond in self.boundary[0]:
                        tmp.append((xbbond[0]+i*shift, xbbond[1]+i*shift, xbbond[2]))
                self.boundary[0] = copy.deepcopy(tmp)

            if self.boundary[1]:
                for i in range(self.size[2]):
                    for ybbond in self.boundary[1]:
                        tmp.append((ybbond[0]+i*shift, ybbond[1]+i*shift, ybbond[2]))
                self.boundary[1] = copy.deepcopy(tmp)
        else:
            new_ltz = copy.deepcopy(new_lty)

        #for i in range(len(new_ltz)):
        #    for j in range(len(new_ltz)):
        #        if i != j:
        #            if new_ltz[i] == new_ltz[j]:
        #                print(i,j)

        return new_ltz

        #print(self.boundary)
        #for a in new_ltz:
        #    print(a)
    
    def _make_supercell(self):
        spcl_fcX = np.empty((self.size[0],self.AtomNum, 3))
        # step by step, first x-direction
        for i in range(self.size[0]):
            for j in range(self.AtomNum):
                spcl_fcX[i,j]=self.frac_coords[j]+np.array([1.,0.,0.])*i

        spcl_fcY = np.empty((self.size[0], self.size[1],self.AtomNum, 3))
        # next y-direction 
        for i in range(self.size[0]):
            current_uc = spcl_fcX[i]
            for j in range(self.size[1]):
                for k in range(self.AtomNum):
                    spcl_fcY[i,j,k]=current_uc[k] + np.array([0.,1.,0.])*j

        spcl_fcZ = np.empty((self.size[0], self.size[1], self.size[2], self.AtomNum, 3))
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                current_uc = spcl_fcY[i,j]
                for k in range(self.size[2]):
                    for l in range(self.AtomNum):
                        spcl_fcZ[i,j,k,l]=current_uc[l] + np.array([0.,0.,1.])*k
        #print(np.matmul(spcl_fcZ[0,0,0],self.lat_vec))
        charges, atom_lt, atom_type_lt = self._generate(self.charges), self._generate(self.atom_lt), self._generate(self.atom_type_lt)
        return spcl_fcZ, charges, atom_lt, atom_type_lt

    def _generate(self, lt):
        ltx=[]
        for i in range(self.size[0]):
            for c in lt:
                ltx.append(c)
        lty=[]
        for i in range(self.size[1]):
            for c in ltx:
                lty.append(c)
        ltz=[]
        for i in range(self.size[2]):
            for c in lty:
                ltz.append(c)
        return copy.deepcopy(np.array(ltz))
    
    def _write_to_pdb(self, spcl_fcZ, spcl_atom_lt=None, new=True):
        f=open('_test.pdb', 'w')
        self.LAT_VEC = np.zeros((3,3))
        for i in range(3):
            self.LAT_VEC[i] = self.lat_vec[i] * self.size[i]
        self.LAT_PARAM = vec_to_param(self.LAT_VEC)
        print('CRYST1   {:.3f}   {:.3f}   {:.3f}  {:.2f}  {:.2f}  {:.2f}'.format(*self.LAT_PARAM), file=f)

        if new:
            for i in range(len(spcl_atom_lt)):
                cc=np.round(np.matmul(spcl_fcZ[i], self.lat_vec),3)
                print('HETATM{:>5} {:<4s} {:>3s} A   1    {:>8}{:>8}{:>8}  1.00  0.00{:>12}'.format(i+1, spcl_atom_lt[i], self.residue, cc[0], cc[1], cc[2], spcl_atom_lt[i]), file=f)

        else:
            shift1=self.size[0]*self.AtomNum
            shift2=shift1*self.size[1]
            for i in range(self.size[2]):
                for j in range(self.size[1]):
                    for k in range(self.size[0]):
                        for l in range(self.AtomNum):
                            cc=np.round(np.matmul(spcl_fcZ[k,j,i,l], self.lat_vec),3)
                            idx = l + k*self.AtomNum + j*shift1 + i*shift2 + 1
                            atom_name = self.atom_lt[l]
                            print('HETATM{:>5} {:<4s} {:>3s} A   1    {:>8}{:>8}{:>8}  1.00  0.00{:>12}'.format(idx, atom_name, self.residue, cc[0], cc[1], cc[2], atom_name), file=f)
        f.close()

def _check_in(lt, bond):
    for bbond in lt:
        if bond[0] in bbond:
            if bond[1] in bbond:
                return True
    return False
