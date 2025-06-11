from .ReadXYZ import XYZ
from .rot_analysis import _detect
import sys
import itertools
from collections import defaultdict, deque
import numpy as np
from numpy.linalg import inv, norm
from ..utils.wrapper_pickle import dump_pkl, load_pkl
from ..utils.cat import cat_to_mol, triangle_build, get_com, uncat, exclude, find_node
from ..utils.UnitConv import wavenumber_to_THz, evtok
from ..utils import get_inertia, WriteToXyz, dist_map
from ..utils.rotor import rot_mat
from ..utils.toy_class import Plot, Cylinder
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.ndimage import gaussian_filter
from scipy import constants
from scipy.interpolate import UnivariateSpline
#from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import copy
import multiprocessing
from itertools import chain, permutations, combinations, accumulate
import math
#from tqdm import tqdm
from typing import Sequence
#import warnings
#from ..FortranLib import fortran_diagonalization as fd
#from .fortran_diagonalization import tred2, imtql2


__author__ = 'Ke Li'
__all__ = ['XYZL', 'Angle', 'Radius']
#np.set_printoptions(threshold=sys.maxsize, suppress=False)
class XYZL(XYZ):
    """
    with ana.XYZL(path=traj_path,  temperature=1200, run_type='unconstrained', 
                  system_name='sodium', dt=timestep, MolIdx=[(1,2,3),(5,6),], lat_vec=lat_vec) as trajl:
        trajl.get_angles(angle='theta')
        trajl.get_angles(angle='phi',)
        trajl.get_angle_density(angle='theta')
        trajl.get_angle_density(angle='theta')
        trajl.get_rtcf()
    
    """
    def __init__(self,
                 path=None,
                 temperature=None,
                 run_type=None,
                 system_name=None,
                 flag=None,
                 slicing=None,
                 from_stream=None,
                 AtomNum=None,
                 FrameNum=None,
                 coords=None,
                 atom_lt=None,
                 dt=None,
                 MolIdx=None,
                 memmap=False,
                 details=False,
                 encoding=None,
                 Cartesian=True,
                 fast=False,
                 n_processes=None,
                 start=None,
                 out=None,
                 lat_vec=None,
                 lat_param=None,
                 ):
        super(XYZL,self).__init__(path,
                                  temperature,
                                  run_type,
                                  system_name,
                                  flag,
                                  slicing,
                                  from_stream,
                                  AtomNum,
                                  FrameNum,
                                  coords,
                                  atom_lt,
                                  dt,
                                  MolIdx,
                                  memmap,
                                  details,
                                  encoding,
                                  Cartesian,
                                  fast,
                                  n_processes,
                                  start,
                                  out,
                                  )
        if slicing is None:
            self._setup_lat(lat_vec, lat_param)
        else:
            return 

        if not Cartesian:
            self.frac_coords = coords
            self.coords = self._get_cc(self.frac_coords)
        else:
            if not fast:
                self.frac_coords = np.matmul(self.coords, self.ilat_vec)

    def _setup_lat(self,lat_vec, lat_param):
        if (lat_vec is None and lat_param is None):
            if self._from_stream:
                return 
            raise TypeError("must supply one of lattice vector or lattice params!")
        elif lat_vec is None:
            lat_vec = param_to_vec(lat_param)
        elif lat_param is None:
            lat_param = vec_to_param(lat_vec)

        self.lat_vec = lat_vec
        self.ilat_vec = inv(lat_vec)
        self.lat_param = lat_param
        self.Gmat = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                self.Gmat[i][j] = np.dot(lat_vec[i], lat_vec[j])


    def get_rtcf(self, central_atom, ligand_atom, threshold, correlation_step, interval, dump_name):
        self._setup_rtcf(central_atom, ligand_atom, threshold)
        self.rtcf = np.zeros(correlation_step)
        self.normalized_factor = np.zeros(correlation_step)
        time_origin_point = self.FrameNum - correlation_step
        tmp=0

        print('calculating residence time correlation function...')
        #for i in range(0,self.FrameNum,interval):
        for i in range(0,time_origin_point,interval):
            if (tmp < i // (self.FrameNum//10)):
                tmp = i // (self.FrameNum//10)
                print(f'{tmp*10} %')
            self._setup_first(i)
            init_pair_lt = self._get_pair_lt()
            lt = []
            for j, f in enumerate(self):
                if j == correlation_step:
                    break
                count = 0
                current_pair_lt=self._get_pair_lt()
                for cr in current_pair_lt:
                    if cr in init_pair_lt:
                        count += 1
                lt.append(count)
            l = len(lt)
            if l == 0:
                continue
            self.rtcf[:l] += np.array(lt)
            self.normalized_factor[:l] += lt[0]
        self.rtcf = self.rtcf / self.normalized_factor
        print(f'dumpping rtcf to {dump_name}')
        dump_pkl(dump_name, self.rtcf)

    def _setup_first(self, num):
        start_char = self._offsets[num]
        self.f.seek(start_char)
        self._get_one_frame()
        self.f.seek(start_char)
        if self.flag:
            self.flag=False

    def _get_pair_lt(self):
        pair_lt = []
        for p in self.pair_lt:
            if (self._get_dist_unwrap(p[0],p[1]) < self.threshold):
                pair_lt.append(p)
        return pair_lt
    
    def _setup_rtcf(self, central_atom, ligand_atom, threshold):
        """simply group this two up."""
        c_idx, l_idx = self.idx_d[central_atom], self.idx_d[ligand_atom]
        self.pair_lt = [(c,l) for c in c_idx for l in l_idx]
        self.threshold = threshold

    def get_wacf(self, group_names, correlation_step, interval, dump_name, drift_correction=True, exclude_atom=None):
        """with ana.XYZL(lat_vec=lat_vec, dt=dt, path=path) as trajl:
            trajl.setup_group({'Mol1': Sequence(P_atom_idx, S1_atom_idx, S2_atom_idx, S3_atom_idx, S4_atom_idx)})
            trajl.get_wacf('Mol1')
            with ana.XYZL(lat_vec=lat_vec, dt=dt, path=path) as trajl:
                trajl.setup_group({'Mol1': Sequence(P_atom_idx, S1_atom_idx, S2_atom_idx, S3_atom_idx, S4_atom_idx)})
                trajl.setup_group({'Mol2': Sequence(P_atom_idx, S1_atom_idx, S2_atom_idx)})
                trajl.get_wacf(['Mol1','Mol2'],cs,i,dn,dc,exclude_atom=[0,5,10])
        """
        unwrap=True
        print(f'drift_correction={drift_correction}')
        print('calculating angular velocities...')
        for gn in group_names:
            assert gn in self.idx_d, f"no atom group {group_name} exists, cannot calculate its power spectrum."
        if self.dt is None:
            raise ValueError("timestep dt is not supplied, it must be supplied to calculate velocity from trajectory")
        MolNum = len(group_names)
        total_lt=list(chain(*[self.idx_d[gn] for gn in group_names]))
        group_atom_num = len(total_lt)
        print(f'MolNum={MolNum}')

        coords=np.zeros((self.FrameNum, self.AtomNum, 3))
        coords_group=np.zeros((self.FrameNum,group_atom_num,3))
        ligand_com_dist=np.zeros((self.FrameNum, group_atom_num))

        vel=np.zeros((self.FrameNum-2, group_atom_num,3))

        for i in range(self.FrameNum):
            coords[i]=copy.deepcopy(self[i].coords)
            u, d = 0, 0
            for gn in group_names:
                idx_lt=list(self.idx_d[gn]) 
                mass_lt=self.mass_lt[idx_lt]
                l=len(idx_lt)

                #unwrap
                coord=self._get_mol_unwrap(idx_lt,i)
                coords[i][idx_lt]=coord
                #print(coord)
                u=d
                d+=l
                #print('u',u)
                #print('d',d)
                #exit()
                com = get_com(coord,mass_lt)
                coord_com = coord-com
                coords_group[i][u:d]=coord_com
                #print('coord-com', (coord-com).shape)
        #print('coords_group', coords_group)
        #WriteToXyz(coords, self.atom_lt, 'TMP.xyz')
                #print('?')
        #exit()
        #print(coords.shape)
        #print(coords_group)
        #print(coords_group.shape)
        #exit() 

        # velocity section
        for i in range(self.FrameNum-2):
            if drift_correction:
                _com, com_ = get_com(coords[i], self.mass_lt), get_com(coords[i+2], self.mass_lt)
                central_drift=com_ - _com
                coords_group[i+2] -= central_drift
            vel[i] = (coords_group[i+2] - coords_group[i]) / self.dt / 2
        coords_group=coords_group[1:-1]

        distance = dist_map(coords_group)
        distance = distance.reshape((distance.shape[0], distance.shape[1], 1))
        anvel = np.cross(coords_group, vel) / distance

        if exclude_atom is not None:
            if isinstance(exclude_atom, int):
                anvel = np.delete(anvel, exclude_atom, axis=1)
            else:
                anvel = np.delete(anvel,list(exclude_atom),axis=1)
        
        #dump_pkl(dump_name, angular_vels)
        print('calculating wacf...')
        WACF = []
        FN = self.FrameNum - 2
        for cs in range(correlation_step):
            w0w0 = 0
            w0wt = 0
            for i in range(0,FN,interval):
                if (i+cs >= FN):
                    break
                anvelocities_t0 = anvel[i]  # ensemble velocities (SeletedAtomNum, 3)
                anvelocities_t = anvel[i+cs] 
                w0w0 += np.sum(anvelocities_t0 * anvelocities_t0)
                w0wt += np.sum(anvelocities_t0 * anvelocities_t)
            WACF.append(w0wt / w0w0)

        wacf = np.array(WACF)
        dump_pkl(dump_name, wacf)

    def get_wacf_inertiaF(self, group_name, correlation_step, interval, dump_name):
        ierr = -1
        idx_lt = list(self.idx_d[group_name])
        mass_lt = self.mass_lt[idx_lt]
    
        print('calculating total angular velocity...')
        # velocity array generation
        vel = np.array([v for v in self._get_vel(group_name)])
        for v in vel:
            vcom = get_com(v, mass_lt)
            v -= vcom
        #print('vel')
        #print(vel)
        #exit()
        # position array generation
        coords = np.zeros((self.FrameNum, len(idx_lt), 3))   # for starter we rule the central atom out.
        #Dist2 = np.zeros((self.FrameNum, len(idx_lt), 1))  
        for i, f in enumerate(self):
            # \vec(r_k) generation
            coord = self.coords[idx_lt]
            rcom = get_com(coord, mass_lt)
            coord = coord - rcom  # now we have coord under the Molecule Center of Mass.
            coords[i,:,:] = coord 
            #lc = coord[1:]
            ## r_k^2 generation
            #for j,c in enumerate(lc):
            #    Dist2[i,j,0]=np.dot(c,c) 

        # generation2
        coords = coords[1:-1]
        #print('coords')
        #print(coords)
        #exit()
        # >>> coords.shape == (self.FrameNum-2,MolAtomNum,3)
        # >>> True
        # >>> vel.shape == (self.FrameNum-2,MolAtomNum,3)
        # >>> True

        # >>> coords, mass_lt 
        # >>> iner=self._get_multiframe_inertia(coords, mass_lt)
        # >>> iner.shape == (self.FrameNum-2, 3,3)
        # >>> True
        print('calculating inertia tensor...')
        I_inertia_inverse = np.zeros((self.FrameNum-2, 3,3))
        EN = np.zeros((self.FrameNum-2, 3))
        Nxyz = 3 # working dimension
        for i in range(coords.shape[0]):
            sub_EN = np.zeros(3,order='F')
            sub_EDDI = np.zeros(3,order='F')
            # get every frame inertia for a cluster.
            coord = coords[i]
            I_inertia=get_inertia(coord, mass_lt)
            #print(coord)
            #print(I_inertia)
            #continue
            # call fortran
            tred2(nm=Nxyz, n=Nxyz, d=sub_EN, e=sub_EDDI, z=I_inertia)
            imtql2(nm=Nxyz, n=Nxyz, d=sub_EN, e=sub_EDDI, z=I_inertia, ierr=ierr)
            #print('sub_EN', sub_EN)
            #print('I_inertia', I_inertia)
            #continue
            # processing...
            #print('EN', i, sub_EN)
            #continue
            I_inertia_tps = np.transpose(I_inertia)
            EN_reprocical = np.diag(1./sub_EN)
            #print('EN_reprocical', i, EN_reprocical)
            #continue
            #print(I_inertia)
            #print(I_inertia_tps)
            #continue
            I_inertia_inverse[i] = np.matmul(np.matmul(I_inertia,EN_reprocical),I_inertia_tps)
            #print(np.matmul(np.matmul(I_inertia,EN_reprocical),I_inertia_tps))
        #print('I_inertia_tps', I_inertia_tps)
        #print('EN_reprocical', EN_reprocical)
        #print('I_inertia_inverse', I_inertia_inverse)
        #exit()

        #Dist2 = Dist2[1:-1]
        #vel = np.delete(vel,0,1) # generation only for the ligand atom.
        rxv=vel = np.cross(coords, vel) # dreamy numpy broadcast.

        #print(rxv)
        #exit()
        L_ang_momentum = np.zeros((self.FrameNum-2, 3))
        mass_ltt = mass_lt.reshape((len(mass_lt),1))
        #print(f'vel[0] {vel[0]}')
        #print(f'mass_ltt {mass_ltt}')
        #print(vel[0] * mass_ltt,)
        #print(np.sum(vel[0] * mass_ltt, axis=0))
        print('calculating total angular velocities...')
        for i in range(rxv.shape[0]):
            L_ang_momentum[i] = np.sum(rxv[i] * mass_ltt, axis=0)
        #print(L_ang_momentum)
        #exit()
        #vel = vel / Dist2

        angular_tot = np.zeros((self.FrameNum-2,3))
        for i in range(self.FrameNum - 2):
            # (3,) = ([self.FrameNum-2,] (3,3)  @ ([self.FrameNum-2,] (3))
            angular_tot[i] = I_inertia_inverse[i] @ L_ang_momentum[i]
        #print('angular_tot')
        #print(angular_tot)
        #exit()
        
        vel = angular_tot
        print('calculating wacf...')
        FN = vel.shape[0]
        WACF = []
        for cs in range(correlation_step):
            v0v0 = 0
            v0vt = 0
            for i in range(0,FN,interval):
                if (i+cs >= FN):
                    break
                velocities_t0 = vel[i]  # ensemble velocities (SeletedAtomNum, 3)
                velocities_t = vel[i+cs] 
                v0v0 += np.sum(velocities_t0 * velocities_t0)
                v0vt += np.sum(velocities_t0 * velocities_t)
            WACF.append(v0vt / v0v0)

        wacf = np.array(WACF)
        dump_pkl(dump_name, wacf)
    
    def get_wacf_inertiaPy(self, group_names, dump_name, correlation_step=None, interval=None, drift_correction=True, calc_anvel=False):
        unwrap=True
        print(f'drift_correction={drift_correction}')
        print('calculating total angular velocities...')
        # group_names = ['PS4_0', 'PS4_1', ...]
        for gn in group_names:
            assert gn in self.idx_d, f"no atom group {group_names} exists, cannot calculate its power spectrum."
        if self.dt is None:
            raise ValueError("timestep dt is not supplied, it must be supplied to calculate velocity from trajectory")
        MolNum = len(group_names)
        total_lt_1d=list(chain(*[self.idx_d[gn] for gn in group_names]))
        #total_lt_2d=list()
        ##
        #for gn in group_names:
        #    total_lt_2d.append(list(self.idx_d[gn]))
        #nodes=find_node(total_lt_2d)
        ##
        group_atom_num = len(total_lt_1d)
        print(f'MolNum={MolNum}')

        coords=np.zeros((self.FrameNum, self.AtomNum, 3))
        coords_group=np.zeros((self.FrameNum,group_atom_num,3))
        inertia_tensor=np.zeros((self.FrameNum,MolNum*3,3))
        #ligand_com_dist=np.zeros((self.FrameNum, group_atom_num))
        
        vel=np.zeros((self.FrameNum-2, group_atom_num,3))

        for i in range(self.FrameNum):
            coords[i]=copy.deepcopy(self[i].coords)
            u, d = 0, 0 # or simply d, dd = 0, 0
            uu, dd = 0, 0
            for gn in group_names:
                idx_lt=list(self.idx_d[gn]) 
                mass_lt=self.mass_lt[idx_lt]
                l=len(idx_lt)

                #unwrap  and... numpy magic!
                coord=self._get_mol_unwrap(idx_lt,i)
                coords[i][idx_lt]=coord
                u=d
                d+=l

                com = get_com(coord,mass_lt)
                coord_com = coord-com
                coords_group[i][u:d]=coord_com
                # note that coords is unwrapped but no com reference coordinates
                # but coords_group is unwrapped and com reference coordinates
                uu = dd
                dd += 3
                inertia_tensor[i][uu:dd]=get_inertia(coord_com, mass_lt)
        #print(coords) # unwrapped system coordinates
        #print('----') 
        #print(coords.shape)
        #WriteToXyz(coords,self.atom_lt,'unwrapped.xyz')
        #exit()
        #print(coords_group) # inspected group `total` coodinates under its own molecular com reference.

        # one frame PS4 under center of mass examination
        #r0, r1 = 0, 0
        #for i, gn in enumerate(group_names):
        #    l = len(self.idx_d[gn])
        #    r0 = r1
        #    r1 += l
        #    c=coords_group[0][r0:r1]
        #    atom_lt=self.atom_lt[list(self.idx_d[gn])]
        #    WriteToXyz(coords_group[0][r0:r1].reshape(1,l,3), atom_lt, f'PS4_{i}.xyz')
        #exit()
            #print(idx_lt)

        #print(inertia_tensor) # inspected group inertia tensor under its own molecular com reference.
        ## inertia tensor solved together with the `coords_group`
        #exit()

        # velocity section
        for i in range(self.FrameNum-2):
            if drift_correction:
                _com, com_ = get_com(coords[i], self.mass_lt), get_com(coords[i+2], self.mass_lt)
                central_drift=com_ - _com
                coords_group[i+2] -= central_drift
            vel[i] = (coords_group[i+2] - coords_group[i]) / self.dt / 2

        coords_group=coords_group[1:-1]
        
        H = np.zeros((self.FrameNum-2, MolNum, 3))
        # `coords_group` Shape(FrameNum-2,group_atom_num,3); 
        # `vel` Shape(FrameNum-2,group_atom_num,3); 
        # `inertia_tensor Shape(FrameNum-2, group_atom_num*3, 3)
        # `H` Shape(self.FrameNum-2, MolNum, 3)
        for i in range(self.FrameNum-2):
            d, u = 0,0
            for j,gn in enumerate(group_names):
                idx_lt=list(self.idx_d[gn])
                mass_lt = self.mass_lt[idx_lt]
                l = len(mass_lt)
                mass_lt_2d = mass_lt.reshape(l, 1)
                d=u
                u+=l

                to_add=np.cross(coords_group[i][d:u], vel[i][d:u]) * mass_lt_2d # Shape(AtomNumPerMol, 3)
                #print(to_add.shape)
                #exit()
                H[i][j]=np.sum(to_add, axis=0) # (3,) vector
                #H[i][j] = np.sum(np.cross(coords_group[i][d:u], vels[i][d:u]) * mass_lt_2d, axis=1)

        anvel = np.zeros((self.FrameNum-2, MolNum, 3))
        # the last loop for angular velocity...
        for i in range(self.FrameNum-2):
            for j in range(MolNum):
                anvel[i][j]=np.matmul(inv(inertia_tensor[i][j*3:(j+1)*3]), H[i][j]) 

        if calc_anvel:
            dump_pkl(dump_name, anvel)
            return
                 
        print('calculating wacf...')
        WACF = []
        FN = self.FrameNum - 2
        for cs in range(correlation_step):
            w0w0 = 0
            w0wt = 0
            for i in range(0,FN,interval):
                if (i+cs >= FN):
                    break
                anvelocities_t0 = anvel[i]  # ensemble velocities (SeletedAtomNum, 3)
                anvelocities_t = anvel[i+cs] 
                w0w0 += np.sum(anvelocities_t0 * anvelocities_t0)
                w0wt += np.sum(anvelocities_t0 * anvelocities_t)
            WACF.append(w0wt / w0w0)

        wacf = np.array(WACF)
        dump_pkl(dump_name, wacf)

    def get_LL(self, MolIdx, DiffAtom):
        """sodium linear momentum nad PS4 com linear momentum correlation function.?
           get_LL([109,188,185,206,203], 'Na')
        """
        # now both PS4 and Na are unwrapped.
        lt = []
        for i in self.idx_d[DiffAtom]:
            if self._get_dist_unwrap(MolIdx[0],i) < 5.1:
                lt.append(i)
        new_lt = MolIdx + lt
        self.write_to_xyz(new_lt, f'Frame{self.FrameNum}_near.xyz')

    def get_HL(self):
        """sodium linear momentum and PS4 angular momentum correlation function.?"""

    def get_com_vacf(self, group_names, dump_name, correlation_step=None, interval=None, drift_correction=True, calc_v=False):
        """with ana.XYZL(lat_vec=lat_vec, dt=dt, path=path) as trajl:
            #
            trajl.setup_group({'Mol1': Sequence(P_atom_idx, S1_atom_idx, S2_atom_idx, S3_atom_idx, S4_atom_idx)})
            trajl.get_com_vacf(['Mol1',])
            #
            MAKE SURE it is unwrapped.?
            with ana.XYZL(lat_vec=lat_vec, dt=dt, path=path) as trajl:
                # MAKE SURE the central atom is at index0
                trajl.setup_group({'Mol1': Sequence(P_atom_idx, S1_atom_idx, S2_atom_idx, S3_atom_idx, S4_atom_idx)})
                trajl.setup_group({'Mol2': Sequence(P_atom_idx, S1_atom_idx, S2_atom_idx, S3_atom_idx, S4_atom_idx)})
                trajl.get_com_vacf(['Mol1','Mol2'])
        """
        unwrap=True
        print(f'drift_correction={drift_correction}')
        print('calculating group center of mass velocities...')
        for gn in group_names:
            assert gn in self.idx_d, f"no atom group {group_name} exists, cannot calculate its power spectrum."
        if self.dt is None:
            raise ValueError("timestep dt is not supplied, it must be supplied to calculate velocity from trajectory")

        MolNum = len(group_names)
        print(f'MolNum={MolNum}')
        #count = 0
        #_lt, lt_ = [], []
        #lt=list(chain(*[self.idx_d[gn] for gn in group_names]))

        vel=np.zeros((self.FrameNum-2,MolNum,3))
        for i in range(self.FrameNum - 2):
            #if count == 1:
            #    exit()
            #count += 1
            r1, r2 = self[i].coords, self[i+2].coords
            _com_all = np.zeros((MolNum, 3))
            com_all_ = np.zeros((MolNum, 3))
            for j, gn in enumerate(group_names):
                idx_lt = list(self.idx_d[gn])
                mass_lt = self.mass_lt[idx_lt]
                _mol_unwrapped=self._get_mol_unwrap(idx_lt,i)
                mol_unwrapped_=self._get_mol_unwrap(idx_lt,i+2)

                r1[idx_lt], r2[idx_lt] = _mol_unwrapped, mol_unwrapped_
                _com_all[j] = get_com(_mol_unwrapped, self.mass_lt[idx_lt])
                com_all_[j] = get_com(mol_unwrapped_, self.mass_lt[idx_lt])

            # now `r1` and `r2` is unwrapped in current frame.
            if drift_correction:
                central_drift=get_com(r2,self.mass_lt) - get_com(r1,self.mass_lt)
                com_all_ -= central_drift
            vel[i] = (com_all_ - _com_all) / 2 / self.dt
            #print(vel)
            #print(vel.shape)
        if calc_v:
            dump_pkl(dump_name, vel)
            return 
            
        print('calculating vacf...')
        vel_frame_num = self.FrameNum-2 # first and last frame have no velocities.
        VACF = []
        for cs in range(correlation_step):
            v0v0 = 0
            v0vt = 0
            for i in range(0,vel_frame_num,interval):
                if (i+cs >= vel_frame_num):
                    break
                velocities_t0 = vel[i]  # ensemble velocities (SeletedAtomNum, 3)
                velocities_t = vel[i+cs]   # (3,)
                v0v0 += np.sum(velocities_t0*velocities_t0)
                v0vt += np.sum(velocities_t0*velocities_t)
            VACF.append(v0vt / v0v0)

        vacf = np.array(VACF)
        dump_pkl(dump_name, vacf)

    def _get_mol_unwrap(self, mol_idx, i=None):
        """wrap the coordinates with respect to the central atom.
        the central atom is the first element in the mol_idx list.

        changes: make sure the atom you are to wrap(or fix) about  are 
        in the first and snd element of the `mol_idx` list respectively.
        """
        mol_frac_coords= np.array([self._get_fc(idx, i) for idx in mol_idx])  
        # note that numpy array does NOT copy the <np object> even after slicing 
        fcc = mol_frac_coords[0]
        ligand_frac_coords = mol_frac_coords[1:]
        for fcl in ligand_frac_coords:
            for i in range(3):
                d = fcc[i] - fcl[i]
                if d > .5:
                    fcl[i] += 1
                elif d < -.5:
                    fcl[i] -= 1
        return np.array([self._get_cc(fc) for fc in mol_frac_coords])

    def _get_vel(self, group_name, mass_weighted):
        """
        `velocity generator`
        get half step kick velocity in step 1 [ v(t + .5dt), v(t + 1.5dt), v(t + 2.5dt) ... ]
        if we have N Frames Then VelFrame generated is (N-1) Frame.  
        WARNINNGS: Unit of the velocity
        PITFALLS: 0-based Frame!
        """
        assert group_name in self.idx_d, f"no atom group {group_name} exists, cannot calculate its power spectrum."

        if self.dt is None:
            raise ValueError("timestep dt is not supplied, it must be supplied to calculate velocity from trajectory")

        idx_lt = self.idx_d[group_name]
        mass_lt2 = np.sqrt(self.mass_lt[idx_lt])
        #idx_lt = [i for i in range(len(self.atom_lt)) if self.atom_lt[i] == atom_symbol]
        if len(idx_lt) == 0:
            raise ValueError(f'no atom kind {atom_symbol} exists, please check...')

        for i in range(self.FrameNum-2):
            r1, r2 = self[i].coords, self[i+2].coords
            com1, com2 = get_com(r1, self.mass_lt), get_com(r2, self.mass_lt) 
            v = (r2[idx_lt] - r1[idx_lt] - (com2 - com1)) / self.dt / 2 # central drift corrected velocity
            if mass_weighted:
                for i in range(len(idx_lt)):
                    v[i] *= mass_lt2[i]
                yield v
            else:
                yield v

    def _get_vel_para(self, group_name, NumProcess):
        pass
        

    def get_vacf(self, group_name, correlation_step, interval, dump_name, mass_weighted=False):
        print('calculating velocities...') 
        vel = np.array([v for v in self._get_vel(group_name, mass_weighted)])
        
        print('calculating vacf...')
        vel_frame_num = self.FrameNum-2 # first and last frame have no velocities.
        VACF = []
        for cs in range(correlation_step):
            v0v0 = 0
            v0vt = 0
            for i in range(0,vel_frame_num,interval):
                if (i+cs >= vel_frame_num):
                    break
                velocities_t0 = vel[i]  # ensemble velocities (SeletedAtomNum, 3)
                velocities_t = vel[i+cs] 
                v0v0 += np.sum(velocities_t0 * velocities_t0)
                v0vt += np.sum(velocities_t0 * velocities_t)
            VACF.append(v0vt / v0v0)

        vacf = np.array(VACF)
        dump_pkl(dump_name, vacf)

    def calc_dist_density(self, 
                          central_atom, 
                          ligand_atom, 
                          sizer, 
                          density_filename, 
                          fe_filename, 
                          smooth=False, 
                          smooth_level=None,
                          dr=.1,):
        """within the shell of a point, inspect the chosen atoms distribution density and free energy curve.
        trajl.setup_group(central_atom)
        trajl.setup_group(distributed_atom)
        trajl.calc_densitiy_r(central_atom, distributed_atom, sizer)
        
        arguments
        ------------
        central_atom 
        distributed_atom
        sizer `shell radius` in Angstrom
        """
        if not hasattr(self, 'T'):
            raise AttributeError('temperature needed to calculate the free energy')
        # density 
        print('calculating distance density distribution...')
        nbins = int(sizer/dr) # if dr=.1 then we have for instance 12.5xxx, and 125 nbins will be set up.
        dist_bin = np.zeros(nbins)
        ci_lt, li_lt =self.idx_d[central_atom], self.idx_d[ligand_atom]
        #for i, f in enumerate(self):
        for f in self:
            #if i % 5000 == 0:
            #    print(f'calculating {i}th Frame')
            for ci in ci_lt:
                for li in li_lt:
                    r=self._get_dist_unwrap(ci, li)
                    if r >= sizer:
                        continue
                    ir = int(r/dr)
                    dist_bin[ir] += 1 / r / r

        # normalize
        dist_bin = dist_bin / self.FrameNum / len(ci_lt) / len(li_lt)
        print(f'dumpping density distribution to {density_filename}')
        if smooth:
            new_dist_bin=gaussian_filter(dist_bin, sigma=smooth_level)
        dump_pkl(density_filename, new_dist_bin)

        # free energy
        print('calculating free energy curve of energy...')

        FES=[]
        for v in dist_bin:
            fes=-self.T*np.log(v) * constants.physical_constants['Boltzmann constant in eV/K'][0]
            FES.append(fes)
        dump_pkl(fe_filename, np.array(FES))

    def get_power_spectrum(self, acf_file, omegamax, domega):
        acf = load_pkl(acf_file)  
        omega = np.arange(0, omegamax, domega)  # omega in THz
        ps = np.zeros(len(omega))
        
        for i in range(len(ps)):
            for j in range(len(acf)):
                ps[i] += acf[j] * np.cos(omega[i] * (j+1) * self.dt) * self.dt * 2

    def get_jtcf(self, triplet, cutoff1, cutoff2, degree,_dump_pkl=True, _dump_npy=False, dump_name=None):
        """`joint time correlation function` combines `bvcf` and `rtcf`.
        arguments
        ------------------------
        triplet=['P', 'S', 'Na']
        P-S -- the reorientation vector 
        Na -- the diffusion cation
        ------------------------
        cutoff1=reorientation vector cutoff (real chemical bond)
        cutoff2=central_atom and sodium distance cutoff
        ------------------------
        degree =1 or 2 (degree of Legendre polynomials)
        """
        #pair_lt = self._setup_pair(triplet[::2], cutoff2)
        tri_lt = self._setup_triplet(triplet, cutoff1, cutoff2)
            
        pseudo_pair_lt = [t[::2] for t in tri_lt]
        legendre_lt0, h_lt0 = [], []
        
        for p in pseudo_pair_lt:
            h_lt0.append(self._get_h(p, cutoff2))
        for t in tri_lt:
            legendre_lt0.append(self._get_legendre(t, degree))
        clt0 = np.array(h_lt0) * np.array(legendre_lt0)

        res = []
        FrameNum = 0
        for f in self:
            if not (FrameNum % 10000):
                print(f'calculating {FrameNum//1e3}ps... ')
            h_lt = []
            legendre_lt = []
            for p in pseudo_pair_lt:
                h_lt.append(self._get_h(p, cutoff2))
            for t in tri_lt:
                legendre_lt.append(self._get_legendre(t, degree))
            res.append(sum(np.array(h_lt) * np.array(legendre_lt) * clt0))
            FrameNum += 1
        self.jtcf = np.array(res) / res[0]

        if _dump_pkl:
            if dump_name is None:
                dump_pkl(f'db/jtcf/{self.temperature}K_{self.run_type}_{degree}{degree}.pkl', self.jtcf)
            else:
                dump_pkl(Path.cwd() / dump_name, self.jtcf)

    def _get_h(self, pair, cutoff):
        """
        pair=[1,2] #['P', 'Na'] 
        """
        d = self._get_dist_pair_wrapped(pair[0], pair[1])
        if d < cutoff:
            return 1.
        return 0.

    def get_bvcf(self, triplet, cutoff1, cutoff2, degree,_dump_pkl=True, _dump_npy=False, dump_name=None):
        """
        `bond vector correlation function` 
        get_bvcf11(triplet=['P', 'S', 'Na'], cutoff1=3., cutoff2=6.3, degree=1 or 2) 
        note that cutoff1 for P-S and cutoff2 for P-Na.
        """
        tri_lt = self._setup_triplet(triplet, cutoff1, cutoff2)
        legendre_lt0 = list()
        res = list()
        FrameNum = 0
        for t in tri_lt:
            legendre_lt0.append(self._get_legendre(t, degree))

        for f in self:
            if not (FrameNum % 10000):
                print(f'calculating {FrameNum//1e3}ps... ')
            legendre_lt = list()
            for t in tri_lt:
                legendre_lt.append(self._get_legendre(t, degree))
            res.append(np.dot(legendre_lt0, legendre_lt))
            FrameNum += 1

        self.bvcf = np.array(res) / res[0] # normalize
        if _dump_pkl:
            if dump_name is None:
                dump_pkl(f'db/bvcf/{self.temperature}K_{self.run_type}_{degree}{degree}.pkl', self.bvcf)
            else:
                dump_pkl(Path.cwd() / dump_name, self.bvcf)
        
    def _get_legendre(self, t, degree): 
        """for degree = 1 or 2 scenario only"""
        central_coord = self.coords[t[0]]
        v1 = self.coords[t[1]] - central_coord  # P -> S
        v2 = self.coords[t[2]] - central_coord  # P -> Na
        cos_theta = np.dot(v1, v2) / norm(v1) / norm(v2)
        if degree == 1:
            return cos_theta
        cos_dtheta = np.cos(2 * np.arccos(cos_theta)) 
        P2 = .25 * (3 * cos_dtheta + 1)
        return P2

    def _setup_triplet(self, triplet, cutoff1, cutoff2):
        tri_lt = list()
        pair_lt1=self._setup_pair(triplet[:-1], cutoff1)
        pair_lt2=self._setup_pair(triplet[::2], cutoff2)
        for p1 in pair_lt1:
            for p2 in pair_lt2:
                if (len(cat_to_mol([p1,p2])) == 1):
                    tri_lt.append(triangle_build(p1, p2))
        return tri_lt

    def _setup_pair(self, pair, cutoff, wrap=True):
        """
        setup pair list based on initial frame
        """
        pair_lt = []
        center = pair[0] # element symbol 
        ligand = pair[1]
        c_lt = [i for i in range(self.AtomNum) if self.atom_lt[i] == center]
        l_lt = [i for i in range(self.AtomNum) if self.atom_lt[i] == ligand]

        for c_idx in c_lt:
            for l_idx in l_lt:
                if wrap:
                    d = self._get_dist_pair_wrapped(c_idx, l_idx)
                else:
                    v = self.coords[c_idx] - self.coords[l_idx]
                    d = norm(v)
                if d < cutoff:
                    pair_lt.append((c_idx,l_idx))
        return pair_lt

    def _get_vec_unwrap(self, c_idx=None, l_idx=None, cc=None, lc=None):
        # keep the central atom fixed, unwrap the ligand atom only.
        # OLD :: assume at most only one box away 
        # NEW :: assume now any boxes away
        if cc is None:
            fcc = self._get_fc(c_idx) # not any copied , but a newly created 3-lengthed array in memory.
            fcl = self._get_fc(l_idx)
        else:
            fcc = self._get_fc(c=cc)
            fcl = self._get_fc(c=lc)

        # first all wrapped back
        fcc = fcc - np.array(fcc,dtype=np.int32)
        for i in range(len(fcc)):
            if fcc[i] < 0:
                fcc[i] += 1
        fcl = fcl - np.array(fcl,dtype=np.int32)
        for i in range(len(fcl)):
            if fcl[i] < 0:
                fcl[i] += 1

        for i in range(3):
            d = fcc[i] - fcl[i]
            if d > .5:
                fcl[i] += 1
            elif d < -.5:
                fcl[i] -= 1
        fr = fcl - fcc 
        cr = self._get_cc(fr)
        return cr

    def _get_dist_unwrap(self, c_idx=None, l_idx=None, cc=None, lc=None):
        return norm(self._get_vec_unwrap(c_idx, l_idx, cc, lc))

    def get_neighbours(self, atom_symbol, threshold, aver=True):
        c = []
        idx_lt = [i for i in range(self.AtomNum) if self.atom_lt[i] == atom_symbol]
        for e, idx in enumerate(idx_lt):
            c.append({})
            for at in self.type_lt:
                c[e][at] = 0
            for i in range(self.AtomNum):
                d=self._get_dist_unwrap(idx, i)
                if d < threshold:
                    s=self.atom_lt[i]
                    c[e][s] += 1
        # self correction
        for cc in c:
            cc[atom_symbol] -= 1
        print(f'Central Atom(shell radius{threshold}): ', atom_symbol)

        if aver:
            aver_c = {}
            l = len(c)
            for at in self.type_lt:
                summation = 0
                for d in c:
                    summation += d[at] 
                aver_c[at] = summation / l
            print('averaged :', aver_c)
        else:
            print(c)

        print('Nr. neighbouring atoms: ', sum(v for v in aver_c.values()))

    def _get_fc(self, idx=None, i=None, c=None):
        """ two scenarios:
            idx is None and i is None but c is not None;
            c is None and idx is not None and i can be either None or not.
        """
        if c is None:
            if i is None:
                return np.matmul(self.coords[idx], self.ilat_vec)
            elif isinstance(i, int|np.int64):
                return np.matmul(self[i].coords[idx], self.ilat_vec)
        if idx is None:
            return np.matmul(c, self.ilat_vec)


    def _get_cc(self, fc):
        return np.matmul(fc, self.lat_vec)

    def _test(self):
        for i, f in enumerate(self):
            print(i)

        lt = np.array([v for v in self._get_vel('P')])
        #self[self.FrameNum-1]   # self.flag now is set to `True`
        for i, f in enumerate(self):
            print(i)
        for i, f in enumerate(self):
            print(i)

    def dump_unwrap_coord(self, unwrap_lt, file_out_name):
        """a huge `self.coords` array will be dumped to file.
        in cp2k, if it is unwrapped the output coordinates are always unwrapped.
        with ana.XYZL(lat_vec=lat_vec, path=path, dt=dt) as trajl:
            self.unwrap_coord([(P_i, S_i, S_i, S_i, S_i), (central_Atom_i, ligand_atom_i,...),...], 'coords.pkl')
        arguments
        -----------
        tell the function which molecule needs to be unwrapped
        """
        unwrap_lt_1d=list(chain(*unwrap_lt))
        rest_lt_1d=exclude(unwrap_lt_1d, range(self.AtomNum))
        coords=np.zeros((self.FrameNum, self.AtomNum, 3))

        for i, f in enumerate(self):
            for t in unwrap_lt:
                c=self._get_mol_unwrap(t)
                coords[i][list(t)] = c
            for j in rest_lt_1d:
                coords[i][j] = self.coords[j]
        dump_pkl(file_out_name, coords)

    def _calc_angle_density(self, group_names, include, central_atom, two_thetas=False):
        if two_thetas:
            self._get_thetas(group_names, include, central_atom)
        else:
            self._get_angles(group_names, include, central_atom,) # generate self.theta(FrameNum, IncludeNum) and self.phi(FrameNum, IncludeNum)

        self.DictDen = {}
        if two_thetas:
            assert self.thetaX.shape == self.thetaY.shape == self.thetaZ.shape
        else:
            assert self.theta.shape == self.phi.shape
        d,u=0,0
        for k, gn in enumerate(group_names):
            d=u
            u+=len(include[k])
            nbins_th = int(180/self.dth[k])
            nbins_phi = int(360/self.dphi[k])
            if two_thetas:
                nbins_phi = int(180/self.dth[k])  # for convience
        # then 180 and 360 bins for theta and phi respectively
        # thus 0, 1, ..., 179 and 0, 1, .., 359 indices for them, respectively
            angles_histogram = np.zeros((nbins_phi, nbins_th))
            if two_thetas:
                theta, phi= self.thetaX[:,d:u], self.thetaY[:,d:u]
            else:
                theta, phi = self.theta[:,d:u], self.phi[:,d:u]
            d1, d2 = theta.shape
            for i in range(d1):
                for j in range(d2):
                    angles_histogram[int(phi[i][j]/self.dphi[k]), int(theta[i][j]/self.dth[k])] += 1
            angles_histogram=gaussian_filter(angles_histogram, sigma=self.sigma[k])
            self.DictDen[gn] = angles_histogram

    def get_angle_fes(self,group_names,include,central_atom,T,dump_name,freq,
                      two_thetas=False, dth=None,dphi=None, sigma=None):
        self._setup_angle(dth, dphi, central_atom, sigma, two_thetas)
        self._calc_angle_density(group_names, include, central_atom,two_thetas)

        self.DictFes = {}
        for i, (k, v) in enumerate(self.DictDen.items()):
            normfac2 = 1. / (2 * np.pi * self.dth[i] * self.dphi[i]) / (self.FrameNum*freq) / (freq*self.FrameNum) / len(include[i])
            ll,mm = int(360/self.dphi[i]), int(180/self.dth[i])
            if two_thetas:
                ll,mm = int(180/self.dth[i]), int(180/self.dth[i])
            fes = np.zeros((ll,mm))
            for iphi in range(ll):
                for ith in range(mm):
                    fes[iphi,ith] = -(T * np.log(v[iphi,ith] * normfac2)) / evtok
            self.DictFes[k] = fes

        # create and dump `Angle` object
        if two_thetas:
            Angle(self.dth, self.dphi, self.thetaX, self.thetaY, group_names, include, 
                    central_atom, sigma, self.DictDen, self.DictFes, T, dump_name)
        else:
            Angle(self.dth, self.dphi, self.theta, self.phi, group_names, include, 
                    central_atom, sigma, self.DictDen, self.DictFes, T, dump_name)

    def detect_rotation_bond_vector(self, group_names, vec, central_atom, angle_threshold, count=None):
        """
        with XYZL(...) as trajl:
            trajl.setup_group(...)
            trajl._detect_rotation_bond_vector([PS4_0, PS4_2,...,PS4_97], [[local_P_idx, local_S_idx], ...], [Fasle,...])
            trajl._detect_rotation_bond_vector([B10H10_0,..., B10H10_100], [[local_B_idx], [local_B_idx],...], [True,...])
        group_names::group_names several group name grouped in a list
        vecs::vecs Nx3 array
        central_atom::list
        """

        for gn in group_names:
            assert gn in self.idx_d, f"no atom group {group_name} exists, cannot calculate its angles."
        assert len(group_names) == len(include) == len(central_atom), "three input arguments don't have equal outermost length, plz check"

        if central_atom:
            assert len(vec) == 2
        print('incomplete')
        if count is None:
            count = self.FrameNum
        assert isinstance(count, int)
        x, y, z = np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])
        THETAX = np.zeros((count, len(list(chain(*include))) ))
        THETAY = np.zeros((count, len(list(chain(*include))) ))
        THETAZ = np.zeros((count, len(list(chain(*include))) ))
                                 

        for i, f in enumerate(self):
            if i == count:
                self.thetaX = THETAX
                self.thetaY = THETAY
                self.thetaZ = THETAZ
                return 
            d=0
            u=0
            for j,gn in enumerate(group_names):
                d=u
                u+=len(include[j])

                lt=list(self.idx_d[gn])
                coord=self._get_mol_unwrap(lt)
                if central_atom[j]:
                    vecs = coord[list(include[j])] - coord[0]
                else:
                    com=get_com(coord, self.mass_lt[lt])
                    vecs = coord[list(include[j])] - com
                # calculate `phi` and `theta`
                cos_thetasX = np.dot(vecs, x) / norm(vecs, axis=1)
                cos_thetasY = np.dot(vecs, y) / norm(vecs, axis=1)
                cos_thetasZ = np.dot(vecs, z) / norm(vecs, axis=1)
                thetasX = np.rad2deg(np.arccos(cos_thetasX))
                thetasY = np.rad2deg(np.arccos(cos_thetasY))
                thetasZ = np.rad2deg(np.arccos(cos_thetasZ))
                THETAX[i][d:u] = thetasX
                THETAY[i][d:u] = thetasY
                THETAZ[i][d:u] = thetasZ

        #print(THETA)
        #print('\n')
        #print(PHI)
        # reshape for later indexing...
        
        self.thetaX = THETAX
        self.thetaY = THETAY
        self.thetaZ = THETAZ

    def _get_thetas(self, group_names, include, central_atom, count=None):
        for gn in group_names:
            assert gn in self.idx_d, f"no atom group {group_name} exists, cannot calculate its angles."
        assert len(group_names) == len(include) == len(central_atom), "three input arguments don't have equal outermost length, plz check"


        if count is None:
            count = self.FrameNum
        assert isinstance(count, int)
        x, y, z = np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])
        THETAX = np.zeros((count, len(list(chain(*include))) ))
        THETAY = np.zeros((count, len(list(chain(*include))) ))
        THETAZ = np.zeros((count, len(list(chain(*include))) ))
                                 
        print(f'calculating total group number: {len(central_atom)}')
        for i, f in enumerate(self):
            if i == count:
                self.thetaX = THETAX
                self.thetaY = THETAY
                self.thetaZ = THETAZ
                return 
            d=0
            u=0
            for j,gn in enumerate(group_names):
                d=u
                u+=len(include[j])

                lt=list(self.idx_d[gn])
                coord=self._get_mol_unwrap(lt)
                if central_atom[j]:
                    vecs = coord[list(include[j])] - coord[0]
                else:
                    com=get_com(coord, self.mass_lt[lt])
                    vecs = coord[list(include[j])] - com
                # calculate `phi` and `theta`
                cos_thetasX = np.dot(vecs, x) / norm(vecs, axis=1)
                cos_thetasY = np.dot(vecs, y) / norm(vecs, axis=1)
                cos_thetasZ = np.dot(vecs, z) / norm(vecs, axis=1)
                thetasX = np.rad2deg(np.arccos(cos_thetasX))
                thetasY = np.rad2deg(np.arccos(cos_thetasY))
                thetasZ = np.rad2deg(np.arccos(cos_thetasZ))
                THETAX[i][d:u] = thetasX
                THETAY[i][d:u] = thetasY
                THETAZ[i][d:u] = thetasZ

        #print(THETA)
        #print('\n')
        #print(PHI)
        # reshape for later indexing...
        
        self.thetaX = THETAX
        self.thetaY = THETAY
        self.thetaZ = THETAZ

    def _get_angles(self, group_names, include, central_atom, count=None):
        """ if `central_atom` argument is True, then we compute the Vector(central_atom - ligand_atom)
        otherwise, Vector(center of mass - ligand_atom) is computed.

        arguments
        -------------
        group_names :: You need to first set up group
        include :: Though you set up the molecule/cluster in group_names (self.idx_d) but you might 
        not get all the vector based on the atoms. You might only want part of them thus the `include` helps
        
        I have, e.g., a water and two PS4s molecule
        d = {'H2O': [0,1,2], 'PS4j': [3,4,5,6,7], 'PS4k': [8,9,10,11,12]}
        name_lt = ['H2O', 'PS4j', 'PS4k']
        with ana.XYZL(lat_vec=lat_vec, path=path) as trajl:
            trajl.setup_group(d)
            trajl._get_angles(name_lt, [[1,2],[1,2],[1,2,3,4]], [True, True, False])
        """
        for gn in group_names:
            assert gn in self.idx_d, f"no atom group {group_name} exists, cannot calculate its angles."
        assert len(group_names) == len(include) == len(central_atom), "three input arguments don't have equal outermost length, plz check"

        if count is None:
            count = self.FrameNum
        assert isinstance(count, int)
        z = np.array([0.,0.,1.])
        THETA, PHI = np.zeros((count, len(list(chain(*include))) )), np.zeros((count, len(list(chain(*include))) ))
        for i, f in enumerate(self):
            if i == count:
                self.theta = THETA
                self.phi = PHI
                return 
            d=0
            u=0
            for j,gn in enumerate(group_names):
                d=u
                u+=len(include[j])

                lt=list(self.idx_d[gn])
                coord=self._get_mol_unwrap(lt)
                if central_atom[j]:
                    vecs = coord[list(include[j])] - coord[0]
                else:
                    com=get_com(coord, self.mass_lt[lt])
                    vecs = coord[list(include[j])] - com
                # calculate `phi` and `theta`
                cos_thetas = np.dot(vecs, z) / norm(vecs, axis=1)
                thetas = np.rad2deg(np.arccos(cos_thetas))
                THETA[i][d:u] = thetas

                phis = self._match_it(vecs)
                PHI[i][d:u] = phis
        #print(THETA)
        #print('\n')
        #print(PHI)
        # reshape for later indexing...
        
        self.theta = THETA
        self.phi = PHI

    def get_angles(self, group_names, include, central_atom, which=0, count=None):
        if which == 0:
            self._get_angles(group_names, include, central_atom, count)
            return self.theta, self.phi
        elif which == 1:
            self._get_thetas(group_names, include, central_atom, count)
            return self.thetaX, self.thetaY, self.thetaZ
        else:
            raise NotImplementedError

    def _match_it(self, vecs):
        vecs_2d = vecs[:,:2]
        l = vecs_2d.shape[0]
        p=np.zeros(l)

        for i in range(l):
            x, y = vecs_2d[i][0]+1e-4, vecs_2d[i][1]+1e-4
            bond_len = norm(vecs_2d[i])
            match bool(x>0), bool(y>0):
                case (True, True) | (False, True):
                    p[i] = np.rad2deg(np.arccos(np.dot(vecs_2d[i], [1,0]) / bond_len))
                case True, False:
                    p[i] = 360 - np.rad2deg(np.arccos(np.dot(vecs_2d[i], [1,0]) / bond_len))
                case False, False:
                    p[i] = 180 + np.rad2deg(np.arccos(np.dot(vecs_2d[i], [-1,0]) /bond_len))
        return p

    def _setup_angle(self, dth, dphi, central_atom, sigma, two_thetas):
        l = len(central_atom)

        self.dth = dth
        if dth is None:
            self.dth = [2.,] * l
        if (len(self.dth) != l):
            raise ValueError

        self.dphi = dphi
        if dphi is None:
            self.dphi = [4.,] * l
            if two_thetas:
                self.dphi = self.dth
        if (len(self.dphi) != l):
            raise ValueError

        self.sigma = sigma
        if sigma is None:
            self.sigma = [3,] * l    
        if (len(self.sigma) != l):
            raise ValueError
    
    def get_r_fel(self, 
                  dump_name,
                  T,
                  ligand_atom, 
                  central_atom=None, 
                  threshold=2,
                  sizer=10., 
                  dr=.05, 
                  first_minimum=5., 
                  sigma=1,
                  mol_idx=None,
                  com=False):
        #if not (central_atom in self.idx_d and ligand_atom in self.idx_d):
        #    raise IndexError(f'{central_atom} or {ligand_atom} not all in self.idx_d, plz setup first')
        T = float(T)

        if central_atom is None:
            assert com 
            assert len(mol_idx) != 0


        dist_bin = np.zeros(int(sizer/dr))
        kb=constants.physical_constants['Boltzmann constant in eV/K'][0]
        print('calculating cation-anion fel...')
        if not com:
            c_idxs = self.idx_d[central_atom]
            l_idxs = self.idx_d[ligand_atom]
            for f in self:
                for c_idx in c_idxs:
                    for l_idx in l_idxs:
                        r=self._get_dist_unwrap(c_idx, l_idx)
                        if r <= sizer:
                            dist_bin[int(r/dr)] += 1 / r / r
        if com:
            l_idxs = self.idx_d[ligand_atom]

            for f in self:
                for mol in mol_idx:
                    c = get_com(self.coords[mol], self.mass_lt[mol])
                    for l_idx in l_idxs:
                        l = self.coords[l_idx]
                        r=self._get_dist_unwrap(cc=c,lc=l)
                        if r <= sizer:
                            dist_bin[int(r/dr)] += 1 / r / r

        for i in range(len(dist_bin)):
            if dist_bin[i] == 0:
                dist_bin[i] += 1e-8
        dist_bin=gaussian_filter(dist_bin, sigma=sigma)
        fel = np.zeros(len(dist_bin))
        for i in range(len(dist_bin)):
            fel[i] = -T * np.log(dist_bin[i]) * kb
        for i in range(len(fel)):
            if fel[i] > threshold:
                fel[i] = threshold

        m=min(fel[:int(first_minimum/dr)])
        fel = fel - m
        Radius(dist_bin, fel,central_atom, ligand_atom, T, threshold, sizer, dr, first_minimum, sigma, dump_name)

        #dump_pkl(dump_name, fel)

    def get_r_fel_first_shell(self, 
                              dump_name,
                              T,
                              ligand_atom, 
                              central_atom=None, 
                              tol=.5,
                              threshold=2,
                              dr=.05, 
                              sizer=10.,
                              first_minimum=5., 
                              sigma=1,
                              mol_idx=None,
                              com=False):
        #if not (central_atom in self.idx_d and ligand_atom in self.idx_d):
        #    raise IndexError(f'{central_atom} or {ligand_atom} not all in self.idx_d, plz setup first')
        T = float(T)

        if central_atom is None:
            assert com 
            assert len(mol_idx) != 0
            print('via com')


        kb=constants.physical_constants['Boltzmann constant in eV/K'][0]
        print('calculating cation-anion fel...')
        if not com:
            c_idx = self.idx_d[central_atom]
            l_idx = self.idx_d[ligand_atom]

            # first find the nearest distance...
            rmin = self._get_dist_unwrap(c_idx[0], l_idx[0])
            for c in c_idx:
                for l in l_idx:
                    ri = self._get_dist_unwrap(c, l)
                    if ri < rmin:
                        rmin = ri
            fst_shell = []
            comp = rmin + tol
            for c in c_idx:
                fst_shell.append([])
                for l in l_idx:
                    if self._get_dist_unwrap(c, l) < comp:
                        fst_shell[-1].append(l)
            distances = []
            for f in self:
                for i in range(len(fst_shell)):
                    for l in fst_shell[i]:
                        distances.append(self._get_dist_unwrap(c_idx[i], l))
            #l = int(max(distances)/dr) + 1
            #dist_bin = np.zeros(l)
            dist_bin = np.zeros(int(np.ceil(sizer/dr)))
            for r in distances:
                if r < sizer:
                    dist_bin[int(r/dr)] += 1 / r / r
            dump_pkl(dump_name, dist_bin)
            return 

    def get_r_fel_within_shell(self, 
                              dump_name,
                              ligand_atom, 
                              rmin,
                              rmax,
                              central_atom=None, 
                              tol=.2,
                              dr=.05, 
                              sizer=10.,
                              first_minimum=5., 
                              sigma=1,
                              mol_idx=None,
                              com=False):
        if central_atom is None:
            assert com 
            assert len(mol_idx) != 0
            print('via com')
        kb=constants.physical_constants['Boltzmann constant in eV/K'][0]
        print('calculating cation-anion fel...')
        if not com:
            c_idx = self.idx_d[central_atom]
            l_idx = self.idx_d[ligand_atom]

            within_shell = []
            comp_ = rmax+tol
            _comp = rmin-tol
            for c in c_idx:
                within_shell.append([])
                for l in l_idx:
                    if _comp < self._get_dist_unwrap(c, l) < comp_:
                        within_shell[-1].append(l)

            distances = []
            for f in self:
                for i in range(len(within_shell)):
                    for l in within_shell[i]:
                        distances.append(self._get_dist_unwrap(c_idx[i], l))

            #length = np.ceil((comp_ - _comp) /dr)
            #dist_bin = np.zeros(int(length))
            dist_bin = np.zeros(int(np.ceil(sizer/dr)))
            for r in distances:
                if r < sizer:
                    dist_bin[int(r/dr)] += 1 / r / r
            dump_pkl(dump_name, dist_bin)
            return 
        
    def get_fel_self(self, atom_name, fn, disp, direction=None, dr=.05):
        """get energy line away from its equilibrium position.
        direction = 'x', 'y', 'z' or 'xy', 'xz', 'yz'.
        """
        print('calculating fel from equil position...')
        idx = self.idx_d[atom_name]
        c0 = self.coords[idx]

        lt = []
        dist_arr = np.zeros((self.FrameNum,len(idx)))
        if direction is None or len(direction) == 3:
            if disp:
                for i, f in enumerate(self):
                    dist_arr[i] = norm(self.coords[idx]- c0, axis=1)
            else:
                for i, f in enumerate(self):
                    dist_arr[i] = norm(self.coords[idx] , axis=1)#- c0, axis=1)

        elif len(direction) == 1:
            d = self.axes_dict[direction]
            if disp:
                c0 = c0[:,d]
                for i, f in enumerate(self):
                    dist_arr[i] = self.coords[idx,d] - c0
            else:
                for i, f in enumerate(self):
                    dist_arr[i] = self.coords[idx,d]

        elif len(direction) == 2:
            d0,d1 = self.axes_dict[direction[0]], self.axes_dict[direction[1]]
            if disp:
                c0 = c0[:,[d0,d1]]
                for i, f in enumerate(self):
                    dist_arr[i] = norm(self.coords[idx][:,[d0,d1]]- c0, axis=1)
            else:
                for i, f in enumerate(self):
                    dist_arr[i] = norm(self.coords[idx][:,[d0,d1]],axis=1)# - c0, axis=1)

        dist_arr = np.abs(dist_arr.flatten())
        
        # now statistics...
        hists = np.zeros(int(np.max(dist_arr)/dr)+1)
        for d in dist_arr:
            hists[int(d/dr)] += 1
        hists = hists/np.sum(hists)
        x = np.arange(0, np.max(dist_arr), dr)
        Plot(x, hists, fn)
    
    def get_fel_self_with_lim(self, atom_name, fn, disp, _sizer, sizer_, direction=None, dr=.05, use_numpy=False):
        """get energy line away from its equilibrium position.
        direction = 'x', 'y', 'z' or 'xy', 'xz', 'yz'.
        """
        print('calculating fel from equil position...(with sizer)')
        idx = self.idx_d[atom_name]
        c0 = self.coords[idx]

        lt = []
        dist_arr = np.zeros((self.FrameNum,len(idx)))
        if direction is None or len(direction) == 3:
            if disp:
                for i, f in enumerate(self):
                    dist_arr[i] = norm(self.coords[idx]- c0, axis=1)
            else:
                for i, f in enumerate(self):
                    dist_arr[i] = norm(self.coords[idx] , axis=1)#- c0, axis=1)

        elif len(direction) == 1:
            d = self.axes_dict[direction]
            if disp:
                c0 = c0[:,d]
                for i, f in enumerate(self):
                    dist_arr[i] = self.coords[idx,d] - c0
            else:
                for i, f in enumerate(self):
                    dist_arr[i] = self.coords[idx,d]

        elif len(direction) == 2:
            d0,d1 = self.axes_dict[direction[0]], self.axes_dict[direction[1]]
            if disp:
                c0 = c0[:,[d0,d1]]
                for i, f in enumerate(self):
                    dist_arr[i] = norm(self.coords[idx][:,[d0,d1]]- c0, axis=1)
            else:
                for i, f in enumerate(self):
                    dist_arr[i] = norm(self.coords[idx][:,[d0,d1]],axis=1)# - c0, axis=1)

        dist_arr = np.abs(dist_arr.flatten())

        if use_numpy:
            pos= [d for d in dist_arr if _sizer<d<sizer_]
            counts, bin_edges = np.histogram(pos, bins=int((sizer_-_sizer)/dr))
            density = counts / (dr * len(pos))
            x = np.arange(_sizer,sizer_, dr)
            Plot(x, density, fn)
        # now statistics...
        #hists = np.zeros(int(np.max(dist_arr)/dr)+1)
        else:
            length = sizer_ - _sizer
            hists = np.zeros(int(length/dr))
            count = 0
            for d in dist_arr:
                if _sizer < d < sizer_:
                    i=(d - _sizer)/dr
                    hists[int(i)] += 1
                    count += 1
            norm_fac = dr * count
            x = np.arange(_sizer,sizer_, dr)
            Plot(x, hists/norm_fac, fn)

    def find_shell(self, idx, cutoff, vertex):
        """_get_dist_unwrap means to wrap with respect to the molecule and then get the distance."""
        lt = list(range(self.AtomNum))
        lt.pop(lt.index(idx))
        ltt = []
        for i in lt:
            r = self._get_dist_unwrap(i, idx)
            if r < cutoff and vertex == self.atom_lt[i]:
                ltt.append(i)
        return ltt
    
    def get_pair_hist(self, a1, a2, BinNum):
        idx_lt1 = self.idx_d[a1]
        idx_lt2 = self.idx_d[a2]
        lt = []
        for i1 in idx_lt1:
            for i2 in idx_lt2:
                lt.append(self._get_dist_unwrap(i1,i2))
        ax = plt.subplot()
        ax.hist(lt, BinNum)

    def get_site(self, group_name, fn, d1,d2, t1, t2):
        """t1, t2 :: threshold1 & threshold2 :: float
           d :: direction :: e.g. [0,1] [0,1,2], [1,2], [1]...

        """
        idx = self.idx_d[group_name]
        c0 = self.coords[idx]

        lt = []

        for i, f in enumerate(self):
            disp = self.coords[idx] - c0
            for dd in norm(disp[d1], axis=1):
                if t1<dd<t2:
                    lt.append(norm(disp[d2], axis=1)[0])
        dump_pkl(fn, lt)

    def ions_diff(self, atom, dump=False, dump_name=None):
        """get cation away from its initial position as a function of evolution time.
        average on all same atoms. But not ensemble average.
        """
        assert atom in self.idx_d, f'no {atom} in our dict, cannot compute the atom diff away from its position'
        lt = list(self.idx_d[atom])
        init = self.coords[lt]
        l = init.shape[0]
        diff = np.zeros(self.FrameNum)
        for i, f in enumerate(self):
            diff[i] = np.sum(norm(self.coords[lt] - init, axis=1)) / l
        if dump:
            dump_pkl(dump_name, diff) 
        else:
            return diff

    def ion_diff(self, idx):
        """return single ion (FrameNum,) displacement."""
        c0 = self.coords[idx]
        dp = np.zeros(len(self))
        for i, f in enumerate(self):
            dp[i] = norm(self.coords[idx] - c0)
        
        return dp

    def set_mol(self,threshold):
        # find bond connectivity
        lt=[]
        for j in range(self.AtomNum):
            for i in range(self.AtomNum):
                if i > j:
                    r = self._get_dist_unwrap(i,j)
                    if r < threshold:
                        lt.append((i,j))
                        #print(i,j)
        self.mol_lt=cat_to_mol(lt)
        #print(self.mol_lt)

    def vib_amplitude(self, 
                      idx_lt=None, 
                      atom_symbol=None, 
                      framework_idx=None, 
                      skip_step=0, 
                      calc_step=None):
        if calc_step is None:
            calc_step = self.FrameNum - skip_step
        assert calc_step <= self.FrameNum -skip_step
        if idx_lt is None:
            if atom_symbol is None:
                raise Exception
            idx_lt = self.idx_d[atom_symbol]
        c0 = self.coords[idx_lt]   # (ChosenAtomNum, 3)
            
        abs_disp=np.zeros((len(idx_lt), calc_step))
        for i, f in enumerate(self):
            if i < skip_step:
                continue
            # `NOT calc to which step` but ` in continuation with skip_step yet we
            # calculate how many steps
            elif i == calc_step+skip_step:
                break
            disp=self.coords[idx_lt]-c0
            abs_disp[:,i-skip_step]=np.sum(disp**2, axis=1)**.5


        amplitude=[]
        speed = np.zeros((len(idx_lt), calc_step))
        for l, atom in enumerate(abs_disp):
            #speed = atom[1] - atom[0]  # 1 - 0
            for i in range(1,calc_step):
                speed[l,i] = atom[i] - atom[i-1]
                if sign(speed[l,i]) == sign(speed[l,i-1]):
                    amplitude[-1] += speed[l,i]
                    #if abs(amplitude[-1]) > 70:
                    #    print(l, i)
                else:
                    #vib_count += 1
                    amplitude.append(speed[l,i])
                    #amplitude[-1] = speed[l,i]
        return amplitude

    @staticmethod
    def single_rot_amplitude(angle):

        amplitude=[]
        speed = np.zeros(len(angle))
        for i in range(1, len(angle)):
            speed[i] = angle[i] - angle[i-1]
            if sign(speed[i]) == sign(speed[i-1]):
                amplitude[-1] += speed[i]
            else:
                amplitude.append(speed[i])

        return amplitude

    @staticmethod
    def num_hits_amplitude(amp_lt, amp):
        return sum( abs(np.array(amp_lt)) > amp )

    def detect_variant(self, ref, timestep, interval):
        """
        return
        ---------------
        return the variant-constant blocks

        arguments
        ---------------
        timestep: real timestep between two frames.
        interval: how you choose the interval between the the frames... e.g., if your freq_dump=5, timestep=2,
        and you choose the interval =2, then the real time length between calculated angles for `variant rotation` 
        are then 2 * freq_dump * timestep = 20 fs
        """
        print('calculating variant rotation(bond vector)...')

        ref = list(ref)
        # for starter we set st = 0
        lt = []
        st = 0
        tailC = self[st].coords
        refV0 = tailC[ref][0] - tailC[ref][1]
        l0 = norm(refV0)
        for i in range(0, self.FrameNum, interval):
            # TODO accumulate triangle while detecting the hopping point
            # 1. determine hopping or not
            headC = self[i].coords[ref]
            refV1 = headC[0] - headC[1]
            angle = np.rad2deg(np.arccos( np.dot(refV0, refV1) / l0 / norm(refV1) ))
            lt.append(angle)
        return np.array(lt)

    def detect_invariant(self, tri_idx, ref, timestep, interval, run_t='UC', angle_threshold=50, lag=20, Round=10):
        self.lag = lag
        if run_t == 'UC':
            self._detect_variant(ref, timestep, interval, angle_threshold)
        else:
            self.dual = [[0, self.FrameNum],]

        print('calculating triangular rotation...')
        print(f'total block number: {len(self.dual)}')
        lt = []
        for count, blk in enumerate(self.dual):
            d, u = blk[0], blk[1]
            tri0 = self._get_tri_vec(self[d].coords[tri_idx], tri_idx)  # representation for triangle
            tri1 = self._get_tri_vec(self[d+1].coords[tri_idx], tri_idx)
            perp01 = np.cross(tri0, tri1)
            #lt.append([])
            #lt[-1].append(self._vec2deg(tri0, tri1))
            print(f'calculating block {count}...')
            lt.append(self._vec2deg(tri0, tri1))
            for i in range(d+2, u):
                tri2 = self._get_tri_vec(self[i].coords[tri_idx], tri_idx)
                angle = self._vec2deg(tri0, tri2)
                
                # determine range
                perp02 = np.cross(tri0, tri2)
                if np.dot(perp01, perp02) >= 0:
                    lt.append(angle)
                elif lt[-1] > 170:
                    lt.append(360-angle)
                else:
                    lt.append(angle)
        return np.array(lt)

    def _vec2deg(self, tri0, tri1):
        """return degree from 0-180. """
        return np.rad2deg( np.arccos( np.dot(tri0, tri1) / norm(tri0) / norm(tri1)) )

    def _get_tri_vec(self, coord, tri_idx):
        com = get_com(coord, self.mass_lt[tri_idx])
        return coord[0] - com

    def _detect_variant(self, ref, timestep, interval, angle_threshold=50, Round=10):
        """
        return
        ---------------
        return the variant-constant blocks

        arguments
        ---------------
        ref = [AtomIdx0, AtomIdx1]  which Atom-Atom Idx of reference to detect insymmetric rotation
        timestep :: ts in fs
        self.lag :: lag time is ps
        within the `tol`(degree) we consider no insymmetric rotation occurs
        """
        print('detecting rotation...')

        # lag(ps) to lag(step)
        lag_step = math.ceil(self.lag*1000/timestep)
        ref = list(ref)
        # for starter we set st = 0
        lt = [0,]
        st = 0
        tailC = self[st].coords
        refV0 = tailC[ref][0] - tailC[ref][1]
        l0 = norm(refV0)
        for i in range(0, self.FrameNum-lag_step, interval):  # last `lag-lengthed` frame is discarded
            # TODO accumulate triangle while detecting the hopping point
            # 1. determine hopping or not
            headC = self[i].coords
            refV1 = headC[ref][0] - headC[ref][1]
            angle = np.rad2deg(np.arccos( np.dot(refV0, refV1) / l0 / norm(refV1) ))
            #print(i*timestep/1e3, angle)
            #lt.append(angle)
            if angle > angle_threshold:
                # trigger second validation
                averV = self._aver(ref, i, timestep)
                if np.rad2deg( np.arccos( np.dot(refV0, averV) / l0 / norm(averV) )) > angle_threshold:
                # change the reference
                    st = i
                    #tailC  =self[st].coords
                    #refV0 = tailC[ref][0] - tailC[ref][1]
                    refV0 = averV
                    l0 = norm(refV0)
                    lt.append(i)
            #self._tri_rot()
        self._gen_blk(lt, timestep, Round)

    def detect_rot(self, ref, timestep, interval, angle_threshold=50, Round=10):
        """
        return
        ---------------
        return the variant-constant blocks

        arguments
        ---------------
        ref = [AtomIdx0, AtomIdx1]  which Atom-Atom Idx of reference to detect insymmetric rotation
        timestep :: ts in fs
        self.lag :: lag time is ps
        within the `tol`(degree) we consider no insymmetric rotation occurs
        """
        print('detecting rotation...')

        # lag(ps) to lag(step)
        lag_step = math.ceil(self.lag*1000/timestep)
        ref = list(ref)
        # for starter we set st = 0
        lt = [0,]
        st = 0
        tailC = self[st].coords
        refV0 = tailC[ref][0] - tailC[ref][1]
        l0 = norm(refV0)
        for i in range(0, self.FrameNum-lag_step, interval):  # last `lag-lengthed` frame is discarded
            # TODO accumulate triangle while detecting the hopping point
            # 1. determine hopping or not
            headC = self[i].coords
            refV1 = headC[ref][0] - headC[ref][1]
            angle = np.rad2deg(np.arccos( np.dot(refV0, refV1) / l0 / norm(refV1) ))
            #print(i*timestep/1e3, angle)
            #lt.append(angle)
            if angle > angle_threshold:
                # trigger second validation
                averV = self._aver(ref, i, timestep)
                if np.rad2deg( np.arccos( np.dot(refV0, averV) / l0 / norm(averV) )) > angle_threshold:
                # change the reference
                    st = i
                    #tailC  =self[st].coords
                    #refV0 = tailC[ref][0] - tailC[ref][1]
                    refV0 = averV
                    l0 = norm(refV0)
                    lt.append(i)
            #self._tri_rot()
        self._gen_blk(lt, timestep, Round)
    def _gen_blk(self, lt, timestep, Round=10):
        dual = []
        Round *= 1e3  # round for .5 ps by default
        Round /= timestep
        Round = int(Round)

        if len(lt) == 1:
            self.dual = [[0,self.FrameNum],]
            return 
        dual.append([0, lt[1]-Round])
        for i in range(1,len(lt)-1):
            dual.append([lt[i]+Round,  lt[i+1]-Round])

        self.dual = dual
        
    def _aver(self, ref, i, timestep) -> bool:
        """ return averaged bond vector. """
        lt = []
        step=int(self.lag*1e3/timestep)  # lag for 20ps by default
        for j in range(i, int(i+step)):
            C = self[j].coords[ref]
            refV1 = C[0] - C[1]
            lt.append(refV1)

        averV = np.average(np.array(lt), axis=0)
        return averV

    def get_dihedral(self, idx):
        #??? Incomplete
        vec10 = self.coords[idx[1]] - self.coords[idx[0]]
        vec12 = self.coords[idx[1]] - self.coords[idx[2]]
        vec23 = self.coords[idx[2]] - self.coords[idx[3]]
        vec21 = -vec12

        v1=np.cross(vec10, vec12)
        v2=np.cross(vec23, vec21)

    def detect_invariant_mat1(self, tri_idx, ref):
        """new com"""
        tri_idx, ref = list(tri_idx), list(ref)

        refC0 = self.coords[ref] 
        BV0 = refC0[0] - refC0[1]
        BV0 = BV0 / norm(BV0)

        #com0 = get_com(self.coords[tri_idx], self.mass_lt[tri_idx])
        com0 = self.coords[ref[0]]
        tri0 = self.coords[tri_idx[0]] - com0
        tri0 = tri0 / norm(tri0)

        #rigid_point0 = self.coords[tri_idx[0]]

        
        #next(self)
        lt = []
        #lt1=[]
        #lt2=[]
        #lt3=[]
        # EACH FRAME
        # build vector from ref
        for f in self:
            BV = self.coords[ref[0]] - self.coords[ref[1]]
            RM = rot_mat(BV0, BV/norm(BV))
            #tri_rigid = RM@tri0  # the supposed-to-be vector under rigid condition
            tri_rigid= RM@tri0 # the supposed-to-be vector under rigid condition

            #tri_rigid = np.matmul(RM, tri0.reshape((3,1)))# the supposed-to-be vector under rigid condition
            #print(tri_rigid)
            #com = get_com(self.coords[tri_idx], self.mass_lt[tri_idx])
            com = self.coords[ref[0]]
            #tri_rigid = rigid_point - com
            
            tri_real = self.coords[tri_idx[0]] - com
            

            #print(com)
            #angle = np.rad2deg( np.arccos(np.dot(tri_rigid, tri_real) 
            cos = np.dot(tri_rigid, tri_real) / norm(tri_real) / norm(tri_rigid)
            angle = np.rad2deg( np.arccos( np.round(cos,5)) )
            #angle = np.rad2deg( np.arccos(np.dot(tri_rigid, tri_real) / norm(tri_real) / norm(tri_rigid)) )
            lt.append(angle)
            #lt1.append(tri_real)
            #lt2.append(tri_rigid)
            #lt3.append(BV)
            #lt.append(angle)

        return np.array(lt)
        #return np.array(lt1), np.array(lt2), np.array(lt3)

    def detect_invariant_mat2(self, tri_idx, ref):
        """old com"""
        
        tri_idx, ref = list(tri_idx), list(ref)

        refC0 = self.coords[ref] 
        BV0 = refC0[0] - refC0[1]
        BV0 = BV0 / norm(BV0)

        com0 = get_com(self.coords[tri_idx], self.mass_lt[tri_idx])
        tri0 = self.coords[tri_idx[0]] - com0

    def r_dist_cc(self):
        pass
        tri0 = tri0 / norm(tri0)

        rigid_point0 = self.coords[tri_idx[0]]

        
        #next(self)
        lt = []
        #lt1=[]
        #lt2=[]
        #lt3=[]
        # EACH FRAME
        # build vector from ref
        for f in self:
            BV = self.coords[ref[0]] - self.coords[ref[1]]
            RM = rot_mat(BV0, BV/norm(BV))
            #tri_rigid = RM@tri0  # the supposed-to-be vector under rigid condition
            rigid_point = RM@rigid_point0 # the supposed-to-be vector under rigid condition
            #tri_rigid = np.matmul(RM, tri0.reshape((3,1)))# the supposed-to-be vector under rigid condition
            #print(tri_rigid)
            com = get_com(self.coords[tri_idx], self.mass_lt[tri_idx])
            tri_rigid = rigid_point - com
            
            tri_real = self.coords[tri_idx[0]] - com
            

            #print(com)
            #angle = np.rad2deg( np.arccos(np.dot(tri_rigid, tri_real) 
            cos = np.dot(tri_rigid, tri_real) / norm(tri_real) / norm(tri_rigid)
            angle = np.rad2deg( np.arccos( np.round(cos,5)) )
            #angle = np.rad2deg( np.arccos(np.dot(tri_rigid, tri_real) / norm(tri_real) / norm(tri_rigid)) )
            lt.append(angle)
            #lt1.append(tri_real)
            #lt2.append(tri_rigid)
            #lt3.append(BV)
            #lt.append(angle)

        return np.array(lt)
        #return np.array(lt1), np.array(lt2), np.array(lt3)

    def accumulate_angle(self):
        pass

    def _get_ethane_one_degree(self, idx):
        """supply the idx in [Cidx, Cidx, Hside1_idx, Hside1_idx, Hside1_idx,
        Hside2_idx, Hside2_idx, Hside2_idx]
        order."""
        #First move to com reference
        c = self.coords[idx] - get_com(self.coords[idx], self.mass_lt[idx])
        #Second rotate about the origin to superimpose the C-C bond and c-axis
        # v0 --> [0,0,1]
        v0 = c[0] / norm(c[0])
        RM = rot_mat(v0, np.array([0.,0.,1.]))
        c0 = RM @ np.transpose(c)
        c0 = c0.T
        #Third get com for triangle
        #tri_com = get_com(c0[[2,3,4]], self.mass_lt[idx[2:5]])
        #print(tri_com)

        #Third get triangle plane and c-axis intersection point
        #NV = get_normal_vector(*c0[2:5])
        #IP = intersect_line_plane(np.array([0.,0.,0.]),np.array([0.,0.,1.]),c0[2], NV)
        
        #Third get the position after rotating 1 degree
        

        #return self.trsfm_vec

    def get_radial_distribution(self, 
                              dump_name,
                              ligand_atom, 
                              central_atom,
                              sizer=10., 
                              dr=.05,):
        #if not (central_atom in self.idx_d and ligand_atom in self.idx_d):
        #    raise IndexError(f'{central_atom} or {ligand_atom} not all in self.idx_d, plz setup first')



        dist_bin = np.zeros(int(sizer/dr)+1)
        kb=constants.physical_constants['Boltzmann constant in eV/K'][0]
        print('calculating cation-anion fel...')
        c_idxs = self.idx_d[central_atom]
        l_idxs = self.idx_d[ligand_atom]
        print('FrameNum: ', self.FrameNum)
        for i, f in enumerate(self):
            print(i)
            for c_idx in c_idxs:
                for l_idx in l_idxs:
                    r=self._get_dist_unwrap(c_idx, l_idx)
                    if r <= sizer:
                        dist_bin[int(r/dr)] += 1 / r / r
        
        dump_pkl(dump_name, dist_bin)
            

    def get_gm(self, s1, s2, Rcut, dr, fn_gm, sigma=.125):
        r"""pair entropy.
        
        arguments
        -----------
        sigma :: the boarding paramters in \AA
        """
        print('collecting the distance...')
        lt = [] # containing all the possible r` -- r histogrm
        idx_lt1 = list(self.idx_d[s1])
        idx_lt2 = list(self.idx_d[s2])
        AtomNum = len(idx_lt1) + len(idx_lt2)

        print('FrameNum: ', self.FrameNum)
        for i, f in enumerate(self):
            for idx1 in idx_lt1:
                for idx2 in idx_lt2:
                    r=self._get_dist_unwrap(idx1,idx2)
                    if r < Rcut:
                        lt.append(r)


        # C alternative?
        #for i, f in enumerate(self):
        #    dist = c_get_dist_unwrap(self.coords, idx1,idx2)
        #    lt+=dist

        # construction of the gm (a R1 function)
        print('constructing the radial distribution...')
        coeff1 = 2*sigma**2
        rho = AtomNum / np.linalg.det(self.lat_vec)
        coeff2 = (32*np.pi**3)**.5 * sigma * AtomNum * rho 

        lt = np.array(lt)
        gm = np.zeros( int(Rcut / dr) )
        R = np.arange(dr,Rcut+dr,dr)
        for i in range(len(gm)):
            head = (R[i] - lt)**2/coeff1*(-1)
            s = np.sum(np.exp(head))
            gm[i] = s / R[i]**2
            #gm[i] = s / R[i]**2
        #gm = gm/coeff2
        Plot(R, gm/np.sum(gm), fn_gm)
        
        # calculating entropy...        
        #kb=constants.physical_constants['Boltzmann constant in eV/K'][0]
        #for i in range(len(gm)):
        #    -2 * np.pi * rho * kb * (gm[i] * np.log(gm[i])  - gm[i] + 1) * R[i]
        #Sr = -2 * np.pi * rho * kb * (gm * np.log(gm)  - gm + 1.) #* R**2

        #Plot(R, Sr, fn_S)

    def get_density_distribution2d(self, atom, fnxy, fnxz, fnyz, da=.1, db=.1, dc=.1):
        """
        arguments
        ------------
        how about calculate xy,xz,yz three scenario directly?

        da, da :: a, b can be any two direction, for example, 
        can be x and y or x and z. da, db -> dx, dy or dx, dz.
        by default 0.1 angstrom.
        """
        # translate 
        print(f'calculating 2d density distribution of {atom}')
        Max, trs = np.zeros(3), np.zeros(3)
        for i in range(3):
            MAX, TRS = self._get_maxNtrs(self.lat_vec[:,i])
            trs[i] = TRS
            Max[i] = MAX

        xy = np.zeros( ( int(Max[0]/da)+1, int(Max[1]/db)+1 ) )
        xz = np.zeros( ( int(Max[0]/da)+1, int(Max[2]/dc)+1 ) )
        yz = np.zeros( ( int(Max[1]/db)+1, int(Max[2]/dc)+1 ) )
        idx_lt = self.idx_d[atom]
        for f in self:
            coords = self.coords[idx_lt] + trs
            for coord in coords:
                self._find_wrapped(coord, Max)
                xy[int(coord[0]/da), int(coord[1]/db) ] += 1
                xz[int(coord[0]/da), int(coord[2]/dc) ] += 1
                yz[int(coord[1]/db), int(coord[2]/dc) ] += 1
        # now generating the mapping X
        Xmin, Xmax = np.max(self.lat_vec[:,0]), np.min(self.lat_vec[:,0])
        Ymin, Ymax = np.max(self.lat_vec[:,1]), np.min(self.lat_vec[:,1])
        Zmin, Zmax = np.max(self.lat_vec[:,2]), np.min(self.lat_vec[:,2])

        X1,Y1 = np.meshgrid(np.linspace(Xmin,Xmax,num=xy.shape[0]),
                            np.linspace(Ymin,Ymax,num=xy.shape[1]),)
         
        X2,Y2 = np.meshgrid(np.linspace(Xmin,Xmax,num=xz.shape[0]),
                            np.linspace(Zmin,Zmax,num=xz.shape[1]),)

        X3,Y3 = np.meshgrid(np.linspace(Ymin,Ymax,num=yz.shape[0]),
                            np.linspace(Zmin,Zmax,num=yz.shape[1]),)
        PlotContour(X1.T, Y1.T, xy, fnxy)
        PlotContour(X2.T, Y2.T, xz, fnxz)
        PlotContour(X3.T, Y3.T, yz, fnyz)

    def _get_maxNtrs(self, vec):
        """first choose `most-positive` and `most-negative`
            note that only translate to positive direction 
        """
        if np.max(vec) > 0:
            most_positive = np.max(vec)
            # now inspect any negatives.
            t = 0
            for v in vec:
                if v < t:
                    t = v
            return most_positive + abs(t), abs(t)

        else:
            # all smaller than zero. just choose the `most-negative`
            return abs(np.min(vec)), abs(np.min(vec))

    def _find_wrapped(self,coord,Max):
        """actually the same as wrapping process."""
        for i in range(3):
            while coord[i] > Max[i]:
                coord[i] -= Max[i]
            while coord[i] < 0:
                coord[i] += Max[i]
    
    @classmethod
    def get_fluctuations(cls, TEMP, part=3, threshold=140):
        """ thetas.shape == (FrameNum, RotorNum). 
        each block should be as long as long possible, cannot be too short...
        """
        DIRECTIONS = ['X', 'Y', 'Z']
        lt = []
        for DIRECTION in DIRECTIONS:
            thetas = load_pkl(f'{TEMP}/theta{DIRECTION}.pkl')
            l = len(thetas) // part
            s = 0
            denom = 0
            part_lt = []
            for i in range(part):
                RotorNum = 0
                d = i * l
                u = (i+1)*l
                theta = thetas[d:u].T
                
                # for rotor in rotors
                for t in theta:
                    if max(t) - min(t) > threshold:
                        continue
                    s += np.sum((t - np.average(t)) ** 2)
                    denom += len(t)
                    RotorNum += 1
                part_lt.append(RotorNum)
            print(part_lt, sum(part_lt))
            lt.append(s/denom)
        print('Calculating RotorNum: ', theta.shape[0])
        #print(part_lt)
        #print(lt)
        return sum(lt) ** .5

    def get_one_diff_dist(self, idx, dump_name, threshold=.3, each_part=200):
        lt = []
        c0 = self.coords[idx]
        for f in self:
            c = self.coords[idx]
            if np.dot(c,c0) ** .5 > threshold:
                c0 = c
                lt.append(copy.copy(c))

        l = len(lt) 
        coords = np.zeros((self.AtomNum+l,3), dtype='<U16')
        coords[:self.AtomNum] = copy.deepcopy(self.coords)

        coords[self.AtomNum:] = lt
        #print(coords)
        self._write_to_xyz_one_frame(coords, dump_name, each_part)

    def get_one_diff(self, idx, step, dump_name, each_part=200):
        lt = list(range(1,len(self),step))
        l = len(lt)
        coords = np.zeros((self.AtomNum-1+l,3), dtype='<U16')
        coords[:self.AtomNum] = copy.deepcopy(self.coords)

        for j, i in enumerate(lt):
            coords[self.AtomNum+j-1] = self[i].coords[idx]
        self._write_to_xyz_one_frame(coords, dump_name, each_part)

    def _write_to_xyz_one_frame(self, coord, dump_name, each_part):
        """ coord == (AtomNum,3)"""
        f = open(dump_name, 'w')
        print(coord.shape[0], file=f, end='\n\n')
        for i in range(self.AtomNum):
            tmp = [self.atom_lt[i],]  + list(coord[i])
            txt = '   '.join(tmp)
            print(txt, file=f)

        # noble gas -> Fe Co Ni Cu Zn
        artificial_ele = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Hf', 'Ta', 
                          'W','Re', 'Os', 'Ir','Pt', 'Au', 'Hg']
        count = 0
        which = 0
        for i in range(self.AtomNum,len(coord)):
            if count > each_part:
                count = 0
                which += 1
            tmp = [artificial_ele[which],] + list(coord[i])
            txt = '   '.join(tmp)
            print(txt, file=f)
            count += 1
        f.close()

    #def _detect_hop(
    def _site_classification(self,symbol):
        # found cation within the first shell of itself
        pass

    def get_hist_cation(self, symbol, NumBin=20):
        idx_lt = self.idx_d[symbol]
        distances = []
        for comb in combinations(idx_lt,2):
            r = self._get_dist_unwrap(*comb)
            #r = np.dot(self.coords[comb[0]] self.coords[comb[1]]) ** .5
            #if 5<r<6.2:
            #    print(comb,self.coords[list(comb)])
            distances.append(r)
            
        ax = plt.subplot()
        ax.hist(distances, bins=NumBin)
        return ax
    
    def detect_ion_hop(self, idx, s_fac=1000, peak_height=.005, multiple=2):
        """get one ion hop trajectory."""
        # first get the ion away from its equil position
        disp = self.ion_diff(idx)
        x = np.arange(len(disp))

        #based on this displacement function(with huge noise)to get a more pure `trend` function
        spline = UnivariateSpline(x, disp, s=s_fac)
        spline_derivative = spline.derivative()
        peaks, _ = find_peaks(spline_derivative(x), height=peak_height)

        # based on peaks we 
    
    def _get_hist_self(self, symbol):
        idx_lt = self.idx_d[symbol]
        rmin = self._get_dist_unwrap(idx_lt[0], idx_lt[1])
        for comb in combinations(idx_lt, 2):
            r = self._get_dist_unwrap(*comb)
            if r < rmin:
                rmin = r
        return rmin

    def _get_wrapped(self, idx):
        """ wrapping into the lattice box. 
        idx is passed in `atom index` 
        """
        fc = self._get_fc(idx=idx)
        for i in range(len(fc)):
            if fc[i] < 0:
                fc[i] += abs(int(fc[i])) + 1
            else:
                fc[i] = fc[i] - int(fc[i])
        return self._get_cc(fc)

    def count_hops(self, cation, add_mins):
        """add_mins: means additional minimum except the equilibrium lattice site."""
        equil_sites=self.coords[cation]
        self.mins = np.array(list(equil_sites)+list(add_mins))
        cation_idx=self.idx_d[cation]
        # look for initial idx
        lt = []
        hops=np.zeros(len(cation_idx))
        for i, idx in enumerate(cation_idx):
            lt.append(self._find_nearest(idx))

        #coords0=self.coords[cation_idx]
        for fn, f in enumerate(self):
            for idx in cation_idx:
                site = self._find_nearest(idx)
                if site != lt[idx] and self._check_not_same(lt[idx],site):
                    hops[idx]+=1
                    lt[idx]=site
        print(hops)

    def _find_nearst(i):
        coord=self._get_wrapped(i)
        return np.argmin(norm(self.mins - coord, axis=1))

    def get_wrapped(self, idx):
        # denoting wrapped coordinates
        return self._get_wrapped(idx)

    def get_r_dist_cc(self, cation, dump_name, threshold=8, dr=.025):
        """
        get radial bin of mobile cation species itself. (cation-cation radial dist)
        (first need to process unwrapped cp2k coordinates into wrapped).
        """
        print('calculating cation-cation distribution function')
        idx_lt=self.idx_d[cation]
        dist_bin = np.zeros(int(threshold/dr))

        for f in self:
            for comb in combinations(idx_lt,2):
                fc0,fc1=self._get_fc(comb[0]), self._get_fc(comb[1])
                #dp=fc0-fc1
                lt = []
                for a0, a1 in zip(fc0, fc1):
                    MAX=max([a0, a1])
                    MIN=min([a0, a1])
                    if MAX-MIN > .5:
                        lt.append(MIN+1-MAX)
                    else:
                        lt.append(MAX-MIN)
                # get the distance and bin it
                r=norm(self._get_cc(np.array(lt)))
                if r >= threshold:
                    continue
                ir = int(r/dr)
                dist_bin[ir] += 1 / r / r

        dump_pkl(dump_name, dist_bin)

    def _get_frame_coordinates(self, start):
        t=self.coord_lt[0].shape[0]
        return (start//t, start%t)

    def _check_lim(self):
        a=self.R_cylinder-int(self.R_cylinder)
        b=self.r_cylinder-int(self.r_cylinder)
        d=abs(a-b)
        n=-1
        while True:
            n+=1
            if np.allclose(d, n*self._dr):
                return True
            if n*self._dr>1:
                return False

    def _get_points(self):
        #central points
        ac1=np.array([ [self.lat_vec[0]/2+self.lat_vec[1]/2, self.lat_vec[0]/2+self.lat_vec[1]/2+self.lat_vec[2] ] ])

        #up-down
        ac2=np.array([ [self.lat_vec[0]/2, self.lat_vec[0]/2+self.lat_vec[2] ],
                       [self.lat_vec[0]/2+self.lat_vec[1], self.lat_vec[0]/2+self.lat_vec[1]+self.lat_vec[2]] ])

        #left-right
        ac3=np.array([ [self.lat_vec[1]/2, self.lat_vec[1]/2+self.lat_vec[2]],
                       [self.lat_vec[0]+self.lat_vec[1]/2, self.lat_vec[0]+self.lat_vec[1]/2+self.lat_vec[2]] ])

        #4 vertex
        ac4=np.array([ [np.zeros(3),self.lat_vec[2] ],
                       [self.lat_vec[0], self.lat_vec[0]+self.lat_vec[2]],
                       [self.lat_vec[1], self.lat_vec[1]+self.lat_vec[2]],
                       [self.lat_vec[0]+self.lat_vec[1],np.sum(self.lat_vec,axis=0)]] )
        return [ac1, ac2, ac3, ac4]

    def _get_idx_per_cage(self, lt):
        #cylinder cage (fast) implementation 
        
        #first setup the cylinder
        # from lists of points to get basic cylinder info
        # points is a dictionary {'cylinder1': [[p1,p2],], 'cylinder2': [[p1,p2],[p3,p4]],..}
        # or just list of (inhomogeneuous) np.2darray
        # because cylinders at the boundary are characterized by more than one axes.
        points=self._get_points()  # fast way to construct point list, may not be correct.
                                   # in this way the V_cylinder's are all the same but their points are different
        cylinder_lt=[Cylinder(p) for p in points] # p is (axis_num, two_points_determining_axes, their components)
                                                  # OR simply (axis_num, 2, 2)

        #bound=[[],]*len(cylinder_lt) !!!CAUTION
        bound=[[],[],[],[]]
        coords=self.coord_lt[0][0]
        for atom in lt:
            d0=0
            lt=[0,0]
            for j, cylinder in enumerate(cylinder_lt):
                for i in range(cylinder.multiplicity):
                    # first frame no pbc issue
                    d = norm(np.cross(coords[atom] - cylinder.points[i], cylinder.V_cylinder[i])) / cylinder.H_cylinder[i]
                    if j==0 and i == 0:
                        d0=d
                    elif d < d0:
                        d0=d
                        lt[0],lt[1]=j,i
            #print(f'doing bound[{lt[0]}].append({atom})')
            bound[lt[0]].append(atom)
        return bound

    def get_rdf_fast_cylinder_distinct(self, cutoff, dr, lt0, lt1):
        idx_lt0= self._get_idx_per_cage(lt0)
        idx_lt1= self._get_idx_per_cage(lt1)
        return self._get_rdf_fast_cylinder_distinct(cutoff, dr, idx_lt0, idx_lt1)

    def _get_rdf_fast_cylinder_distinct(self, cutoff, dr, idx_lt0, idx_lt1):
        """
        arguments
        --------------
        `r_cylinder` :: inner radius 
        `R_cylinder` :: outer radius 
        idx_lt :: lists of number of cages idx_lt0=[[Zn_idx_in_cage1, Zn_idx_in_cage2,...,Zn_idx_in_cageN]]
                                           idx_lt1=[[S_idx_in_cage1, S_idx_in_cage2, ..., S_idx_in_cageN]]
        return
        --------------
        histogram for further manipulation
        """
        self._cutoff=cutoff
        self._dr=dr
        self._to_inspect0=idx_lt0
        self._to_inspect1=idx_lt1
        
        # self.coord_lt have already been constructed and constructed for multiprocessing
        # it's list(np.array(PartFrameNum, AtomNum,3), np.array(PartFrameNum, AtomNum, 3), ..)
        # it's length is number of cores allocated.
        print(f'n_processes: {self.n_processes}')
        print(f'n_chunks: {len(self.coord_lt)}')
        print(f'n_FramePerChunk: {[coord.shape[0] for coord in self.coord_lt]}')
        with multiprocessing.Pool(processes=self.n_processes) as pool:
            res = pool.map(self._get_rdf_fast_periodicZ_distinct, self.coord_lt)
        new_res, self.Nref = [], 0
        for r in res:
            new_res.append(r[0])
            self.Nref += r[1]
        return new_res

    def _get_rdf_fast_periodicZ_distinct(self, coord_lt):
        # calculate RDF
        nn=0
        gr=np.zeros(int(np.round(np.round(self._cutoff,3)/self._dr,3) ) )
        for coord in coord_lt:
            for s1, s2 in zip(self._to_inspect0, self._to_inspect1):
                for i in s1:
                    for j in s2:
                        d=self._get_cylinder_dist(coord[i], coord[j])
                        if d < self._cutoff:
                            gr[int(d/self._dr)]+=1/d/d
                            nn+=1
            
        gr=gr/4/np.pi/self._dr 
        return gr, nn

    def get_rdf_fast_cylinder(self, r_cylinder, R_cylinder, dr, idx_lt, calc_den=False, calc_rdf=False, wrap=False):
        """
        arguments
        --------------
        `r_cylinder` :: inner radius 
        `R_cylinder` :: outer radius 
        idx_lt :: atom indices of which you want to inspect in terms of its position and the defined cylinder

        return
        --------------
        histogram for further manipulation
        """
        assert (calc_den != calc_rdf) # calculation one of them
        self._calc_den=calc_den
        self._calc_rdf = calc_rdf

        # first setup the new cylinder axes based on c_vec in lattice vector
        upperC=self.lat_vec[2]+ self.lat_vec[0]+self.lat_vec[2]+\
               self.lat_vec[1]+self.lat_vec[2]+ np.sum(self.lat_vec, axis=0)
        upperC/=4
        lowerC=self.lat_vec[0]+self.lat_vec[1]
        lowerC/=2
        self.axis = (upperC, lowerC)
        self._to_inspect = list(sorted(idx_lt))
        self.r_cylinder=r_cylinder # inner cylinder
        self.R_cylinder=R_cylinder # outter cylinder
        self._cutoff=R_cylinder-r_cylinder
        self._dr=dr
        self._cylinder_wrap=wrap
        assert self._check_lim()
        
        # self.coord_lt have already been constructed and constructed for multiprocessing
        # it's list(np.array(PartFrameNum, AtomNum,3), np.array(PartFrameNum, AtomNum, 3), ..)
        # it's length is number of cores allocated.
        print(f'n_processes: {self.n_processes}')
        print(f'n_chunks: {len(self.coord_lt)}')
        print(f'n_FramePerChunk: {[coord.shape[0] for coord in self.coord_lt]}')
        self.V_cylinder = self.axis[0] - self.axis[1]
        self.H_cylinder = norm(self.V_cylinder)
        with multiprocessing.Pool(processes=self.n_processes) as pool:
            res = pool.map(self._get_rdf_fast_periodicZ, self.coord_lt)
        if self._calc_rdf:
            new_res, self.Nref = [], 0
            for r in res:
                new_res.append(r[0])
                self.Nref += r[1]
            return new_res
        return res

    def _get_rdf_fast_periodicZ(self, coord_lt):
        # aiming for obtain the water numbers within r_cylinder~R_cylinder
        if self._calc_den:
            #gr=np.zeros(int(np.ceil(np.round(self._cutoff,3)/self._dr))) # new
            gr=np.zeros(int(np.round(np.round(self._cutoff,3)/self._dr,3) ) )
            if self._cylinder_wrap:
                for coord in coord_lt:
                    fcs=np.matmul(coord[self._to_inspect], self.ilat_vec)
                    # do wrap
                    fcs=fcs-np.array(fcs,dtype=np.int32)
                    mask=fcs<0
                    fcs[mask]+=1
                    ccs=np.matmul(fcs, self.lat_vec)
                    for cc in ccs:
                        d = norm(np.cross(cc - self.axis[0], self.V_cylinder)) / self.H_cylinder
                        if self.r_cylinder<=d < self.R_cylinder:
                            to_add=1/d
                            gr[int((d-self.r_cylinder)/self._dr)]+=to_add
            else:
                #gr=np.zeros(len(np.arange(self.r_cylinder, self.R_cylinder, self._dr))-1)
                for coord in coord_lt:
                    for atom in self._to_inspect:
                        d = norm(np.cross(coord[atom] - self.axis[0], self.V_cylinder)) / self.H_cylinder
                        if self.r_cylinder<=d < self.R_cylinder:
                            to_add=1/d
                            gr[int((d-self.r_cylinder)/self._dr)]+=to_add
            gr=gr/self._dr/2/np.pi/self.H_cylinder/self.EquilFrameNum
            return gr

        # calculate RDF
        else:
            nn=0
            gr=np.zeros(int(np.round(np.round(self._cutoff,3)/self._dr,3) ) )
            for coord in coord_lt:
                lt, ltt = [], []
                for atom in self._to_inspect:
                    d = norm(np.cross(coord[atom] - self.axis[0], self.V_cylinder)) / self.H_cylinder
                    if d <= self.r_cylinder:
                        lt.append(atom)
                    elif d<self.R_cylinder:
                        ltt.append(atom)
                ltt = ltt + lt
                for i in lt:
                    for j in ltt:
                        if j in lt and j <= i:
                            continue
                        d=self._get_cylinder_dist(coord[i], coord[j])
                        if d < self._cutoff:
                            gr[int(d/self._dr)]+=1/d/d
                nn+=len(lt)
                
            gr=gr/4/np.pi/self._dr # the plane density of the spherical shell
            return gr, nn

    def _get_cylinder_dist(self,c1,c2):
        #did not need to wrap about x- and y-axis anyway.
        l=self._get_fc(c=c1)
        l[-1] = l[-1] - int(l[-1])
        m=self._get_fc(c=c2)
        m[-1] = m[-1] - int(m[-1])
        if l[-1] - m[-1] > .5:
            m[-1] += 1
        elif l[-1] - m[-1] < -.5:
            l[-1] += 1
        return norm(self._get_cc(l-m))

    def get_rdf_fast_periodic(self, cutoff, dr, idx_lt):
        """
        For now the implementation is between the same element, e.g., Ow-Ow.
        Note that the the pair should not be calculated twice.

        arguments
        ----------------------
        cutoff :: to which extent(distance) you calculate the RDF
        idx_lt :: from which atoms you will be using to construct the RDF
        atom :: convienient way to construct atom idx_lt

        """
        self._cutoff = cutoff
        self._dr = dr
        self._pair_lt = [(i,j) for i in idx_lt for j in idx_lt if j > i]
        print('calculating RDF...')
        print(f'Maximum FrameNum: {self.coord_lt[1].shape[0]}')
        # now self.coord_lt has been trimmed and we can directly pass it to `chunk-processing` function.
        with multiprocessing.Pool(processes=self.n_processes) as pool:
            res = pool.map(self._get_rdf_fast_low, self.coord_lt)
        return res

    def _get_rdf_fast_low(self, coord_lt):
        #gr=np.zeros(int(np.ceil(self._cutoff/self._dr)))
        gr=np.zeros(int(np.round(np.round(self._cutoff,3)/self._dr,3) ) )
        assert all(self.lat_param[:3]/2 > self._cutoff)

        for i, coord in enumerate(coord_lt):
            if not (i % 10):
                print(i)
            for p in self._pair_lt:
                d=self._get_dist_unwrap(cc=coord[p[0]], lc=coord[p[1]])
                if d < self._cutoff:
                    gr[int(d/self._dr)] += 1/d/d
        #vol=np.linalg.det(self.lat_vec)
        return gr/4/np.pi/self._dr/self.EquilFrameNum
        #return gr/vol/(len(idx_lt)**2)

    def get_short_dist_cages(self, species1, species2):
        """
        species1 : list of lists for species1
        species2 : list of lists for species2
        """
        
        assert len(species1) == len(species2)
        self.species1, self.species2 = species1, species2
        self.CageNum=len(species1)
        self.LenList=[len(s) for s in species1]
        self.accum=deque(accumulate(self.LenList))
        self.accum.appendleft(0)
        
        print('constructing minimum distance list within each cage...')
        print(f'n_processes: {self.n_processes}')
        print(f'n_chunks: {len(self.coord_lt)}')
        print(f'n_FramePerChunk: {[coord.shape[0] for coord in self.coord_lt]}')
        with multiprocessing.Pool(processes=self.n_processes) as pool:
            res = pool.map(self._get_short_dist_cages_low, self.coord_lt)
        return np.concatenate(res,axis=0) # (FrameNum, NrSpecies1Atom)

    def _get_short_dist_cages_low(self, coord_lt):
        ltt=[]
        for coord in coord_lt:
            for s1, s2 in zip(self.species1, self.species2):
                ltt.append([])
                for k in s1:
                    lt=[]
                    for l in s2:
                        d=self._get_dist_unwrap(cc=coord[k], lc=coord[l])
                        lt.append(d)
                    ltt[-1].append(min(lt))
        dists=np.empty((len(coord_lt), sum(self.LenList)))        
        for i, lt in enumerate(ltt):
            dists[i//self.CageNum,self.accum[i%self.CageNum]:self.accum[1+i%self.CageNum]] = lt

        return np.array(dists)

    def get_r_dist_cce(self, cation, dump_name, threshold=8, dr=.025):
        """
        get radial bin of mobile cation species itself. (cation-cation_equilradial dist)
        (first need to process unwrapped cp2k coordinates into wrapped).
        """
        print('calculating cation-cation distribution function')
        idx_lt=self.idx_d[cation]
        dist_bin = np.zeros(int(threshold/dr))
        FC0=np.array([self._get_fc(i) for i in idx_lt])

        for f in self:
            for fc0 in FC0:
                for i in idx_lt:
                    fc1 = self._get_fc(i)
                    for j in range(len(fc1)):
                        fc1[j] = fc1[j] - int(fc1[j])
                    lt=[]
                    for a0, a1 in zip(fc0, fc1):
                        MAX=max([a0, a1])
                        MIN=min([a0, a1])
                        if MAX-MIN > .5:
                            lt.append(MIN+1-MAX)
                        else:
                            lt.append(MAX-MIN)
                    # get the distance and bin it
                    r=norm(self._get_cc(np.array(lt)))
                    if r >= threshold:
                        continue
                    ir = int(r/dr)
                    dist_bin[ir] += 1 / r / r
        dump_pkl(dump_name, dist_bin)



class PlotContour:
    def __init__(self, x, y, z, dump_name):
        self.x = x
        self.y = y
        self.z = z
        dump_pkl(dump_name, self)

class Radius:
    def __init__(self, dist_bin, fel, central_atom, ligand_atom, T, threshold, sizer, dr, first_minimum, sigma, dump_name):
        self.dist_bin = dist_bin
        self.fel = fel
        self.central_atom = central_atom
        self.ligand_atom = ligand_atom
        self.T = f'{T}K'
        self.threshold = threshold
        self.sizer=sizer
        self.dr = dr
        self.first_minimum=first_minimum
        self.sigma = sigma
        dump_pkl(dump_name, self)

class Angle:
    def __init__(self, dth, dphi, theta, phi, group_names, include, central_atom, sigma, DictDen, DictFes, T, dump_name):
        self.T = f'{T}K'
        self.dth=dth
        self.dphi=dphi
        self.central_atom = central_atom
        self.include = include
        self.sigma=sigma
        self.DictDen = DictDen
        self.DictFes = DictFes
        self.group_names = group_names
        d, u = 0, 0

        self.theta, self.phi = {}, {}
        for i, gn in enumerate(group_names):
            d=u
            u+=len(include[i])
            self.theta[gn] = theta[:,d:u].T  # [theta1_Frame1, theta1_Frame2, ...]
            self.phi[gn] = phi[:,d:u].T
                
        dump_pkl(dump_name, self)


def vec_to_param(lat_vec):
    lat_param = np.zeros(6)
    for i in range(3):
        lat_param[i] = np.dot(lat_vec[i], lat_vec[i]) ** .5

    comb = itertools.combinations([0,1,2],2)
    for c in comb:
        cos = np.dot(lat_vec[c[0]], lat_vec[c[1]]) / norm(lat_vec[c[0]]) / norm(lat_vec[c[1]]) 
        lat_param[-sum(c)] = np.rad2deg(np.arccos(cos))
    return lat_param

    #lat_param[-1] = np.arccos( np.dot(lat_vec[0], lat_vec[1]) / norm(lat_vec[0]) / norm(lat_vec[1]) )
    #lat_param[-2] = np.arccos( np.dot(lat_vec[0], lat_vec[2]) / norm(lat_vec[0]) / norm(lat_vec[2]) )
    #lat_param[-3] = np.arccos( np.dot(lat_vec[1], lat_vec[2]) / norm(lat_vec[1]) / norm(lat_vec[2]) )
        
def param_to_vec(lat_param):
    """lat_param = [a, b, c, alpha, beta, gamma]"""
    cos_alpha=np.cos(np.deg2rad(lat_param[3]))
    cos_beta=np.cos(np.deg2rad(lat_param[4]))
    cos_gamma=np.cos(np.deg2rad(lat_param[5]))
    sin_gamma=np.sin(np.deg2rad(lat_param[5]))
    a = lat_param[0]
    b = lat_param[1]
    c = lat_param[2]

    lat_vec = np.zeros((3,3))
    lat_vec[0][0] = a
    lat_vec[1][0] = b * cos_gamma
    lat_vec[1][1] = b * sin_gamma
    lat_vec[2][0] = c * cos_beta
    lat_vec[2][1] = c * (cos_alpha - cos_gamma * cos_beta) / sin_gamma
    s = (cos_alpha - cos_gamma * cos_beta) ** 2 / sin_gamma ** 2
    lat_vec[2][2] = c * (1. - cos_beta**2 - s) ** .5
    return lat_vec

def sign(num):
    if num == 0:
        return 0
    elif num > 0:
        return 1
    return -1

def get_normal_vector(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)

    return normal

def intersect_line_plane(line_point, line_direction, plane_point, plane_normal):
    # Check if line and plane are parallel
    if np.dot(line_direction, plane_normal) == 0:
        return None

    # Calculate the parameter t
    t = np.dot(plane_point - line_point, plane_normal) / np.dot(line_direction, plane_normal)

    # Calculate the intersection point
    intersection_point = line_point + t * line_direction

    return intersection_point
