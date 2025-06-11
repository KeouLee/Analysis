import numpy as np
import scipy
import math
from pandas import DataFrame
import sys
from ..utils.wrapper_pickle import load_pkl, dump_pkl
#import matplotlib.pyplot as plt
#from matplotlib.ticker import MultipleLocator
from pathlib import Path
from numpy.linalg import norm

np.set_printoptions(threshold=sys.maxsize)
class RotAna(XYZL):
        #         dth=2., 
        #         dphi=4., 
        #         sizer=13., 
        #         n_sigma=4,
        #         nbins_theta=90, 
        #         nbins_phi=90, 
        #         nbins_r=130,):
        #if calc_den_angle:
        #    self.dth, self.dphi = dth, dphi
        #    self.nbins_theta, self.nbins_phi = nbins_theta, nbins_phi
        #    self.calc_den_angle=True
        #    return 

        #if calc_rbin:
        #    try:
        #        self.edges=edges
        #    except:
        #        raise TypeError('edges(3-lengthed tuple)  must be provided to calculate the dist_bins_distribution')
        #    self.sizer=sizer
        #    self.nbins_r=nbins_r
        #    self.dr=sizer/nbins_r
        #    self.n_sigma=n_sigma
        #    self.sigma_r=self.dr*self.n_sigma
        #    self.calc_rbin=True
        #    return

        #self.temperature = traj.temperature
        #self.run_type = traj.run_type
        #self.name = traj.name
        #self.traj = traj

    def calc_bin_distance(self, central_atom, ligand_atom, sizer, density_filename, fe_filenamedr=.1):
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
        # density 
        print('calculating distance density distribution...')
        nbins = int(sizer/dr) # if dr=.1 then we have for instance 12.5xxx, and 125 nbins will be set up.
        dist_bin = np.zeros(nbins)
        ci_lt, li_lt =self.idx_d[central_atom], self.idx_d[ligand_atom]
        for i, f in enumerate(self):
        #for f in traj:
            if A % 5000 == 0:
                print(f'calculating {FN}th Frame')
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
        dump_pkl(dist_bin, dist_bin)

        # free energy
        print('calculating free energy curve of energy...')
        # smooth the density for free energy calculation
        ?

        FES=[]
        for v in dist_bin:
            fes=-T*np.log(b) * constants.physical_constants['Boltzmann constant in eV/K'][0]
            FES.append(fes)
        dump_pkl(fe_filename, np.array(FES))

    def _get_angle_crystal(self,step):
        if self.angle_type == 'theta':
            return [get_central_theta(self.traj.coords[mol_idx], step) for mol_idx in self.sel_idx]
        return [get_central_phi(self.traj.coords[mol_idx], step) for mol_idx in self.sel_idx]

    def get_angles(self, sel_idx, angle_type):
        """rot = Rotate(<Analysis.FormatBase object>)
        rot.get_phis([[center_atom_idx, bonding_atom1_idx, bonding_atom2_idx, ...],
                                   [center_atom_idx, bonding_atom1_idx, bonding_atom2_idx, ...],
                                   ...,])
        """
        self.angle_type = angle_type
        self.fancy_io()
        self.sel_idx = sel_idx
        self.chk_dim()
        angles = np.array([self._get_angle_crystal(i) for i, frame in enumerate(self.traj)])
        return angles

    def fancy_io(self):
        print(f"RunSystem: {self.name}  RunType: {self.traj.run_type}  Temperature: {self.traj.temperature}K\n")
        print(f"calculating the {self.angle_type}...")
        #file = f"/home/keli/solid_state/battery/TRAJ/MD_{self.temperature}K_{self.run_type}/concatenate/coords.xyz"

    def chk_dim(self):
        sp = np.array(self.sel_idx).shape
        if len(sp) != 2:
            raise ValueError("must specify 2d selection list!")

    @classmethod
    def calc_den_angle(cls, theta_pkl, phi_pkl, dump_=True ):
        """
        calculate density distribution and free energy surface from 
        dumped post-processed(by gaussian) data.

        arguments
        ----------
        `theta_pkl`, `phi_pkl` : .pkl file that stores the every-frame angle
        unserialized data should be of shape (FrameNum, 1, AngleNum)

        `dump` dump the calculated gthphi and free energy surface ::Always Dump::

        """
        # basic setup
        obj = cls(calc_den_angle=True)
        T, run_type = name_parser(theta_pkl)
        theta, phi = load_pkl(theta_pkl),  load_pkl(phi_pkl)

        histogram_angle = np.zeros((obj.nbins_theta, obj.nbins_phi))
        gthphi = np.zeros((obj.nbins_theta+1, obj.nbins_phi+1)) # gthphi[0..90]
        obj.FrameNum = theta.shape[0]
        obj.AngleNum = theta.shape[-1]

        theta = theta.reshape((obj.FrameNum, obj.AngleNum)).T # [[theta0_frame0, theta0_frame1, theta0_frame2, ...]
                                                      #  [theta1_frame0, theta1_frame1, theta1_frame2, ...]
                                                      #  [theta2_frame0, theta2_frame1, ...
                                                      #  [theta3_frame0, ...] ]
        phi = phi.reshape((obj.FrameNum, obj.AngleNum)).T # [[phi0_frame0, phi0_frame1, phi0_frame2, ...]
                                                      #  [phi1_frame0, phi1_frame1, phi1_frame2, ...]
                                                      #  [phi2_frame0, phi2_frame1, ...
                                                      #  [phi3_frame0, ...] ]
                
        for i in range(obj.AngleNum):
            for t, p in zip(theta[i], phi[i]):
                ith = int(t/obj.dth)
                iphi = int(p/obj.dphi)
                if t == 180:
                    ith = int((t- 1e-4)/obj.dth)
                if p == 360:
                    iphi = int((p - 1e-4)/obj.dphi)
                histogram_angle[ith, iphi] += 1

        width = 1/(2*obj.dth**2)
        widphi= 1/(2*obj.dphi**2)
        normfac2 = 1. / (2 * np.pi * obj.dth * obj.dphi) / obj.FrameNum / obj.AngleNum
        
        print("calculating gthphi...")
        for i in range(obj.nbins_theta):
            for j in range(obj.nbins_phi):
                for ith in range(obj.nbins_theta+1):
                    for iphi in range(obj.nbins_phi+1):
                        gthphi[ith, iphi] += histogram_angle[i,j] * \
                                np.exp(-width*(ith-(i+1)+.5)**2) * np.exp(-widphi*(iphi-(j+1)+.5)**2)
        norm_dens = 0
        for i in range(obj.nbins_theta+1):
            for j in range(obj.nbins_phi+1):
                norm_dens += normfac2 * gthphi[i,j]

        obj.gthphi = gthphi
        obj.norm_dens = norm_dens
        obj.T = T
        obj.run_type = run_type
        obj.normfac2 = normfac2
        obj._get_fes_angle()
        # note that here we dump the whole object!
        if dump_: 
            print(f'dumping GTHPHI_DB/{run_type}_{T}K_gthphi.pkl ...')
            #dump_pkl(f'GTHPHI_DB/{run_type}_{T}K_gthphi.pkl', obj)
            dump_pkl(f'{run_type}_{T}K_gthphi.pkl', obj)

    def _get_fes_angle(self):
        assert self.calc_den_angle

        evtok = 11604.588577015

        fes_container = np.zeros(self.gthphi.shape)

        for ith in range(self.nbins_theta+1):
            for iphi in range(self.nbins_phi+1):
                fes = -(self.T*np.log(self.gthphi[ith,iphi] *self.normfac2)) / evtok
                if fes > 1.85:
                    fes = 1.85
                fes_container[ith,iphi] = fes
        self.fes_angles = fes_container

    @classmethod
    def calc_bin_distance(cls, traj, edges, sel_atom, sizer=13.,dump_=True):
        # index list building
        obj = cls(edges=edges,calc_rbin=True,sizer=sizer) # obj.nbins_r obj.sizer  obj.dr
        Na_idx=obj._get_idx('Na', traj.atom_lt)

        if sel_atom == 'P': 
            P_idx=obj._get_idx('P', traj.atom_lt)
        # actually S-index
        elif sel_atom == 'S':
            P_idx=[178,192,197,183,204,187,190,201,184,189,207,202,177,198,195,180,206,188,203,185,199,176,181,194,193,179,196,182,205,186,191,200]
        obj.NaNum=len(Na_idx)
        obj.PNum=len(P_idx)
        dist_bin = np.zeros(obj.nbins_r)
        count = 0
        FN=0
        obj.temperature = traj.temperature
        obj.FrameNum = traj.FrameNum

        print(f'system name: {traj.name}  temperature: {traj.temperature}K  run_type: {traj.run_type}')
        print('calculating...')
        for f in traj:
            FN+=1
            if FN % 1000 == 0:
                print(f'calculating {FN}th Frame')
            for P in P_idx:
                for Na in Na_idx:
                    vec = traj.coords[Na] - traj.coords[P] # P -> Na
                    for i in range(3):
                        if vec[i] > obj.edges[i]*.5:
                            vec[i]-=obj.edges[i]
                        elif vec[i] < -obj.edges[i]*.5:
                            vec[i]+=obj.edges[i]
                    r=norm(vec)
                    if r >= sizer:
                        continue
                    count+=1
                    ir = r/obj.dr
                    dist_bin[int(ir)] += 1

        # radial processing
        for i in range(len(dist_bin)):
            r=obj.dr * (i+.5)
            dist_bin[i] = dist_bin[i] / r / r

        # normalize
        dist_bin = dist_bin / obj.FrameNum / obj.NaNum / obj.PNum

        obj.dist_bin = dist_bin
        obj.count = count
        #obj.normfac_fes=1/count
        #print(dist_bin)
        if dump_:
            if obj.PNum == 32:
                dump_pkl(f'dist_bin_S.pkl', obj.dist_bin)
            elif obj.PNum == 8:
                #dump_pkl(f'{traj.run_type}_{traj.temperature}_dist_bin_P.pkl', obj.dist_bin)
                dump_pkl(f'dist_bin_P.pkl', obj.dist_bin)
        return obj

    def calc_den_distance(self, sizer=15., _dump=True,):
        wid_r=1/(2*self.sigma_r**2)
        self.normfac4=1/( (2.*np.pi)**.5 * self.sigma_r) / self.FrameNum / self.NaNum / self.PNum
        g_rPNa=np.zeros(self.nbins_r+1)
        for i in range(self.nbins_r):
            cp=(i+.5)*self.dr
            for ir in range(self.nbins_r+1):
                bp=ir*self.dr
                g_rPNa[ir]+=self.dist_bin[i] * np.exp(-wid_r*(bp-cp)**2)
        self.g_rPNa = g_rPNa
        
        if _dump:
            if self.PNum == 32:
                dump_pkl(f'dist_density_S.pkl', self)
            elif self.PNum == 8:
                dump_pkl(f'dist_density_P.pkl', self)

    def calc_fes_distance_den(self, sizer=15.):
        fes=np.zeros(self.nbins_r+1)
        for i in range(len(fes)):
            fes[i]=-self.temperature*np.log(self.g_rPNa[i]*self.normfac4)
        return fes/evtok

    def _get_idx(self, symbol, atom_lt):
        return [i for i, s in enumerate(atom_lt) if symbol == s]
             
    
def dump_fes_xlsx(temperature, run_type, bth=None, eth=None, bphi=None, ephi=None, test=False):
    """fes[bth:eth,bphi:ephi]"""
    lt = []
    part=True
    if bth is None:
        bth=0
        bphi=0
        eth=91
        ephi=91
        part=False
    try:
        if test:
            rot = load_pkl(f'{run_type}_{temperature}.0K_gthphi.pkl')
        else:
            rot = load_pkl(f'GTHPHI_DB/{run_type}_{temperature}.0K_gthphi.pkl')
    except:
        raise RuntimeError ('please calulate the gthphi and FES first and dump it to `GTHPHI_DB` file then'
                'use this method to dump part of the FES to xlsx')
    real_fes = rot.fes_angles[bth:eth, bphi:ephi]
    for i in range(real_fes.shape[0]):
        for j in range(real_fes.shape[1]):
            lt.append((i,j,real_fes[i,j]))

    df = DataFrame(lt)
    if test:
        df.to_excel(f"fes_sheet/later_{temperature}K_{run_type}_thphi_fes.xlsx", sheet_name="Sheet1")
    elif part:
        df.to_excel(f"fes_sheet/{temperature}K_{run_type}_part_thphi_fes.xlsx", sheet_name="Sheet1")
    else:
        df.to_excel(f"fes_sheet/{temperature}K_{run_type}_thphi_fes.xlsx", sheet_name="Sheet1")
        
def name_parser(name):
    lt = name.strip().split('_')
    T = float(lt[-2][:-1])
    
    run_type = lt[-1].split('.')[0]
    print(f'{T}K')
    print(run_type)

    return T, run_type

def get_central_theta(coords, step,):
    """[center_atom_coords, bonding_atom1_coords, bonding_atom2_coords, ...]"""
    theta_lt = list()
    central_atom = coords[0]
    bound_atom = coords[1:]
    for b in bound_atom:
        vec = get_vec(central_atom, b, step)
        cos_theta = np.dot(vec, np.array([0.,0.,1.])) / norm(vec)
        theta_lt.append(np.rad2deg(np.arccos(cos_theta)))
    return theta_lt

def get_central_phi(coords, step, ):
    """[center_atom_coords, bonding_atom1_coords, ...]"""
    phi_lt = list()
    central_atom = coords[0]
    bound_atom = coords[1:]
    for b in bound_atom:
        vec = get_vec(central_atom, b, step)
        vec_2d = vec[:2]
        phi=match_it(vec_2d)
        phi_lt.append(phi)
    return phi_lt

def match_it(vec_2d):
    #x, y = np.dot(vec_2d, [1,0]), np.dot(vec_2d, [0,-1])
    x, y = vec_2d[0]+1e-4, vec_2d[1]+1e-4
    bond_len = norm(vec_2d)
    match bool(x>0), bool(y>0):
        case (True, True) | (False, True):
            return np.rad2deg(np.arccos(np.dot(vec_2d, [1,0]) / bond_len))
        case True, False:
            return 360 - np.rad2deg(np.arccos(np.dot(vec_2d, [1,0]) / bond_len))
        case False, False:
            return 180 + np.rad2deg(np.arccos(np.dot(vec_2d, [-1,0]) /bond_len))

def get_vec(central_atom, b, step, len_threshold=3.0):
    vec = central_atom - b
    bond_len = norm(vec)
    if bond_len > len_threshold:
        print(f"bond length = {bond_len} at frame{step}")
        raise ValueError(f"central atom and atom_idx too distant to form a bond, check you've translated the atom already or not!")
    return vec

def plot(angles, t=None, T=None, time_ps=None, FrameNum=None, angle_type=None):
    chk_angle_type(angle_type)
    angles = np.array(angles) 
    print(f"plotting {angle_type}...")
    ax = plt.subplot(111)
    x = np.linspace(0, time_ps, FrameNum)

    color = ["red", "purple", "green", "blue"]
    # [ [ Angle0Frame0, Angle0Frame1, Angle0Frame2, ...] 
    #  [ Angle1Frame0, Angle1Frame1, Angle1Frame2, ...] ...]
    angles = angles.reshape(angles.shape[0], angles.shape[-1]).T  
    for i in range(angles.shape[0]):
        if angle_type == 'phi':
            #if i == 3:
            #    print(angles[i])
            angles[i] = get_flipped_phi(angles[i].copy(), i)   # one of the calculated angles.

            #if i == 3:
            #    print("###########")
            #    print(angles[i])
            #print(angles[i][[9330, 9331, 9419, 9420, 68971, 68972, 69043, 69044, 73598, 73599, 73626, 73627]])
        ax.plot(x, angles[i], color=color[i], label=f"S_atom {203+i}")

    ax.legend(bbox_to_anchor=(0, 1.02, 1, .102), loc="lower left", ncols=4, mode="expand", borderaxespad=0.,
            fontsize="small")
    ax.set_ylabel(rf"$\{angle_type} (\degree)$ ", fontsize=15, labelpad=7)
    ax.set_xlabel(r"Time (ps)", fontsize=15, labelpad=5)
    plt.text(.01, .99, f'{T}K/{t}', ha='left', va='top', transform=ax.transAxes)
    #plt.show()
    plt.savefig(f'pics/{T}K_{t}_{angle_type}_{time_ps}ps.png',dpi=1200)
    plt.clf()

def get_flipped_phi(one_phi, i):
    """one_phi -> one phi corresponds to a PS bond in multiple frames."""
    diff = one_phi[1:] - one_phi[:-1]
    lt = [j for j, d in enumerate(diff) if abs(d) > 325]
    if (len(lt) == 0):
        return one_phi
    grouped_lt = groupit(lt)
    return flip(one_phi, grouped_lt, i)

def flip(one_phi, grouped_lt, i):
    for idx_group in grouped_lt:
        if upper_half(one_phi, idx_group[-1]):
            one_phi = flip_it(one_phi,idx_group,1)
        else:
            one_phi = flip_it(one_phi,idx_group,0)
    return one_phi

def flip_it(one_phi, idx_group, flag):
    last_idx = idx_group[-1] 
    fst_idx = idx_group[0]
    for i in range(fst_idx, last_idx+1):
        if flag:
            if one_phi[i] < 180:
                one_phi[i] += 360.
        else:
            if one_phi[i] >= 180:
                one_phi[i] -= 360
    return one_phi

def upper_half(col, idx):
    if (np.average(col[idx:idx+650]) >= 180):
        return True
    return False

def groupit(lt):
    grouped_lt = [[lt[0]]]
    for i in range(len(lt)-1):
        if lt[i+1] - lt[i] > 700:
            grouped_lt.append([])
        grouped_lt[-1].append(lt[i+1])
    return grouped_lt

def chk_angle_type(angle_type):
    if (angle_type != 'theta' and angle_type != 'phi'):
        raise ValueError ('theta or phi angle type expected!')

