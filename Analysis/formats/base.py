#import pathlib
from typing import Sequence
import numpy as np
from . import func_space as fs
from abc import ABCMeta, abstractmethod
import zipfile
from io import TextIOWrapper
import numbers
#from typing import TYPE_CHECKING
import _io
from collections import defaultdict
import math
from pathlib import Path
from ..data import *
from ..utils.cat import get_com, same_cat_pair
from ..utils.cConv import fc_to_cc, cc_to_fc
import mmap, time


class FormatBase(metaclass=ABCMeta):
    __supported = ( "xyz", "xyzl", "POSCAR")
    axes_dict = {'x': 0, 'y': 1, 'z': 2}

    def __init__(self, 
                 path=None, 
                 temperature=None,
                 run_type=None,
                 system_name=None,
                 flag=False, 
                 slicing=None, 
                 from_stream=False,
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
                 ):
        """
        arguments
        ------------
        path :: trajectory file path
        temperature :: system temperature if it is a canonical ensemble
        run_type :: unconstrained or rotc or rot_trsc
        system_name :: 'sodium' for Na11Sn2PS4 'lithium' for LiBH4
        flag :: internal parameter controlling trajectory reading
        slicing :: internal parameter controlling trajectory slicing
        from_stream :: internal parameter controllinng instantiation
        AtomNum :: number of system atom
        FrameNum :: Frame Number
        coords :: coordinates of the atom
        atom_lt :: atom symbol list
        dt :: timestep in ps

        """
        self.fout = out
        if fast:
            self.fast=True
            assert path is not None
            self._load(path, start, n_processes)
            return 

        self.fast=False
        self._from_stream = from_stream
        if from_stream:
            self._from_file = False
            self._setup_from_stream(AtomNum, FrameNum, coords, atom_lt, Cartesian)
            return 

        self._from_file = True
        self._memmap:bool=memmap
        self.flag=flag
        if slicing:
            self.start: None|int = slicing.start
            self.stop: None|int = slicing.stop
            self.step: None|int = slicing.step
            self.count=0
            
            return 
            
        # no `Cartesian` arg because read from file must be xyz file whose coordinates must be `Cartesian`
        self._setup(path, temperature, run_type, system_name, dt, MolIdx, details, encoding)
    
    def _setup(self, path, temperature, run_type, system_name, dt, MolIdx, details, encoding):
        """
        run_type = ['unconstrained', 'constrained', 'trs_rot_constrained', 'trans_constraint']
        name = 'sodium' or 'lithium'
        T = 450, 300, 600, 900, 1200
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found, please check again.")
        #print(path)
        self.path = path
        self.T = temperature
        self.run_type = run_type
        self.name = system_name
        self.dt = dt
        self.MolIdx = MolIdx
        self.file_suffix = str(path).split('/')[-1].split('.')[-1]
        self.details = details
        if encoding is None:
            self._encoding = 'utf-8'
        else:
            self._encoding = encoding

        self._setup_low()

    @abstractmethod
    def _get_one_frame(self):
        """traj_fo is the trajectory file 
        object waiting to be analyzed."""

    @abstractmethod
    def _read_info(self, ):
        """read trajectory information."""

    @abstractmethod
    def _get_one_frame_mmap(self):
        """mmap version."""

    def __enter__(self):
        return self

    def _setup_low(self):
        if not self.file_suffix in FormatBase.__supported:
            raise NotImplementedError(
                    f"supported formats now are {', ' .join(FormatBase.__supported)}")
        f = open(self.path, 'r', encoding=self._encoding)
        if self._memmap:
            self.f = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
            f.close()
        else:
            self.f = f
        #st=time.time() 
        #print(0)
        self._read_info()
        #print(1)
        self.f.seek(0)
        self._get_one_frame()
        #print(2)
        self.f.seek(0)
        #print('_read_info', time.time() - st)
        self._setup_mass()
        self._setup_group()
        #print(3)
        #self.f.seek(0)

        #if details:
        ##st=time.time()
        #if self._memmap:
        #    self._get_one_frame_mmap()
        #else:
        #    self._get_one_frame()
        ##print('_get_one_frame', time.time() - st)
        ## here we already read one frame but in this case in
        ## the `for` loop we start from the second frame.
        ## but we still want to start with the first frame.
        #self.f.seek(self._offsets[0])

        ##if self._chk_traj:
        ##print('counting and checking...')
        ##st = time.time() 
        #if self.details:
        #    if self._memmap:
        #        self._check_and_count_mmap()
        #    else:
        #        self._check_and_count()
        ##print('_check_and_count', time.time() - st)

        #self.f.seek(self._offsets[0]) # anyway, we go back to the first AtomNum line
        #self.flag=False # set to False in case OneFrame scenario.
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.fast:
            self.f.close()

    def __next__(self):
        if self._count == self.FrameNum:
            raise StopIteration
        self._count+=1
        self._get_one_frame()

    def __iter__(self):
        if self.fast is True:
            raise Exception("you should iterate coord_lt attribute in fast mode ")
        self.f.seek(0)
        self._count = 0
        return self

    @classmethod
    def gen(cls, AtomNum, FrameNum, coords, atom_lt, dt=None, slicing=None, Cartesian=True, ):
        #bypass the file-related `trajetory` object generation, automatic generation via 
        #info supplemented.
        return cls(from_stream=True, AtomNum=AtomNum, FrameNum=FrameNum, 
                   coords=coords, atom_lt=atom_lt, dt=dt, slicing=slicing)

    def __getitem__(self, indice):
        """The usage of traj_object is, e.g., 
        `with ReadXYZ(filename) as traj:
            pass
            traj_50 = traj[50] # works
            # this returns a traj object with start, stop and step attribute
            # the implementation of `for` loop will just looping over this slice.
            traj_200_300 = traj[200:300] # works
            for f in traj_200_300:
                pass
            traj_300_2=traj[:300:2] # works
        Traj=traj[:22] # out of the context manager thus fails.
        """
        if isinstance(indice, numbers.Integral):
            if indice >= self.FrameNum:
                raise IndexError("index out of current frame range")
            start_char = self._offsets[indice]
            #end_char = self._offsets[indice+1]
            # this lead to the beginning of the cmt line
            self.f.seek(start_char)            
            if self._memmap:
                self._get_one_frame_mmap()
            else:
                self._get_one_frame()
            traj = self.__class__.gen(self.AtomNum, 1, self.coords, self.atom_lt, self.dt, slicing=indice, Cartesian=True)
            if hasattr(self, 'lat_vec'):
                traj.lat_vec = self.lat_vec
                traj.ilat_vec = self.ilat_vec
            self.f.seek(self._offsets[0])
            if (self.FrameNum -1 == indice):
                self.flag = False
            return traj

        elif isinstance(indice, slice):
            # creating new traj object with marked `start`
            # `stop` and `step` attr
            traj=self.__class__(slicing=indice,)
            # then simply verify the three input indices
            for s in (traj.stop, traj.start, traj.step):
                if not (s is None or isinstance(s, numbers.Integral)):
                    raise TypeError("slice object elements must be None or Integral")

            if traj.stop is not None:
                if traj.stop > self.FrameNum:
                    raise IndexError("index out of range")

            elif (traj.start is not None and traj.start < 0):
                raise ValueError

            elif (traj.step is not None and raj.step < 1):
                raise ValueError
            # now the both pointing to the self.f object 
            # when self.f object is closed, traj.f object is close
            # as well!
            traj.f = self.f
            traj._offsets = self._offsets   
            traj.atom_lt = self.atom_lt
            #traj.FrameNum
            traj.AtomNum = self.AtomNum
            traj.annotations = self.annotations
            traj.dt = self.dt

            traj.start = 0 if traj.start is None else traj.start
            traj.stop = self.FrameNum if traj.stop is None else traj.stop
            traj.step = 1 if traj.step is None else traj.step

            traj.f.seek(traj._offsets[traj.start])
            traj._memmap=self._memmap
            if traj._memmap:
                traj._get_one_frame_memmap
            else:
                traj._get_one_frame()

            res=traj.f.seek(traj._offsets[traj.start])
            # (((stop - 1) - start) + 1) / step -> floor
            traj.FrameNum = math.floor((traj.stop - traj.start) / traj.step)
            if hasattr(self, 'lat_vec'):
                traj.lat_vec = self.lat_vec
            return traj
        else:
            raise TypeError("input must be of class 'slice' or 'int'")

    def _setup_from_stream(self, AtomNum, FrameNum, coords, atom_lt, Cartesian=True):
        if AtomNum is None:
            raise ValueError ("must supply atom number in the trajectory")
        if FrameNum is None:
            raise ValueError ("must supply frame number in the trajectory")
        if coords is None:
            raise ValueError ("must supply atom coordinates in the trajectory")
        if atom_lt is None:
            raise ValueError ("must supply atom coordinates in the trajectory")
        self.AtomNum = AtomNum
        self.FrameNum = FrameNum
        if Cartesian:
            self.coords = coords
        #else:
        #    self.frac_coords = coords
        self.atom_lt = atom_lt
        self._setup_group()

    def write_to_xyz(self, atom_labels: Sequence, filepaths, digit=8, frame_range=None, cmt=None):
        """select the frame and wanted atoms you want to write to the xyz file.
        examples
        might with the utility of reducing digit
        ------------
        atom_labels :: [0, 1, 3] this three atoms.
        with Analysis.XYZ(file_name) as traj:
            T = traj[200:]
            T.write_to_xyz([0, 1, 3])
        digit :: how many digit you want to have for coordinates
        range
            
        """
        print('writing to xyz file...')
        if isinstance(atom_labels, int):
            assert (atom_labels == self.AtomNum)
            atom_labels = np.array(range(atom_labels))
        if frame_range is None:
            frame_range=list(range(self.FrameNum))
        if cmt is None:
            cmt='\n'
    
        for m, filepath in enumerate(filepaths):
            fr = frame_range[m]
            with open(filepath, "w") as file:
                if self.fast:
                    assert self.FrameNum >= fr[-1]
                    for i in range(self.n_processes):
                        line = self.coord_lt[i]
                        #print(line.shape)
                        BaseNr = i*line.shape[0]
                        for j, l in enumerate(line):
                            if j+BaseNr in fr:
                                file.write(f"    {str(len(atom_labels))}\n")
                                file.write(cmt)
                                for label in atom_labels:
                                    file.write(f"{self.atom_lt[label]}    ")
                                    coords=str(np.round(l[label],digit)).strip("[] ")
                                    file.write(coords)
                                    file.write("\n")
                elif self._from_file:
                    for i, frame in enumerate(self):
                        if i in fr:
                            file.write(f"    {str(len(atom_labels))}\n")
                            file.write(cmt)
                            for label in atom_labels:
                                file.write(f"{self.atom_lt[label]}    ")
                                coords=str(np.round(self.coords[label],digit)).strip("[] ")
                                file.write(coords)
                                file.write("\n")
                elif self._from_stream:
                    #3d (self.FrameNum, self.AtomNum, 3)
                    for i in range(self.FrameNum):
                        file.write(f"    {str(len(atom_labels))}\n\n")
                        for label in atom_labels:
                            file.write(f"{self.atom_lt[label]}    ")
                            coord=str(self.coords[i][label]).strip("[] ").split() # ['x', 'y', 'z']
                            for c in coord:
                                c = '{:.8f}'.format(float(c))
                                file.write('{:<17}'.format(c))
                            file.write("\n")

            #elif self._from_stream:
            #    #2d coordinates array 
            #    for i in range(self.FrameNum):
            #        new_labels = i * self.AtomNum + atom_labels
            #        file.write(f"    {str(len(atom_labels))}\n")
            #        file.write(f"{cmt}\n")
            #        for label in new_labels:
            #            file.write(f"{self.atom_lt[label]}    ")
            #            coords=str(self.coords[label]).strip("[] ")
            #            file.write(coords)
            #            file.write("\n")

    def _setup_mass(self):
        self.mass_lt = np.array([atomic_masses[chemical_symbols.index(symbol)] for symbol in self.atom_lt])
        #self.com0 = get_com(self.coords, self.mass_lt) 
        # note that in this case we first calculate com and unwrap/wrap the atom

    def _setup_group(self):
        idx_atom_lt = [(atom, i) for i, atom in enumerate(self.atom_lt)]
        self.idx_d = defaultdict(list)
        for k, v in idx_atom_lt:
            self.idx_d[k].append(v)
        self.type_lt = np.array([k for k in self.idx_d.keys()])

    def setup_group(self, idx_d):
        """
        with ana.XYZL(path=path, lat_vec=lat_vec) as trajl:
            trajl.setup_group({'S1': [1,2,3], 'S2':[4,5,6]})
        """
        for k, v in idx_d.items():
            if k in self.idx_d:
                raise ValueError('errors happened when trying to modify original system-self-generated group dictionary')
            self.idx_d[k] = v

    def write_to_poscar(self, dump_name=None, Cartesian=True, frame=0):
        if (not hasattr(self, 'lat_vec')):
            raise AttributeError('need lattice info to write to POSCAR file')
        if dump_name is None:
            if Path('POSCAR').exists():
                raise RuntimeError('POSCAR already exists, you can pass in new name you want to dump')
            dump_name = 'POSCAR'

        poscar_dict=same_cat_pair(self.atom_lt)
        if self.file_suffix.lower() != 'poscar': 
            set_lt=[]
            for ele in self.atom_lt:
                if ele in set_lt:
                    continue
                set_lt.append(ele)
                idx = []
                for ele in set_lt:
                    for i, eele in enumerate(self.atom_lt):
                        if ele == eele:
                            idx.append(i)

        file = open(dump_name, 'w')
        if self.name is None:
            self.name = ' '
        file.write(f'{self.name.strip()} POSCAR\n')
        file.write('1.0\n') # scale factor we here set as 1.0
        for vec in self.lat_vec:
            vec_str=str(vec).strip('[] ')
            file.write(f'\t{vec_str}\n')
        for k in poscar_dict.keys():
            file.write(f'{k}  ')
        file.write('\n')
        for v in poscar_dict.values():
            file.write(f'{v}  ')
        file.write('\n')
        if self.file_suffix.upper() == 'POSCAR':
            idx_coords = self.coords 
        else:
            idx_coords = self[frame].coords[idx] # correctly sorted for POSCAR output
        if Cartesian:
            file.write('Cartesian\n')
        else:
            file.write('Direct\n')
            idx_coords = cc_to_fc(idx_coords, lat_vec=self.lat_vec)
        # poscar is one frame object
        for coords in idx_coords:
            coord = str(coords).strip('[] ').split()
            file.write('\t')
            for c in coord:
                c = '{:.8f}'.format(float(c))
                file.write('{:<15}'.format(c))
            file.write('\n')
    
    @abstractmethod
    def _load(self,fn,start,n_processes):
        """fast loading large trajectory."""
