"""read xyz(especially cp2k generated) 
file and xyz zipped inside the zip file."""
import os
import re
from typing import Sequence
import logging
import multiprocessing
import collections
import time

import numpy as np
from itertools import islice, chain, accumulate
#from Analysis.CLib.get_coords_wrapper import Cget_coords
from ..utils.cat import get_com

from .base import FormatBase

np.set_printoptions(suppress=True) 
class XYZ(FormatBase):
    """read xyz file or zipped xyz file and return 
    the XYZ object representing the trajectory."""

    atomnum_part = r"\s*\d+\s+"
    cmt_part = r"[^\n]*\n"
    coords_part = r"\s*[A-Z][a-z]*\s+[\.0-9Ee\-\+]+\s+[\.0-9Ee\-\+]+\s+[\.0-9Ee\-\+]+\s+"

    atomnum_part_mmap = rb"\s*\d+\s+"
    cmt_part_mmap = rb"[^\n]*\n"
    coords_part_mmap = rb"\s*[A-Z][a-z]*\s+[\.0-9Ee\-\+]+\s+[\.0-9Ee\-\+]+\s+[\.0-9Ee\-\+]+\s+"
    #def __init__(self, path, traj_file_name=None):
    #    super().__init__(path, traj_file_name)
    #    self.flag=False
    
    @classmethod
    def check_defect(cls, fn, AtomNum):
        f = open(fn)
        LinesPerFrame = AtomNum + 2
        OneFrame = list(islice(f, LinesPerFrame))
        count = 0

        AP = re.compile(cls.atomnum_part)
        cp = re.compile(cls.cmt_part)
        CP = re.compile(cls.coords_part)
        while OneFrame:
            if not AP.match(OneFrame[0]):
                print('atomnum_part')
                print('LineNr:', count)
                raise Exception()
            if not cp.match(OneFrame[1]):
                print('cmt_part')
                print('LineNr:', count)
                raise Exception()
            for i in range(2,LinesPerFrame):
                if not CP.match(OneFrame[i]):
                    print('coords_part')
                    print('LineNr:', count)
                    raise Exception()
            count += LinesPerFrame

            OneFrame = list(islice(f, LinesPerFrame))



    def _read_info(self):    
        #coords_part = r"(\s*[A-Z][a-z]*\s+[\.0-9\-]+\s+[\.0-9\-]+\s+[\.0-9\-]+\s+)+"
        # you should assert `AtomNum` on the first line of the file...

        self.Cartesian=True
        #self._offsets = []
        self.AtomNum = int(self.f.readline())
        self.f.seek(0)
        self.LinesPerFrame = self.AtomNum + 2

        self._offsets = [0,]
        OneFrame = list(islice(self.f, self.LinesPerFrame))
        #OneFrame = [self.f.readline() for _ in range(self.LinesPerFrame)]
        # from first frame we get the `self.atom_lt`
        self.atom_lt = [line.split()[0]  for line in OneFrame[2:]]

        while OneFrame:
            self._offsets.append(len(tuple(chain.from_iterable(OneFrame))))
            #l=0
            #for line in OneFrame:
            #    l+=len(line)
            #self._offsets.append(l)
            OneFrame = list(islice(self.f, self.LinesPerFrame))
            #OneFrame = [self.f.readline() for _ in range(self.LinesPerFrame)]
        self._offsets = list(accumulate(self._offsets))
        self.FrameNum = len(self._offsets) - 1
         
        #cmt part
        # just the next behind the AtomNum is the cmt line

    #def _read_info(self):
    #    lines=self.f.readlines()

    def _get_one_frame(self):
        #OneFrame = list(islice(self.f, self.LinesPerFrame))
        #print('q')
        #OneFrame[2:]
        #self.coords=Cget_coords(OneFrame[2:])
        
        self.coords = np.array([s.split()[1:] for i, s in enumerate(islice(self.f, self.LinesPerFrame)) if i > 1], dtype=np.float32)


    def _get_one_frame_mmap(self):
        pass

    def __len__(self):
        return self.FrameNum

    def com_evolve(self, idx):
        com0 = get_com(self.coords[idx], self.mass_lt[idx])
        drifts=np.zeros((self.FrameNum, 3))
        for i, f in enumerate(self):
            drifts[i] = get_com(self.coords[idx], self.mass_lt[idx]) - com0
        return np.linalg.norm(drifts, axis=1)


    def _load(self, fn, start=0, n_processes=1):
        """fast load large trajectory and calculate the RDF of each chunk as they are only structural analysis.
           we directly trim the trajectory (discarding the pre-equil and retaining the equilbrium frame only)
           as not matter structural or dynamical are calculated from equilibirated frames only.
        """
        assert n_processes <= multiprocessing.cpu_count()
        self.n_processes=n_processes
        f=open(fn,'r')
        lines=f.readlines()
        self.AtomNum=int(lines[0])
        self.LinePerFrame=self.AtomNum+2
        self.atom_lt=[line.split()[0] for line in lines[2:self.LinePerFrame]] # quick peek for atom list generation 
        self._setup_group()
        LineNum=len(lines)
        assert (LineNum%self.LinePerFrame==0)
        self.FrameNum=LineNum//self.LinePerFrame
        self.EquilFrameNum=self.FrameNum-start

        n=self.EquilFrameNum//self.n_processes
        _shift=self.EquilFrameNum%self.n_processes
        allo=[[start, start+n*(i+1)+_shift] if i==0\
        else [start+i*n+_shift, start+(i+1)*n+_shift ]\
        for i in range(self.n_processes)]

        allo=[[p[0]*self.LinePerFrame, p[1]*self.LinePerFrame] for p in allo]

        lines=[lines[p[0]:p[1]] for p in allo]
        # then deal with lines and get self.coords
        #self.coords=np.empty((self.FrameNum,self.AtomNum,3))
        #allo=[self.EquilFrameNum//n_processes]*n_processes
        #for i in range(self.FrameNum%n_processes):
        #    allo[i] += 1
        #for i in range(len(allo)):
        #    allo[i] *= self.LinePerFrame
        #allo=collections.deque(accumulate(allo))
        #allo.appendleft(0)

        #lines = [lines[allo[i]:allo[i+1]] for i in range(len(allo)-1)]
        f.close()

        with multiprocessing.Pool(processes=self.n_processes) as pool:
            lines = pool.map(self._get_coords, lines)
            #now the `lines` is [(PartFrameNum, self.AtomNum,3), (PartFrameNum, self.AtomNum,3), ...]
            # we can now parallelize this arrays within list
        self.coord_lt = lines 

    def _get_coords(self, sub_lines):
        fn=-1
        coords = np.empty((len(sub_lines)//self.LinePerFrame,self.AtomNum,3))
        for i, sl in enumerate(sub_lines):
            if i%self.LinePerFrame == 0:
                fn+=1
            elif i%self.LinePerFrame == 1:
                pass
            else:
                ai=i-self.LinePerFrame*fn-2
                coords[fn,ai]=np.array(sl.split()[1:],dtype=np.float32)
        return coords
