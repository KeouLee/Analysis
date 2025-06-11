import numpy as np
import copy
import re
from pathlib import Path

class OUTCAR():
    @classmethod
    def chk_cver(cls, fn=None):
        """
        check convergence
        For different SINGLE-POINT calculation comparison. 
        Write related labels into a data dict .
        """
        cutoff_patt = r'\s*energy-cutoff\s+:\s+([\d\.]+)'
        coords_patt = r'\s*POSITION\s+TOTAL\-FORCE \(eV/Angst\)'
        virial_patt = r'\s*in kB\s+([-\.\d]+)\s+([-\.\d]+)\s+([-\.\d]+)\s+([-\.\d]+)\s+([-\.\d]+)\s+([-\.\d]+)\s+'
        if fn is None:
            fn = 'OUTCAR'
        data = {}
        f = open(fn)
        while (line := f.readline()):
            if (mo := re.match(cutoff_patt, line)):
                data['energy-cutoff'] = float(mo.group(1))
            if (mo := re.search('NIONS', line)):
                data['AtomNum'] = int(line.split()[-1])
                data['coord'] = np.zeros((data['AtomNum'], 3))
                data['frc'] = np.zeros((data['AtomNum'], 3))
            if re.match(r'\s*VOLUME and BASIS-vectors are now :', line):
                _ = [f.readline() for i in range(4)]
                lat_vec = [np.array(f.readline().split()[:3], dtype=np.float64) for i in range(3)]
                data['lat_vec'] = np.array(lat_vec)
            if (mo:=re.match(virial_patt, line)):
                data['virial'] = np.zeros((3,3))
                data['virial'][0][0] = mo.group(1)
                data['virial'][1][1] = mo.group(2)
                data['virial'][2][2] = mo.group(3)
                data['virial'][0][1] = mo.group(4)
                data['virial'][1][2] = mo.group(5)
                data['virial'][0][2] = mo.group(6)
                data['virial'][1][0] = data['virial'][0][1]
                data['virial'][2][0] = data['virial'][0][2]
                data['virial'][2][1] = data['virial'][1][2]
            if re.match(coords_patt, line):
                f.readline()
                for i in range(data['AtomNum']):
                    line = f.readline()
                    pos_frc=line.split()
                    data['coord'][i] = np.array(pos_frc[:3], dtype=np.float64)
                    data['frc'][i] = pos_frc[3:]
            if re.match('\s+FREE ENERGIE OF THE ION-ELECTRON SYST', line):
                f.readline()
                data['energy'] = float(f.readline().split()[-2]) #eV
            if line == '':
                break
        #print(data) 
        # rescale virial
        vol = np.linalg.det(data['lat_vec'])
        v_pref = 1 * 1e3 / 1.602176621e6
        data['virial'] *= vol * v_pref
        f.close()
        return data

    @classmethod
    def chk_rlx(cls, fn=None, relative=True, each=False, threshold=0.01):
        """
        monitor the force evolution on a given rlx process.
        """
        frc_patt = r'\s*POSITION\s+TOTAL\-FORCE \(eV/Angst\)'
        if fn is None:
            fn = 'OUTCAR'
        frcs = [] 
        f = open(fn)
        while (line := f.readline()):
            if re.search('NIONS', line):
                AtomNum = int(line.split()[-1])
            if re.match(frc_patt, line):
                f.readline()
                frc = np.zeros((AtomNum,3))
                for i in range(AtomNum):
                    frc[i]=np.array(f.readline().split()[-3:], dtype=np.float64)
                frcs.append(copy.deepcopy(frc))
        f.close()
        if each:
            count = [AtomNum-np.sum(np.linalg.norm(fs, axis=1)<threshold) for fs in frcs]
            return count

        if relative:
            return np.linalg.norm(np.sum(np.abs(frcs - frcs[-1]), axis=1) / AtomNum, axis=1)
        return np.linalg.norm(np.sum(np.abs(frcs), axis=1) / AtomNum, axis=1)

    @classmethod
    def get_data_dp(cls, fn=None):
        pat_cell = r'\s*VOLUME and BASIS-vectors are now :'
        pat_coord_frc = r'\s*POSITION\s+TOTAL\-FORCE \(eV/Angst\)'
        pat_ener = '\s+FREE ENERGIE OF THE ION-ELECTRON SYST'

        if fn is None:
            fn = 'OUTCAR'
        f = open(fn)
        FrameNum = 0
        while (line := f.readline()):
            if re.match(pat_cell, line):
                FrameNum += 1 
            if (mo := re.search('NIONS\s+=', line)):
                AtomNum = int(line.split()[-1])

        f.seek(0)
        lat_vec = np.zeros((FrameNum,9))
        coord = np.zeros((FrameNum, AtomNum * 3))
        frc = np.zeros((FrameNum, AtomNum * 3))
        ener = np.zeros((FrameNum,))

        i = 0
        while (line := f.readline()):
            if re.match(pat_cell, line):
                _ = [f.readline() for i in range(4)]
                for j in range(3):
                    lat_vec[i][3*j:3*(j+1)] = np.array(f.readline().split()[:3], dtype=np.float64)
                #lat_vec = [np.array(f.readline().split()[:3], dtype=np.float64) for i in range(3)]

            if re.match(pat_coord_frc, line):
                f.readline()
                for j in range(AtomNum):
                    line = f.readline()
                    pos_frc=np.array(line.split(), dtype=np.float64)
                    coord[i][3*j:3*(1+j)] = pos_frc[:3]
                    frc[i][3*j:3*(1+j)] = pos_frc[3:]
            if re.match('\s+FREE ENERGIE OF THE ION-ELECTRON SYST', line):
                f.readline()
                ener[i] = float(f.readline().split()[-2]) #eV
                i+=1
            if line == '':
                break
        data = {}
        data['frc'] = frc
        data['coord'] = coord
        data['lat_vec'] = lat_vec
        data['energy'] = ener
        return data
