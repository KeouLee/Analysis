import time
import numpy as np
import re
import os
from monty.io import reverse_readfile
#from Analysis.data.periodic import chemical_symbols

def dict_pretty(d, file_name):
    with open(file_name, 'a') as f:
       f.write(f"#{time.ctime()}\n")
       for k, v in d.items():
           f.write(f"{k}\t\t{v}\n")

def WriteToXyz(coords, atom_lt, cmt, file_name):
    FrameNum = coords.shape[0]
    AtomNum = len(atom_lt)
    assert cmt[-1] == '\n'

    if os.path.exists(file_name):
        os.remove(file_name)
    print('writing to xyz file...')
    with open(file_name, 'a') as f:
        for i in range(FrameNum):
            f.write(f'  {AtomNum}\n')
            f.write(cmt)
            for j in range(AtomNum):
                c = str(coords[i][j]).strip("[] ")
                f.write(f' {atom_lt[j]}  ')
                f.write(c)
                f.write('\n')

def cond_text2dict(fn):
    count=0
    f=reverse_readfile(fn)
    data={}
    for line in f:
        if (mo:=re.match(r'RUN_TYEP\s+(\w+)',line)):
            data['run_t']=mo.group(1)
        elif (mo:=re.match(r'TIME\s+([\.\d]+)',line)):
            data['TIME'] = float(mo.group(1))
        elif (mo:=re.match(r'temperature\s+(\d+)', line)):
            data['T']=int(mo.group(1))
        elif (mo:=re.match(r'S\s+([\d\.e-]+)', line)):
            data['S']=float(mo.group(1))
            #return data
        #elif (mo:=re.match(r'S_list\s+(\[(.)+\])', line)):
        #    data['S_list']=np.array(mo.group(1).strip('[] ').split(), dtype=np.float64)
        #    count+=1
        #    if count == 2:
        #        return data
        #elif(mo:=re.match(r'TIME_list\s+(\[(.)+\])', line)):
        #    count+=1
        #    data['TIME_list']=np.array(mo.group(1).strip('[] ').split(), dtype=np.float64)
        #    if count == 2:
        #        return data
        elif (mo:=re.match(r'S_sigma\s+(\d+\.\d+)', line)):
            data['S_sigma']=float(mo.group(1))
        elif (mo:=re.match(r'Haven_ratio\s+(\d+\.\d+)', line)):
            data['HR']=float(mo.group(1))
        elif (mo:=re.match(r'D\s+(\d+\.\d+e-\d+)', line)):
            data['D']=float(mo.group(1))
            f.close()
            return data

def WritePath(coords, atom_lt, part_num, each_part, dump_name):
    """write transport pathway."""
    SymbolList = np.setdiff1d(chemical_symbols, atom_lt)
    AtomNum = coords.shape[0]
    for i in range(part_num):
        for j in range(each_part):
            atom_lt.append(SymbolList[i])
    WriteOneFrame(coords, atom_lt, dump_name)

def WriteOneFrame(coords, atom_lt, dump_name):
    with open(dump_name, 'w') as f:
        print(len(coords), file=f, end='\n\n')
        for i in range(len(coords)):
            print(atom_lt[i], file=f, end=' ')
            text = '  '.join(np.array(coords[i],dtype='<U16'))
            print(text, file=f)

def write_to_xyz_one_frame(coords, atom_lt, fn):
    Scoords = np.array(coords, dtype='<U16')
    with open(fn, 'w') as f:
        print(len(atom_lt), file=f, end='\n\n')
        for i in range(len(atom_lt)):
            print(atom_lt[i],file=f, end=' ')
            print('  '.join(Scoords[i]), file=f)

def np_fmt(coords):
    for coord in coords:
        print("{:>8.4f}{:>8.4f}{:>8.4f}".format(coord[0], coord[1], coord[2]))
        #print('HETATM{:>5} {:<4s} {:>3s} A   1    {:>8}{:>8}{:>8}  1.00  0.00{:>12}'.format(i+1, spcl_atom_lt[i], self.residue, cc[0], cc[1], cc[2], spcl_atom_lt[i]), file=f)

def replace(fn, dn, idx_lt):
    """
    please provide idx_lt[i] as int.
    """
    j=0
    r=0
    with open(dn, 'w') as ff:
        with open(fn, 'r') as f:
            for i in range(3):
                ff.write(f.readline())
            for i, line in enumerate(f):
                #lines.append(line)
                if line == 'TER\n' or line == 'END\n':
                    ff.write(line)
                    continue
                if int(line.split()[1]) in idx_lt:
                    line=line.replace('S   SO4', 'Zn  ZN ')
                    line=line.replace(' S', 'Zn ')
                    j=1
                    r+=1
                    ff.write(line) 
                if j%6!=0:
                    j+=1
                    continue
                num=line.split()[1]
                line=line.replace(num,str(int(num)-r*4))
                ff.write(line) 
