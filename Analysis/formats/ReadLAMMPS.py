import re
import io
import pandas as pd
import numpy as np

class LAMMPS:
    def __init__(self, AtomNum, coords=None, type_lt=None, atom_lt=None, df=None, restricted_lat=None):
        """kind = 'thermo' or 'conf'."""
        self.AtomNum=AtomNum
        if df is None:
            self.coords = coords
            self.type_lt = type_lt
            self.atom_lt = atom_lt

            # self.rl = [xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz]
            for i in range(len(restricted_lat)):
                if abs(restricted_lat[i]) < 1e-4:
                    restricted_lat[i] = 0.
            
            self.rl = restricted_lat
            self.kind = 'conf'
            

        else:
            self.thermo = df
            self.kind='thermo'

    @classmethod
    def log_file(cls, fn):
        """convinient class method for construction of lammps post-info."""
        offsets=[]
        with open(fn, 'r') as f:
            start=False
            while (line := f.readline()):
                if start:
                    if re.match(r'(\s+[\w\.-]+){13}', line):
                        pass
                    else:
                        offsets.append(p)
                        break
                elif re.match(r"\s+Step\s+(\w+\s+)+", line):
                    offsets.append(int(p))
                    NrField=len(line.split())
                    start=True
                elif (mo:=re.match(r'\s+(\d+)\satoms', line)):
                    AtomNum=int(mo.group(1))
                p=f.tell()
            f.seek(offsets[0])
            s=''
            while True:
                s+=f.readline()
                p=f.tell()
                if p == offsets[1]:
                    break
        ff=io.StringIO(s)
        df=pd.read_csv(ff, sep='\s+', )
        return cls(AtomNum=AtomNum, df=df)

    def write_to_conf(self, fout='conf.lmp'):
        # self.rl = [xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz]
        xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = self.rl
        
        f=open(fout, 'w')
        print('', file=f)
        print(f'{self.AtomNum} atoms', file=f)
        print(f'{len(self.type_lt)} atom types', file=f)
        print(f'   {xlo}     {xhi} xlo xhi', file=f)
        print(f'   {ylo}     {yhi} ylo yhi', file=f)
        print(f'   {zlo}     {zhi} zlo zhi', file=f)
        print(f'   {xy}     {xz}    {yz} xy xz yz\n', file=f)
        print('Atoms # atomic\n', file=f)
        for i in range(self.AtomNum):

            idx=list(self.type_lt).index(self.atom_lt[i])
            s='   '.join(list(np.array(self.coords[i], dtype=str)))
            #
            #s=list(self.coords[i]).join('    ')
            print(f'     {i+1}     {idx+1}   {s}', file=f)
