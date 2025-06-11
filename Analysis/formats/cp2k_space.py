import re
import numpy as np
from itertools import chain

def get_lattice_vector_cp2k(inp):
    cell_reg = r'\s+&CELL'
    reg = r'(ALPHA_BETA_GAMMA|[A-C]+)\s*(\[)?(?(2).*\]\s*|\s*)([-\dE\+.]+\s+[-\dE\+.]+\s+[-\dE\+.]+)'
    #lat_vec = np.zeros((3,3))
    vec_flag=True
    with open(inp, 'r') as f:
        while (line := f.readline()):
            if (re.search(cell_reg, line)):
                line = f.readline()
                match_obj = re.search(reg, line)
                head, tail = match_obj.groups()[0].upper(), match_obj.groups()[-1]
                if len(head.strip()) == 1:
                    print('reading lattice vector info from cp2k input file')
                    line2=f.readline()
                    line3=f.readline()
                    return _get_lat_vec(line,line2,line3)

                elif len(head.strip()) == 3:
                    print('reading lattice parameters info from cp2k input file')
                    edges = np.array(tail.split(), dtype=np.float64)
                    vec_flag=False
                    line2 = f.readline()
                    angles = np.array(line2.split()[1:], dtype=np.float64)
                    return np.array(list(chain(edges, angles)))
                elif len(head.strip()) == 16:
                    print('reading lattice parameters info from cp2k input file')
                    angles = np.array(tail.split(), dtype=np.float64)
                    vec_flag=False
                    line2 = f.readline()
                    edges = np.array(line2.split()[1:], dtype=np.float64)
                    return np.array(list(chain(edges, angles)))
                #match head:
                #    
                #    #<<<?>>>
                #    case 'ABC':
                #        edges = np.array(tail.split(), dtype=np.float64)
                #        vec_flag=False
                #    case 'A':
                #        print('A')
                #        lat_vec[0] = np.array(tail.split(), dtype=np.float64)
                #    case 'B':
                #        print('B')
                #        lat_vec[1] = np.array(tail.split(), dtype=np.float64)
                #    case 'C':
                #        print('C')
                #        lat_vec[2] = np.array(tail.split(), dtype=np.float64)
                #    case 'ALPHA_BETA_GAMMA':
                #        #assert not flag, 'multiple ALPHA_BETA_GAMMA statements in the cp2k input file!'
                #        angles = np.array(tail.split(), dtype=np.float64)
                
def _get_lat_vec(line1,line2,line3):
    L = []
    lines=[line1,line2,line3]
    characters = [line1.strip()[0], line2.strip()[0], line3.strip()[0]]
    if characters[0] == 'A':
        L.append(0)
        if characters[1] == 'B':
            L.append(1)
            L.append(2)
        else:
            L.append(2)
            L.append(1)

    elif characters[0] == 'B':
        L.append(1)
        if characters[1] == 'A':
            L.append(0)
            L.append(2)
        else:
            L.append(2)
            L.append(0)

    elif characters[0] == 'C':
        L.append(2)
        if characters[1] == 'A':
            L.append(0)
            L.append(1)
        else:
            L.append(2)
            L.append(0)
    lat_vec = np.zeros((3,3))    
    for i,l in enumerate(L):
        lat_vec[l] = np.array(lines[i].split()[1:], dtype=np.float64)
    return lat_vec

def get_ts_and_df(inp):
    f = open(inp, 'r') 
    line = f.readline()
    count = 0 
    while True:
        if count == 2 or line == '':
            break
        elif (mo:=re.match(r'\s*TIMESTEP\s+([\d+\.]+)', line)):
            ts = float(mo.group(1))
            count += 1
        elif re.search(r'\&TRAJECTORY',line):
            count += 1
            LINE = f.readline()
            while True:
                if (mo:=re.match(r'\s*MD\s+([\d+\.]+)', LINE)):
                    df = float(mo.group(1))
                    break
                LINE = f.readline()
        line = f.readline()
    #print(ts, df)
    return ts, df


if __name__ == "__main__":
    def test():
        #get_lattice_vector("/home/keli/solid_state/battery/TRAJ/MD_600K_normal/phase1/Na11Sn2PS12_normal.inp")
        #get_lattice_vector_cp2k("/home/keli/solid_state/battery/TRAJ/BOROHYDRIDE/MD_450K_normal/phase1/LiBH4_HT_normal.inp")
        get_ts_and_df('/home/keli/solid_state/battery/MLP/LiBH4/Run/TRAJ/unconstrained/550K/run.inp')
    test()
