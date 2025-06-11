import numpy as np
from .wrapper_pickle import load_pkl, dump_pkl
"""rotation-related utils."""

def get_inertia(coord, mass_lt):
    mass_lt=np.array(mass_lt)
    if (len(mass_lt.shape) != 1):
        raise ValueError('expecting 1-d array `mass_lt`')

    if (not coord.shape[0] == len(mass_lt)):
        raise ValueError('mass array `mass_lt` and corresponding atom position are incompatible'
                f' detecting AtomNum {coord.shape[0]} but only {mass_lt.shape[0]} masses are provided')
    l = len(mass_lt)
    inertia_tensor = np.zeros((3,3),order='F')
    for i in range(l):
        inertia_tensor[0][0] += mass_lt[i] * (coord[i][1]**2 + coord[i][2]**2)
        inertia_tensor[1][1] += mass_lt[i] * (coord[i][0]**2 + coord[i][2]**2)
        inertia_tensor[2][2] += mass_lt[i] * (coord[i][0]**2 + coord[i][1]**2)
        inertia_tensor[0][1] += mass_lt[i] * coord[i][0] * coord[i][1]
        inertia_tensor[0][2] += mass_lt[i] * coord[i][0] * coord[i][2]
        inertia_tensor[1][2] += mass_lt[i] * coord[i][1] * coord[i][2]
    inertia_tensor[0][1] = -inertia_tensor[0][1]
    inertia_tensor[0][2] = -inertia_tensor[0][2]
    inertia_tensor[1][2] = -inertia_tensor[1][2]
    inertia_tensor[1][0] = inertia_tensor[0][1]
    inertia_tensor[2][0] = inertia_tensor[0][2]
    inertia_tensor[2][1] = inertia_tensor[1][2]
    return inertia_tensor
        
def get_power_spectrum(omegamax, domega, dt, dump_name, load_name=None, acf=None):
    """
    Description
    -----------
    Fourier Transform Routinue to obtain power spectrum.
    Arguments
    -----------
    omegamax is actually `omegamax = 2 * pi * w`.
    the real omega we use to plot is 'w'.
    """
    if (load_name is None and acf is None):
        raise ValueError('must supply acf file name or acf stream')

    if acf is None:
        acf = load_pkl(load_name)
    omega = np.arange(0, omegamax, domega)  # omega in THz 
    ps = np.zeros(len(omega))
     
    for i in range(len(ps)):
        for j in range(len(acf)):
            ps[i] += acf[j] * np.cos(omega[i] * (j+1) * dt) * dt * 2
    dump_pkl(dump_name, ps)

def rot_mat(a, b):
    """
    rotation matrix that transforms a to b 
    arguments
    -------------
    two normalized vector
    """
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    vx2 = vx@vx
    R = np.eye(3) + vx + vx2 / (1+c)
    return R

def detect_rotations( angle_threshold, pkl_file, timestep, T, vec_idx=0):
    """
    pkl_file :: pickle file shaped (FrameNum, GroupNum)
    timestep :: (in fs) convert the anglestep to ps 
    """
    total = 0
    f=open(f'RotRec_{T}K', 'w')
    f.write('time[ps]\n')

    # ensemble average 
    ANGLES = load_pkl(pkl_file)
    for j in range(ANGLES.shape[1]):
        f.write(f'GroupNum{j}\n')
        angles = ANGLES[:,j]

        count = 0
        RefAngle = angles[0]
        for i, angle in enumerate(angles):
            delta = angle - RefAngle
            if abs(delta) > angle_threshold:
                RefAngle=angle
                ps = i*timestep/1e3
                f.write(f'{ps}\n')
                count += 1
        total += count
        f.write(f'{T}K: {count}\n')
    f.write(f'Ensemble Averaged Rotations Number: {total/ANGLES.shape[1]}')
    f.close()
    return total/ANGLES.shape[1]

def get_fluct(fn):
    # real space transformation
    ARR=load_pkl(fn) # should be a 1d arr
    lt = []
    for arr in ARR:
        lt.append( (np.sum((arr - np.average(arr)) ** 2)/len(arr))**.5 )
    return np.average(lt)

def RM(v,degree):
    """
    k, axis about which to rotate
    v, rotated vector
    degree, angle
    """
    #k=k/np.linalg.norm(k)
    v=v/np.linalg.norm(v)

    v1=v.reshape(3,1)
    v2=v.reshape(1,3)
    m=np.array([[0, -v[2], v[1]],
                [v[2],0,-v[0]],
                [-v[1],v[0],0]])

    cos=np.cos(np.pi*degree/180)
    sin=np.sin(np.pi*degree/180)
    
    #print((1-cos)*np.matmul(v1,v2)) 
    return cos*np.eye(3) + (1-cos)*np.matmul(v1,v2) + sin*m

def RM1(k,v,degree):
    """
    k, axis about which to rotate
    v, rotated vector
    degree, angle
    """
    k=k/np.linalg.norm(k)
    cos=np.cos(np.pi*degree/180)
    sin=np.sin(np.pi*degree/180)

    return v*cos+np.cross(k,v)*sin+(1-cos)*np.dot(k,v)*k
