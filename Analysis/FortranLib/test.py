import numpy as np
import scipy
import fortran_diagonalization as fd
#import tred

def generate_float_matrix():
    np.random.seed(5)
    mat=np.random.rand(3,3,) * 10
    mat=mat @ mat.T
    I_inertia = np.around(mat,2)
    I_inertia=np.asfortranarray(I_inertia)
    return I_inertia

ierr=-1
Nxyz=3
EN = np.zeros(3)
EDDI = np.zeros(3)
I_inertia = generate_float_matrix()
print(I_inertia)


#print(tred.tred2.__doc__)
#tred.tred2( d=EN, e=EDDI, nm=3, n=3,z=I_inertia)
fd.tred2(nm=Nxyz, n=Nxyz,d=EN, e=EDDI,z=I_inertia)
fd.imtql2(nm=Nxyz, n=Nxyz, d=EN, e=EDDI, z=I_inertia, ierr=ierr)
print(I_inertia)
print('D', EN)
print('E', EDDI)
#a = scipy.linalg.lapack.ssytrd(I_inertia,lower=0)
#print(a)
