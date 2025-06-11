import scipy
from scipy.constants import physical_constants
from scipy import constants

# this is a unitconv environment and operation such as `t_in_fs * fs_to_au` 
# renders us `t_in_au`

fs_to_au = 1.0e-15 / physical_constants["atomic unit of time"][0]
ang_to_au = constants.angstrom / physical_constants["atomic unit of length"][0]

kcalPermol_to_J = constants.calorie * 1e3 / constants.N_A

kcalPermol_to_au = kcalPermol_to_J / physical_constants["atomic unit of energy"][0]
bond_const_conv_fac = 1 / ang_to_au**2 * kcalPermol_to_au
bend_const_conv_fac = kcalPermol_to_au

amu = physical_constants["atomic mass constant"][0]
AuMass = physical_constants["atomic unit of mass"][0]
amu_to_AuMass = amu/AuMass

wavenumber_to_THz = 0.0299792458
evtok = 11604.588577015

hartree_to_eV = 27.2114079527
ang_to_bohr = ang_to_au
kJ_to_kcal = 0.23900574

__all__ = ['evtok',]

if __name__ == "__main__":
    #print(fs_to_au)
    #print(ang_to_au)
    #print(kcalPermol_to_J)
    #print(kcalPermol_to_au)
    #print(bond_const_conv_fac)

    #print(110 / 6.27509468713739E+02)
    #print(110 * kcalPermol_to_au)
    #print(.01 / hartree_to_eV / ang_to_bohr)
    print(43.3 / (hartree_to_eV*1000) / ang_to_bohr)
