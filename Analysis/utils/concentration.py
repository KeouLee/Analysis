from scipy import constants

def c2n(c, M, density):
    """
    concentration to number ratio (ion number : water num)
    c in `mol/L (M)`
    M in `g/mol`
    density in `g/cm^3`
    """ 
    n_h2o=(1000-c*M/density)/18.01528
    # return number ratio
    r=n_h2o/c
    print(f'1:{r}')

def n2c(n, N, M, density):
    """
    number to concentration
    n :: number of solute
    N :: number of water molecule
    M :: molar mass of solute in `g/mol`
    density :: density of solute in `g/cm^3`
    """
    NA=constants.Avogadro
    #NA=6.022e23
    V_h2o=N*18.01528/NA # cm^3
    mol_solute=n/NA
    V_solute=mol_solute*M/density  # cm^3
    #print(mol_solute/(V_h2o)*1e3)
    print(mol_solute/(V_solute+V_h2o)*1e3)
