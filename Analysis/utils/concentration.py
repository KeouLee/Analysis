def get_concentration(c, M, density):
    """
    c in `mol/L (M)`
    M in `g/mol`
    density in `g/cm^3`
    """ 
    n_h2o=(1000-c*M/density)/18.01528
    # return number ratio
    r=n_h2o/c
    print(f'1:{r}')
