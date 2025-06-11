from typing import Sequence
from .wrapper_pickle import dump_pkl
import numpy as np

class Boson:
    def __init__(self, var: Sequence):
        self.var = var

    def __eq__(self, f):
        return all(v in self for v in f.var)


class Plot:
    """
    Generation of a Plot object,
    which is a data container class for quick plot.
    
    ARGUMENTS
    ------------
    x, y :: for plot
    dump_name :: to which file you dump current `Plot` object
    data :: a dict contains additional info

    USAGE
    ------------
    ~for dump: 
    Plot(x,y,data,dump_name)

    ~for load:
    p=load_pkl(dump_name)
    plt.plot(p.x, p.y, label=p.data['label'])
    """
    def __init__(self, x, y, dump_name, data):
        self.x = x
        self.y = y
        self.data = data
        dump_pkl(dump_name, self)


class Cylinder:
    def __init__(self, points):
        """points :: np.3darray"""
        self.points=points
        l=len(points)
        self.V_cylinder, self.H_cylinder, = np.zeros((l,3)), np.zeros(l)
        for i, p in enumerate(points):
            self.V_cylinder[i]=p[1]-p[0]
            self.H_cylinder[i]=np.linalg.norm(self.V_cylinder[0])
        self.multiplicity=len(self.H_cylinder)
