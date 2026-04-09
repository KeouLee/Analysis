#import glob
#from .cp2k_space import *
from itertools import chain
import importlib
from pathlib import Path
from inspect import getmembers, isclass, isfunction

__all__ = ['XYZ', 'XTC', 'XYZL', 'POSCAR', 'XDATCAR', 'get_lattice_vector_cp2k', 'OUTCAR', 'LAMMPS', 'Rot', 'get_ts_and_df', 'CIF', 'MOL2', 'PDB', 'DCD', 'get_temperature_cp2k']
NOT_LOADABLE = ("__init__.py", "base.py", "__pycache__", "test.py", )#"func_space.py", "cp2k_space.py")
PACKAGE_BASE = "Analysis.formats"
TO_LOAD = ("Read*py", "*space.py", "rot_analysis.py")


# module_file is PosixPath(Read*py)
# module_file.name is Read*.py
# module_file.stem is Read*
load = chain(*(Path(__file__).parent.glob(L) for L in TO_LOAD))
#print(list(load))

#p = Path(__file__).parent.glob("Read*py")
for module_file in load:
    if module_file.name in NOT_LOADABLE:
        continue
    module_name = f".{module_file.stem}"
    module = importlib.import_module(module_name, PACKAGE_BASE)
    for name, cls in getmembers(module, isclass):
        exec(f"{name}=module.{name}")
    for name, func in getmembers(module, isfunction):
        exec(f"{name}=module.{name}")
