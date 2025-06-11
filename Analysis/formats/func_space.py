import numpy as np
from typing import Sequence

def adjacent_subtract(seq: Sequence, last_one):
    """for one dimensional adjacent subtraction."""
    a = list(seq)[1:] + [last_one,]
    res = np.array(a) - np.array(seq)
    return res
