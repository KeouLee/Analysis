import numpy as np
import scipy
import math
from pandas import DataFrame
from ..utils import load_pkl, dump_pkl
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from numpy.linalg import norm

class Rot:
    @classmethod
    def detect(cls, theta_file, block_size, step_size, interval, diff_angle=50, diff_threshold=30):
        """
        arguments
        ----------
        block_size: block_size in ps
        step_size: orginal chosen dump_step in trajectory
        interval: in step (interval x step_size = actually time in fs)
        diff_angle: threshold to distinguish a hopping
        """
        r=[]
        r.append(_detect(theta, block_size, step_size, interval, diff_angle, diff_threshold))
        return r



def _detect(theta, block_size, step_size, interval, diff_angle=50, diff_threshold=30):
    #theta=load_pkl(theta_file)#.flatten() # 40fs
    _block_size=block_size*1e3/step_size   # a block equals how many step in theta array
    assert int(_block_size) == _block_size
    _block_size = int(_block_size)

    lt = []
    start_points = range(0,len(theta)-2*_block_size,interval) #interval argument: how many times of steps you want to choose the starting point
    for sp in start_points:
        tail=theta[sp:_block_size+sp]
        head=theta[_block_size+sp:2*_block_size+sp]
        res=abs(np.average(head) - np.average(tail))
        if res > diff_angle:
            lt.append((sp+_block_size)*step_size/1e3) # store time in ps
    #print(get_last(lt, diff_threshold))
    #print(get_last(lt, diff_threshold))
    return get_last(lt, diff_threshold)


def get_last(lt, threshold):
    new_lt = []
    for i in range(len(lt)-1):
        tmp = abs(lt[i+1] - lt[i])
        if tmp > threshold:
            new_lt.append(lt[i])
    if len(lt) != 0:
        new_lt.append(lt[-1])
    return new_lt
