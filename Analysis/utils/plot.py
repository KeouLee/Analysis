from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import os
from ..examples.var_repo import parse_path

def get_df(fn, term):
    # get the filename string 
    #fn = glob.glob('*ener')[0]
    with open(fn, 'r') as f:
        fl = f.readlines()
    fl[0] = fl[0].strip('#')

    fstr = ''.join(fl)
    fo = StringIO(fstr)

    df = pd.read_csv(fo, sep='\s\s+', engine='python')
    y=df[term].to_numpy()
    return y

def plot(fn, term):
    fig, ax = plt.subplots()
    ax.plot(get_df(fn, term),label=term, linewidth=.7)
    ax.set_ylabel(term, fontsize='15')
    ax.set_xlabel('fs', fontsize='15')
    ax.legend(fontsize='9')

def plot_once(term, *args):
    fig, ax = plt.subplots()
    for fn in args:
        T, run_t = parse_path(fn)
        ax.plot(get_df(fn, term),label=f'{T}K_{run_t}', linewidth=.7)
        ax.legend(fontsize='9')
    ax.set_ylabel(term, fontsize='15')
    ax.set_xlabel('fs', fontsize='15')

def plot_twiny(fn, term1, term2):
    fig, ax1 = plt.subplots()
    y1=get_df(fn, term1)
    y2=get_df(fn, term2)
    
    ax1.set_xlabel('fs',fontsize='15')
    ax1.set_ylabel(term1,fontsize='15',color='tab:red')
    ax1.plot(y1,color='tab:red',linewidth=.7)
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()

    ax2.set_ylabel(term2,fontsize='15',color='tab:blue')
    ax2.plot(y2,color='tab:blue',linewidth=.7)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    fig.tight_layout()

