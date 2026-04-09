import numpy as np
import matplotlib.pyplot as plt

def read_odd_lines(fname,num,offset=0):
  with open(fname, 'r') as f:
    for i in range(offset):
      f.readline()
    for i, line in enumerate(f):
      if i % num == 0:
        yield line

tmp = np.loadtxt(read_odd_lines('libh4-HILLS.metadynLog',7))
TIME = tmp[:,0]/1e3

CV = tmp[:,1]


HEIGHTS = np.loadtxt(read_odd_lines('libh4-HILLS.metadynLog',7,6))[:,-1]
WIDTHS = np.ones(len(CV))
#heights = np.concatenate((heights1, heights2),0)

#data = np.concatenate((data1,data2,),0)
print(len(WIDTHS))
for i in [6900,7900,8900,9900]:
    time = TIME[:i]
    cv = CV[:i]
    widths = WIDTHS[:i]
    heights = HEIGHTS[:i]


    min_grid = np.min(cv) - 2 * np.max(widths)
    max_grid = np.max(cv) + 2 * np.max(widths)

    n_bins = 500
    grid = np.linspace(min_grid,max_grid,n_bins)

    s = grid[:, np.newaxis]
    s_t = cv[np.newaxis,:]

    w_t = heights[np.newaxis, :]
    sig = widths[np.newaxis, :]

    gaussians = w_t * np.exp( - (s - s_t)**2 / (2 * sig**2) )

    bias_potential = np.sum(gaussians, axis=1)
    T_sim, delta_T = 300, 4000
    prefactor = (T_sim + delta_T) / delta_T 
    fes = (- bias_potential) * prefactor * 27.2114
    #fes = (- bias_potential) *  27.2114
    fes = fes - np.min(fes)
    try:
        plt.plot(grid, fes, linewidth=1.8, label=f'{int(time[i])}ps')
    except:
        plt.plot(grid, fes, linewidth=1.8, label=f'{int(time[-1])}ps')
#plt.annotate(restart,(1,1))
plt.ylabel('free energy (eV)',size=12)
plt.xlabel(r'cv ($\AA$)',size=12)
plt.legend()
plt.show()
