import matplotlib.pyplot as plt
import numpy as np

def read_odd_lines(fname):
  with open(fname, 'r') as f:
    for i, line in enumerate(f):
      if i % 8 == 0:
        yield line

data = np.loadtxt(read_odd_lines('libh4-COLVAR.metadynLog'))


stride = 1
time = data[:, 0]/1e3  
cv = data[:, 1]


plt.scatter(
    time[::stride],      
    cv[::stride],        
    s=10,                
    c='blue',            
    marker='.',          
    alpha=0.6,           
    edgecolors='none'    
)

plt.xlabel('Time / ps', fontsize=12, fontweight='bold')
plt.ylabel('CV Value / Bohr', fontsize=12, fontweight='bold')

plt.xlim(min(time), max(time))
# plt.ylim(-10, 10)

plt.tick_params(direction='in', top=True, right=True)
plt.title('300K/uc')

plt.tight_layout()
plt.show()
