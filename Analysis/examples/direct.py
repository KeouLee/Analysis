import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from wrapper_pickle import load_pkl
#from dd import legacy

#@legacy
def plot_fel(T):
    """ plot free energy line """
    run_type_lt = ['unconstrained', 'rot_trsc', 'constrained', ]
    labels=['no constraint', 'translational and rotational constraint', 'rotational constraint, free translation']
    colors=['royalblue', 'purple', 'darkorange']
    nbins_r=130
    sizer=13
    dr=sizer/nbins_r
    #dist_bin=load_pkl('DistBin/dist_bin_S.pkl')
    #dist_bin=load_pkl('DistBin/dist_bin_P.pkl')
    #dist_bin=load_pkl('dist_bin_S.pkl')
    for i, run_type in enumerate(run_type_lt):
        dist_bin=load_pkl(f'DistBin/{run_type}_{T}_dist_bin_P.pkl')
        #dist_bin=load_pkl(f'dist_bin_P.pkl')
        FES=[]
        for b in dist_bin:
            fes=-T*np.log(b) * constants.physical_constants['Boltzmann constant in eV/K'][0]
            #print(fes)
            FES.append(fes)
        #
        fes_arr = np.array(FES)  
        fes_arr = fes_arr - np.min(fes_arr[:40])
        x=np.linspace(0, sizer, nbins_r)

        ax = plt.subplot()
        ax.set_xlim(2.5,8)
        ax.set_ylim(0,0.4)
        ax.plot(x,fes_arr, label=labels[i],color=colors[i])
    ax.set_xlabel(r'$r\ (\AA)$')
    ax.set_ylabel('Free Energy (eV)')
    ax.legend(handlelength=1.3)
    #ax.annotate('0.29', xy=(6.3493, 0.29249), xytext=(6.3+.3, .29), arrowprops=dict(arrowstyle='-', color=colors[-1]),)
    #ax.annotate('0.22', xy=(6.2488, 0.22202), xytext=(6.2+.6, .223), arrowprops=dict(arrowstyle='-', color=colors[1]),)
    #ax.annotate('0.20', xy=(6.1477, 0.20385), xytext=(6.1+.9, .205), arrowprops=dict(arrowstyle='-', color=colors[0]),)
    #plt.savefig('pics/free_energy/fel.png',dpi=1200)
    plt.show()



if __name__ == '__main__':
    plot_fel(1200)
    pass
