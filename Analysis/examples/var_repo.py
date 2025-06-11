import re
# run_type_lt and MolIdx are in correspondence
run_type_lt = ['free rotation, free translation', 'rotational constraint, free translation', 'rotational and translational constraint', ]

MolIdxList = [ [(110,205,186,191,200),(107,193,179,196,182),(109,188,185,206,203),(104,199,176,181,194),
            (105,198,177,180,195),(108,207,202,184,189),(106,197,192,183,178),(111,204,187,201,190)],
           [tuple(range(168,173)), tuple(range(173, 178)), tuple(range(178,183)), tuple(range(183,188)),
            tuple(range(188,193)), tuple(range(193, 198)), tuple(range(198,203)), tuple(range(203,208)),],
           [tuple(range(168,173)), tuple(range(173, 178)), tuple(range(178,183)), tuple(range(183,188)),
            tuple(range(188,193)), tuple(range(193, 198)), tuple(range(198,203)), tuple(range(203,208)),] ] # all PS4 moieties...
S_in_PS4 = [  {'S_in_PS4': [205, 186, 191, 200, 193, 179, 196, 182, 188, 185, 206, 203, 199, 176, 181, 194,
                            198,177,180,195,207,202,184,189,197,192,183,178,204,187,201,190]},
              {'S_in_PS4': [169, 170, 171, 172, 174, 175, 176, 177, 179, 180, 181, 182, 184, 185, 186, 187, 
                            189, 190, 191, 192, 194, 195, 196, 197, 199, 200, 201, 202, 204, 205, 206, 207]},
              {'S_in_PS4': [169, 170, 171, 172, 174, 175, 176, 177, 179, 180, 181, 182, 184, 185, 186, 187, 
                            189, 190, 191, 192, 194, 195, 196, 197, 199, 200, 201, 202, 204, 205, 206, 207]},
           ]
MolIdxListLiBH = [ tuple(range(36,41)), tuple(range(41,46)), tuple(range(46,51)), tuple(range(51,56)),
        tuple(range(56,61)), tuple(range(61,66)), tuple(range(66,71)), tuple(range(71,76)),
        tuple(range(76,81)), tuple(range(81,86)), tuple(range(86,91)), tuple(range(91,96)),
        tuple(range(96,101)), tuple(range(101,106)), tuple(range(106,111)), tuple(range(111,116)),
        tuple(range(116,121)), tuple(range(121,126)), tuple(range(126,131)), tuple(range(131,136)),
        tuple(range(136,141)), tuple(range(141,146)), tuple(range(146,151)), tuple(range(151,156)),
        tuple(range(156,161)), tuple(range(161,166)), tuple(range(166,171)), tuple(range(171,176)),
        tuple(range(176,181)), tuple(range(181,186)), tuple(range(186,191)), tuple(range(191,196)),
        tuple(range(196,201)), tuple(range(201,206)), tuple(range(206,211)) ]

EnerList = ('Temp[K]', 'Pot.[a.u.]', 'Cons Qty[a.u.]', 'UsedTime[s]')

temperature = [300, 600, 900, 1200]

traj_path_lt = ['/home/keli/solid_state/battery/TRAJ/MD_1200K_unconstrained/concatenate',
                '/home/keli/solid_state/battery/TRAJ/MD_900K_unconstrained/concatenate',
                '/home/keli/solid_state/battery/TRAJ/MD_600K_unconstrained/concatenate',
                '/home/keli/solid_state/battery/TRAJ/MD_300K_unconstrained/concatenate',

                '/home/keli/solid_state/battery/TRAJ/MD_1200K_constrained/concatenate',
                '/home/keli/solid_state/battery/TRAJ/MD_900K_constrained/concatenate',
                '/home/keli/solid_state/battery/TRAJ/MD_600K_constrained/concatenate',
                '/home/keli/solid_state/battery/TRAJ/MD_300K_constrained/concatenate',

                '/home/keli/solid_state/battery/TRAJ/MD_1200K_trs_rot_constrained/concatenate',
                '/home/keli/solid_state/battery/TRAJ/MD_900K_trs_rot_constrained/concatenate',
                '/home/keli/solid_state/battery/TRAJ/MD_600K_trs_rot_constrained/concatenate',
                '/home/keli/solid_state/battery/TRAJ/MD_300K_trs_rot_constrained/concatenate', ]

traj_path_lt_Li = ['/home/keli/solid_state/battery/TRAJ/BOROHYDRIDE/MD_450K_constrained/phase1/LiBH4_HT-HT_C_332-pos-1.xyz',
                   '/home/keli/solid_state/battery/TRAJ/BOROHYDRIDE/MD_450K_unconstrained/concatenate/coords.xyz', # old
                    #new
                   '/home/keli/solid_state/battery/TRAJ/BOROHYDRIDE/MD_450K_unconstrained/phase1/rotc_executable/RESTART/800K_NEW/LiBH4_HT-LiBH4_HT_N_332.xyz-pos-1.xyz',
                   '/home/keli/solid_state/battery/TRAJ/BOROHYDRIDE/MD_450K_constrained/phase1/RESTART/800K_NEW/LiBH4_HT-HT_C_332-pos-1.xyz',
                   '/home/keli/solid_state/battery/TRAJ/BOROHYDRIDE/MD_450K_trs_rot_constrained/parameters/800K_NEW/LiBH4_HT-HT_C_332-pos-1.xyz',
                   # .5fs 800K 
                   '/home/keli/solid_state/battery/TRAJ/BOROHYDRIDE/MD_450K_unconstrained/phase1/rotc_executable/RESTART/800K_NEW/0.5_fs/LiBH4_HT-LiBH4_HT_N_332.xyz-pos-1.xyz',
                   '/home/keli/solid_state/battery/TRAJ/BOROHYDRIDE/MD_450K_constrained/phase1/RESTART/800K_NEW/0.5_fs/LiBH4_HT-HT_C_332-pos-1.xyz',
                   '/home/keli/solid_state/battery/TRAJ/BOROHYDRIDE/MD_450K_trs_rot_constrained/parameters/800K_NEW/0.5_fs/LiBH4_HT-HT_C_332-pos-1.xyz'
                   ]

traj_path_lt_anti = ['/home/keli/solid_state/battery/TRAJ/ANTIPEROVSKITE/MD_800K_unconstrained/cutoff420/Na3OBH4-pos-1.xyz',
        '/home/keli/solid_state/battery/TRAJ/ANTIPEROVSKITE/MD_800K_constrained/cutoff420/Na3OBH4-pos-1.xyz',
        '/home/keli/solid_state/battery/TRAJ/ANTIPEROVSKITE/MD_800K_trs_rot_constrained/Na3OBH4-pos-1.xyz',
        ]#


dump_path_lt = ['test/short10',
                'test/short30000',
                'test/long100000', ]

lat_vec_path = ['/home/keli/solid_state/battery/TRAJ/MD_1200K_unconstrained/phase1/Na11Sn2PS12_normal.inp', 
                '/home/keli/solid_state/battery/TRAJ/BOROHYDRIDE/MD_450K_unconstrained/phase1/LiBH4_HT_normal.inp',
                '/home/keli/solid_state/battery/TRAJ/ANTIPEROVSKITE/MD_800K_unconstrained/cutoff420/Na3OBH4.inp',
                '/home/keli/solid_state/battery/TRAJ/CLOSO/MD_1200K_unconstrained/Na2B10H10.inp',
                '/home/keli/Software/colombo-academy-tutorials/_dp/00.data/RotTest/Li10SiP2S12/unconstrained_1000K/run.inp',
                ]

traj_path_lt_Li10SiP2S12 = ['/home/keli/Software/colombo-academy-tutorials/_dp/00.data/RotTest/Li10SiP2S12/unconstrained_1000K/LSPS.xyz',
    '/home/keli/Software/colombo-academy-tutorials/_dp/00.data/RotTest/Li10SiP2S12/rotc_1000K/LSPS.xyz',
    '/home/keli/Software/colombo-academy-tutorials/_dp/00.data/RotTest/Li10SiP2S12/trs_rotc_1000K/LSPS.xyz',]

system_name_lt = ['UNCONSTRAINED_1200K', 'UNCONSTRAINED_900K', 'UNCONSTRAINED_600K', 'UNCONSTRAINED_300K',
                  'CONSTRAINED_1200K', 'CONSTRAINED_900K', 'CONSTRAINED_600K', 'CONSTRAINED_300K',
                  'trs_rot_constrained_1200K', 'trs_rot_constrained_900K', 'trs_rot_constrained_600K',
                  'trs_rot_constrained_300K']
def print_env_vars():
    for name, r in zip(str_lt, lt):
        print(f'{name}: {r}', sep='  ')

def parse_path(p):
    #re.search(reg,s)
    T = int(re.search(r'\d+',p).group())
    lt = ['constrained', 'unconstrained', 'trs_rot_constrained']
    if 'trs_rot_constrained' in p:
        return T, lt[2]
    elif 'unconstrained' in p:
        return T, lt[1]
    return T, lt[0]

if __name__ == '__main__':
    #print(MolIdxList[1])
    print('don\'t run me!')
