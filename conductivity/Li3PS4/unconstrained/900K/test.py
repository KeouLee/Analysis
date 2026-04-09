from Analysis import DCD, XYZ
import numpy as np
import time
import tensorflow_probability as tfp
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
#from variables import *

DataDict = {'carrier_species': 'Li', 'ts1': 200}

with XYZ('/public/home/lihui_grp/like/solid/nc/Li3PS4/unconstrained/900K/1/init_frame.xyz') as traj:
    P_idx = traj.idx_d['P']
    idx = P_idx
    DataDict['carrier_list'] = [i for i, atom in enumerate(traj.atom_lt) if atom == DataDict['carrier_species']]

for traj in [1,2,3]:
    dcd = DCD(f'{traj}/Li3PS4-pos-1.dcd',200)

    #dcd = DCD.merge([dcd,],[1,])

    com = np.sum(dcd.coords[idx],axis=0)/len(idx)
    com_drift = com - com[0]
    dcd.coords = dcd.coords - com_drift

    calc_collective = False

    if calc_collective:
        dipms = tf.reduce_sum(dcd.coords[DataDict['carrier_list']], axis=0) # (FrameNum, 3)

        nframe = dipms.shape[0]

        print('calculating collective msd...')
        print(f'FrameNum :{dcd.FrameNum}')
        print(f'AtomNum :{dcd.AtomNum}')
        start_time = time.time()
        #len_ens = int(40000)
        len_ens = int(10000)
        msd_array = np.zeros(len_ens)
        count = 0
        for i in range(0, nframe-len_ens+1, 50):
            msd = tf.math.squared_difference(dipms[i:i+len_ens],dipms[i])
            msd = tf.math.reduce_sum(msd, axis=1)
            msd_array += msd
            count += 1
        _length = int(len_ens * 200 / 1e6)
        np.save(f'{traj}/{_length}ns_collective_new.npy', msd_array/count)
            
        #for i, ens in enumerate(dcd.get_msd_ensemble(len_ens, 50)) :
        #    msd = tf.math.squared_difference(ens, ens[0,:])
        #    msd = tf.reduce_sum(msd, axis=1)
        #    msd_array += msd

        #    #msd = self.prefactor * tf.reduce_sum(msd, axis=2)
        #    #self.msd_array += np.array(msd)[0, :]
        #_length = int(len_ens * 200 / 1e6)
        #np.save(f'{_length}ns_collective.npy', msd_array/(i+1))
        #print(i+1)
    else:
        print('single ion msd')
        coords = tf.convert_to_tensor(
        dcd.coords[DataDict['carrier_list']],
        dtype=tf.float64
    )

        # len_ens = int(40000)
        len_ens = int(10000)
        origin_step = 50

        msd_array = np.zeros(len_ens, dtype=np.float64)

        ncarrier = coords.shape[0]
        nframe = coords.shape[1]

        count = 0
        for t0 in range(0, nframe - len_ens + 1, origin_step):
            # ens shape = (Ncarrier, len_ens, 3)
            ens = coords[:, t0:t0+len_ens, :]

            # 相对各自时间原点的位移
            dr = ens - ens[:, 0:1, :]   # shape = (Ncarrier, len_ens, 3)

            # 每个粒子的 |r_i(t)-r_i(0)|^2
            msd = tf.reduce_sum(tf.square(dr), axis=2)   # shape = (Ncarrier, len_ens)

            # 对粒子平均，得到这一组 time origin 的 single-particle MSD
            msd = tf.reduce_mean(msd, axis=0)            # shape = (len_ens,)

            msd_array += msd.numpy()
            count += 1

        msd_array /= count

        _length = int(len_ens * 200 / 1e6)
        np.save(f'{traj}/{_length}ns_single.npy', msd_array)

        print(f'ensemble count = {count}')
