import numpy as np
import warnings

class DCD:
    def __init__(self, file_name=None, timestep=None, merge=False):
        if merge is False and timestep is None:
            raise Exception('need to supply timestep')
        if merge:
            return 
        self.AtomNum = 0
        self.block_size = 0
        self.InfoBlk = 0
        self.timestep = timestep # fs
        self.dipms: np.ndarray = None

        file = open(file_name, 'rb')
        # do not even check the integrity of the file
        InvariantHeader = 4+84+4
        file.read(InvariantHeader)
        # parse the info block
        for i, b in enumerate(reversed(file.read(4))):
            if i == 0:
                assert b == 0
            elif i == 1:
                assert b == 0
            elif i == 2:
                f1, t1 = hex(b)[-2], hex(b)[-1]
                if f1 != 'x':
                    self.InfoBlk+= int(f1,16) * 16 ** 3
                    self.InfoBlk += int(t1,16) * 16 ** 2
                else:
                    self.InfoBlk += int(t1,16) * 16 ** 2
            else:
                self.InfoBlk += b
        file.read(self.InfoBlk+4)
        # the pre-guardian+main_block+post-guardian size
        self.InfoBlk = self.InfoBlk + 8
        AtomBlockSize = 12
        # read the atom num pre-guardian
        file.read(4)
        # parse atom number
        for i, b in enumerate(reversed(file.read(4))):
            if i == 0:
                assert b == 0
            elif i == 1:
                assert b == 0
            elif i == 2:
                f1, t1 = hex(b)[-2], hex(b)[-1]
                if f1 != 'x':
                    self.AtomNum += int(f1,16) * 16 ** 3
                    self.AtomNum += int(t1,16) * 16 ** 2
                else:
                    self.AtomNum += int(t1,16) * 16 ** 2
            else:
                self.AtomNum += b
        # read the post-guardian for atomnum
        file.read(4)
        cell_block_size = 4+48+4
        file.read(cell_block_size)
        # read the blocksize
        for i, b in enumerate(reversed(file.read(4))):
            if i == 0:
                assert b == 0
            elif i == 1:
                assert b == 0
            elif i == 2:
                f1, t1 = hex(b)[-2], hex(b)[-1]
                if f1 != 'x':
                    self.block_size += int(f1,16) * 16 ** 3
                    self.block_size += int(t1,16) * 16 ** 2
                else:
                    self.block_size += int(t1,16) * 16 ** 2
            else:
                self.block_size += b
        file.seek(0)
        # point just right before cell block
        file.read(InvariantHeader+self.InfoBlk+AtomBlockSize)

        frame_dtype = np.dtype([
            ('skip_cell', f'V{cell_block_size+4}'),
            ('x', '<f4', (self.AtomNum,)), 
            ('skip_xy',   'V8'),  
            ('y', '<f4', (self.AtomNum,)),  
            ('skip_yz',   'V8'),      
            ('z', '<f4', (self.AtomNum,)),   
            ('skip_end',  'V4')        
        ])
        raw_data = np.fromfile(file, dtype=frame_dtype)
        traj_frames_first = np.dstack((raw_data['x'], raw_data['y'], raw_data['z']))
        # (AtomNum, FrameNum, 3)
        self._coords = traj_frames_first.transpose(1, 0, 2)
        self.FrameNum = self._coords.shape[1]
        self.TrajLen = self.FrameNum * self.timestep # in fs
        file.close()

    def get_vacf_ensemble(self, len_ens, carrier_list, interval=None):
        """
        NEED TO BE MODIFIED BEFORE USE
        len_ens: how many fs
        carrier_list: index list of selected atom for calculating VACF
        interval: ensemble interval in fs
        """
        if self.timestep > 5:
            warnings.warn('timestep of trajectory might be too large to calculate vel autocorrelation!')
        assert self.TrajLen > len_ens
        if interval is None:
            interval = len_ens
        end = len_ens 
        start = 0
        while end < self.TrajLen:
            yield self._coords[carrier_list,int(start/self.timestep):int(end/self.timestep),:]
            start += interval
            end += interval

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self,value):
        self._coords = value
        self.FrameNum = value.shape[1]

    def get_msd_ensemble(self, len_ens, interval, FrameNum=None):
        """
        len_ens: how many steps
        carrier_list: index list of selected atom for calculating VACF
        interval: ensemble interval in step
        """
        assert hasattr(self, 'dipms')
        assert self.FrameNum > len_ens
        assert len_ens == int(len_ens)
        end = int(len_ens)
        start = 0
        if FrameNum is None:
            FrameNum = self.FrameNum
        while end < FrameNum:
            yield self.dipms[start:end]
            #yield self.dipms[int(start/self.timestep):int(end/self.timestep),:]
            start += interval
            end += interval

    @classmethod
    def merge(cls, dcd_list, step_list):
        """
        merge a list of dcd object into one
        """
        assert len(dcd_list) == len(step_list)
        dcd = cls(merge=True)
        dcd.AtomNum = dcd_list[0].AtomNum
        TotalFrameNum=0
        for i in range(len(dcd_list)):
            #dcd._coords[start:end
            #TotalFrameNum += np.ceil(dcd_list[i].FrameNum/step_list[i])
            TotalFrameNum += np.ceil(dcd_list[i]._coords.shape[1]/step_list[i])
        assert TotalFrameNum == int(TotalFrameNum)
        TotalFrameNum = int(TotalFrameNum)
        dcd._coords = np.zeros((dcd_list[0].AtomNum,TotalFrameNum,3))
        dcd.FrameNum = TotalFrameNum

        start = 0
        end = int(np.ceil(dcd_list[0]._coords.shape[1]/step_list[0]))
        for i in range(len(dcd_list)):
            dcd._coords[:,start:end,:] = dcd_list[i]._coords[:,::step_list[i],:]
            if i == len(dcd_list) - 1:
                continue
            start = end
            end += int(np.ceil(dcd_list[i+1]._coords.shape[1]/step_list[i+1]))
        return dcd
