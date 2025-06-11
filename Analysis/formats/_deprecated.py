
    def _get_power_spectrum1(self, 
                 omegamax,
                 domega,
                 unit=None,
                 atom_symbol=None,
                 temperature=None,
                 run_type=None,
                 vacf_type=None,
                 path=None,
                 smooth=True,
                 _dump_pkl=True,
                 _dump_npy=False, 
                 dump_name=None,):
        """
        based on computed `vacf` for a specific atom species, calculating the power spectrum (pdos) now
        it first read the `path` argument if not provided it searches
        the `db/vacf/{atom_symbol}_{temperature}K_{run_type}.pkl
        p(w) = integral{vacf[t] * e^[iwt]}*dt [-inf, inf]
        p(w) = 2 * integral{vacf[t] * cos[wt]} *dt
        """
        vacf = self._setup_ps(path=path, 
                                atom_symbol=atom_symbol, 
                                temperature=temperature, 
                                run_type=run_type, 
                                omegamax=omegamax,
                                domega=domega, 
                                unit=unit,)

        # Fourier Transform of velocity autocorrelation function
        PDOS = []
        for thz in self.THz:
            sumt = complex(0,0)
            for i in range(len(vacf)):
                sumt += complex(vacf[i],) * np.exp(complex(0, -2 * np.pi * i * self.dt * thz)) * self.dt
            PDOS.append(sumt)
        
        pdos = np.array(PDOS).real # length = len(THz)
        if smooth:
            gs = np.zeros(len(pdos))
            nsigma = 8
            sigma = nsigma
            width = 1 / 2 / sigma ** 2
            normfac = 1 / (2 * np.pi) ** .5 * sigma
            for i in range(nomega):
                x0=i*domega
                for j in range(nomega):   # j (0, 1, 2, ... nomega-1)
                    x=j*domega
                    gs[j] += pdos[i] * np.exp(-width*(x-x0)**2) * normfac
                    #gs[j] += pdos[i] * np.exp(-width*(x-x0)**2)

        if smooth:
            d = {'x': THz, 'y': gs}
        else:
            d = {'x': THz, 'y': pdos}
        if _dump_pkl:
            if dump_name is None:
                dump_pkl(f'db/pdos1/{atom_symbol}_{self.temperature}K_{self.run_type}.pkl', d)
            else:
                dump_pkl(Path.cwd() / dump_name, d)

    def _get_pdos1(self, 
                 start_omega=0,
                 stop_omega=190.5,
                 atom_symbol=None,
                 temperature=None,
                 run_type=None,
                 vacf_type=None,
                 path=None,):
        """https://zhuanlan.zhihu.com/p/390774537"""
        try:
            #print(f'db/vacf/{atom_symbol}_{temperature}K_{run_type}{vacf_type}')
            vacf = load_pkl(f'db/vacf/{atom_symbol}_{temperature}K_{run_type}{vacf_type}.pkl')
        except FileNotFoundError:
            if path is None:
                raise FileNotFoundError ("please either provide `atom_symbol` `temperature` `run_type` `vacf_type` to "
                        "read db/vacf/`atom_symbol`_`temperature`K_`run_type``vacf_type`.pkl or specific vacf.pkl path!")
            vacf = load_pkl(path)
        if not (abs(vacf[0] - 1) < 1e-4):
            vacf = vacf / vacf[0]

        omega = np.arange(start_omega, stop_omega, .5)   # omega in THz (angular frequency in THz)
        nu = omega / 2 / np.pi          # nu in THz 
        #self.dt = 0.001

        Nc = len(vacf)
        #vacf = vacf * (np.cos(np.pi * np.arange(Nc)/Nc) + 1) / 2  # window function
        vacf = vacf * np.append(np.ones(1), 2*np.ones(Nc-1))/np.pi

        pdos = np.zeros(len(omega))
        for i in range(len(omega)):
            pdos[i] = dt * sum(vacf * np.cos(omega[i] * np.arange(Nc) * self.dt))  # dt in ps unit
        Np = len(pdos)

        #smooth
        if smooth:
            gs = np.zeros(Np+1)
            n_sigma = 2
            dnu = nu[1] - nu[0]

            sigma = n_sigma * dnu
            wid = 1 / (2 * sigma ** 2) 

            for i in range(Np):
                cp = (i + .5) * dnu
                for j in range(Np+1):
                    bp = j * dnu
                    gs[j] += pdos[i] * np.exp(-wid*(bp-cp)**2)
            return gs
        return pdos

    """deprecated method."""
    def _get_vacf1(self, atom_symbol, correlation_step=20000, step=10, _dump_pkl=True, _dump_npy=False, dump_name=None,):
        """get the velocity autocorrelation function 1 """
        warnings.warn("_get_vacf1 is deprecated, use more accurate get_vacf")
        vel = np.array([v for v in self._get_vel(atom_symbol)])  # (self.FrameNum, self.SelectedAtomNum, 3)
        vacf = np.zeros(correlation_step) # 1d autocorrelation function
        origin_step = (self.FrameNum - correlation_step) // step
        for cs in range(correlation_step):
            for i in range(origin_step):
                vel0 = vel[i * step]
                velt = vel[i * step + cs]
                vacf[cs] += np.sum(vel0 * velt)

        nor_fac = vel.shape[1] * origin_step
        #print(f'nor_fac is {nor_fac}')
        vacf = vacf / vacf[0]

        if _dump_pkl:
            if dump_name is None:
                dump_pkl(f'db/vacf1/{atom_symbol}_{self.temperature}K_{self.run_type}1.pkl', vacf)
            else:
                dump_pkl(Path.cwd() / dump_name, vacf)

    """deprecated method"""
    def _get_vacf2(self, atom_symbol, correlation_step=20000, step=10, _dump_pkl=True, _dump_npy=False, dump_name=None,):
        """ get velocity autocorrelation function2. 
            vacf normalization in this version has some error
            the normalization factors for each discrete value in the VACF(tau) have disparency
            in `get_vacf1` we have also use `vacf/vacf[0]` but some statistical points is traded
            off for simplicity. in get_vacf3 we don't sacrifice some points and correct the normalization factor
        """
        warnings.warn("_get_vacf2 is deprecated, use more accurate get_vacf")
        vel = np.array([v for v in self._get_vel(atom_symbol)])
        #print(vel.shape)
        vacf = np.zeros(correlation_step)
        normal_lt = np.zeros(correlation_step)
        vel_frame_num = self.FrameNum - 1
        
        for origin in range(0, vel_frame_num, step):
            m = min(correlation_step, vel_frame_num - origin)
            v0vt = vel[origin] * vel[origin:m+origin]
            #print(f'vel[{origin}:{m+origin}] {vel[origin:m+origin].shape}')
            #print(f'm = {m}, v0vt shape {v0vt.shape}, origin={origin}')
            vacf[:m] += np.array([np.sum(vv) for vv in v0vt])
            #normal_lt[:m] += 1
        #vacf = vacf / normal_lt
        vacf = vacf / vacf[0]
        if _dump_pkl:
            if dump_name is None:
                dump_pkl(f'db/vacf2/{atom_symbol}_{self.temperature}K_{self.run_type}2.pkl', vacf)
            else:
                dump_pkl(Path.cwd() / dump_name, vacf)

    def _build_mol(self, central_atom, bond_threshold=2.4, bond_lower_threshold=None): 
        """ build molecule based on atom distance. threhold=2.5 angstrom.
        general threhold 2.5, specific threshold depends on running system.
        feel free to adjust the threshold to generate correct molecule.
        now it's implemented for `one central atom, several ligands case only.

        make sure the central atom index is at the first element of the mol list
        """
        pair_lt = []
        for i in range(self.AtomNum):
            for j in range(self.AtomNum):
                if i < j:
                    d=self._get_dist_pair_wrapped(i,j)
                    if d < bond_threshold:
                        pair_lt.append([i,j])
        mol_lt = cat_to_mol(pair_lt) # still of built-in type `list`

        sel_mol_lt = []
        c_idx = [i for i in range(self.AtomNum) if self.atom_lt[i] == central_atom] 
        for c in c_idx:
            for imol, mol in enumerate(mol_lt):
                if c in mol:
                    i=mol.index(c)
                    mol_lt[imol][i], mol_lt[imol][0] = mol_lt[imol][0], mol_lt[imol][i]
                    sel_mol_lt.append(tuple(mol))  # tuple fix 
                    break
        self.sel_mol_lt = copy.deepcopy(sel_mol_lt)
        #return sel_mol_lt

    def _setup_by_induction(self):
        if path is not None:
            self.path = pathlib.Path(path)
        elif (self.name == 'sodium' and run_type == 'trans_constraint'):
            self.path = pathlib.Path(f'/home/keli/solid_state/battery/TRAJ/MD_{run_type}_{temperature}K/concatenate/coords.xyz')
        elif self.name == 'sodium':
            self.path = pathlib.Path(f'/home/keli/solid_state/battery/TRAJ/MD_{temperature}K_{run_type}/concatenate/coords.xyz')
        elif self.name == 'lithium':
            self.path = pathlib.Path(f'/home/keli/solid_state/battery/TRAJ/BOROHYDRIDE/MD_{temperature}K_{run_type}/concatenate/coords.xyz')
        else:
            raise RuntimeError("please provide trajectory path or set the temperature, run_type, system_name to induct the path")
        self.suffix = self.path.name.split('.')[-1]

    def _setup_group(self, segments=None):
        if segments is None:
            lt = [(self.atom_lt[0], 0), ]
            for i in range(len(self.atom_lt) - 1):
                if (self.atom_lt[i] == self.atom_lt[i+1]):
                    lt.append((self.atom_lt[i+1], i+1) )
            ltt = []
            for i, c1 in enumerate(lt):
                for j, c2 in enumerate(lt):
                    if (c1[0] == c2[0] and i != j):
                        ltt.append(i, j) 
    def _get_pair_dist_unwrap(self, c_idx, l_idx):
        # keep the central atom fixed, unwrap the ligand atom only.
        # assume at most only one box away 
        fcc = self._get_fc(c_idx) # not any copied , but a newly created 3-lengthed array in memory.
        fcl = self._get_fc(l_idx)

        for i in range(3):
            d = fcc[i] - fcl[i]
            if d > .5:
                fcl[i] += 1
            elif d < -.5:
                fcl[i] -= 1
        v = fcl - fcc # central atom -> ligand atom
        left=v.reshape((1,3))
        right=v.reshape((3,1))
        r2 = left @ self.Gmat @ right
        return r2 ** .5

    def get_rtcf(self, pair, cutoff, _dump_pkl=True, _dump_npy=False, dump_name=None):
        """
        The statistics performs on a supercell trajectory of 1e5 lasting roughly 2min
        usually you can dump the statistical `residence time correlation function` to .npy or .pkl
        """
        # preparation
        pair_lt = self._setup_pair(pair, cutoff)

        statistic = list()
        FrameNum = 0
        for f in self:
            if not (FrameNum % 10000):
                print(f'calculating {FrameNum//1e3}ps... ')
            count = 0
            for p in pair_lt:
                 d = self._get_dist_pair_wrapped(p[0], p[1])
                 if d < cutoff:
                     count += 1
            statistic.append(count)
            FrameNum += 1
        self.rtcf = np.array(statistic) / statistic[0]

        if _dump_pkl:
            if dump_name is None:
                dump_pkl(f'db/rtcf/{self.temperature}K_{self.run_type}.pkl', self.rtcf)
            else:
                dump_pkl(Path.cwd() / dump_name, self.rtcf)


    def _get_vel(self, atom_symbol):
        """
        `velocity generator`
        get half step kick velocity in step 1 [ v(t + .5dt), v(t + 1.5dt), v(t + 2.5dt) ... ]
        if we have N Frames Then VelFrame generated is (N-1) Frame.  
        WARNINNGS: Unit of the velocity
        PITFALLS: 0-based Frame!
        """
        if self.dt is None:
            raise ValueError("timestep dt is not supplied, it must be supplied to calculate velocity")

        idx_lt = [i for i in range(len(self.atom_lt)) if self.atom_lt[i] == atom_symbol]
        if len(idx_lt) == 0:
            raise ValueError(f'no atom kind {atom_symbol} exists, please check...')

        for i in range(self.FrameNum-1):
            r1 = self[i].coords[idx_lt]
            r2 = self[i+1].coords[idx_lt]
            yield (r2 - r1) / self.dt

    def get_vacf(self, 
                 atom_symbol, 
                 correlation_step=15000, 
                 step=10, 
                 path=None,
                 _dump_pkl=True, 
                 _dump_npy=False, 
                 dump_name=None,):
        """
        `standard method for velocity auto-correlation function calculation`
        calculating the velocity autocorrelation function for a single atom species 
        with ensemble average implemented with statistical maximum.

        atom_symbol :: which atom species you are computing
        correlation_step :: several ps is enough
        step :: step of which you choose the time origin
        path :: where you want to dump your .pkl file if not provided, make sure `$CWD/db/vacf directory exists
        """
        vel = np.array([v for v in self._get_vel(atom_symbol)])
        vel_frame_num = self.FrameNum-1
        VACF = []
        for cs in range(correlation_step):
            v0v0 = 0
            v0vt = 0
            for i in range(0,vel_frame_num,step):
                if (i+cs >= vel_frame_num):
                    break
                velocities_t0 = vel[i]  # ensemble velocities (SeletedAtomNum, 3)
                velocities_t = vel[i+cs] 
                v0v0 += np.sum(velocities_t0 * velocities_t0)
                v0vt += np.sum(velocities_t0 * velocities_t)
            VACF.append(v0vt / v0v0)

        vacf = np.array(VACF)
        if _dump_pkl:
            if dump_name is None:
                dump_pkl(f'db/vacf/{atom_symbol}_{self.temperature}K_{self.run_type}.pkl', vacf)
            else:
                dump_pkl(Path.cwd() / dump_name, vacf)
        
    def _get_angular_velocity(self, central_atom, bond_threshold=2.4, stop=None, normalized=True):
        # w_vec = np.cross(r_vec x v_vec) / np.dot(r_vec, r_vec)
        if self.dt is None:
            raise ValueError("timestep dt is not supplied, it must be supplied to calculate velocity")

        self._build_mol(central_atom, bond_threshold)   # setup the `self.sel_mol_lt` and `sel_mol_lt` attr
        pair_lt = list(itertools.chain(*[uncat(sel_mol) for sel_mol in self.sel_mol_lt]))
        
        for count, p in enumerate(pair_lt):
            if count == stop:
                break                
            if stop is not None:
                print(f'pair {count+1} (in total {stop})')
            else:
                print(f'pair {count+1} (in total {len(pair_lt)})')
            angular_vels = []
            mol = self._get_pair_belong(p) # mol = (104, 105, 106, 107, 108)

            pre_rcom = self._get_rcom(mol)  # mol is a tuple index iterable NOT a `customized mol` object! 
            pre_wrapped_pair_r = self._get_mol_wrapped(p) # coordinates
            pre_wrapped_mol_r = self._get_mol_wrapped(mol) # coordinates

            self._chk_pair_sequence(pre_wrapped_pair_r, pre_rcom, 0, p)
            next(self)
            for i, f in enumerate(self):
                wrapped_mol_r = self._get_mol_wrapped(mol)
                pre_vcom = self._get_vcom(pre_wrapped_mol_r, wrapped_mol_r, mol)

                wrapped_pair_r = self._get_mol_wrapped(p)  
                pre_vel = (wrapped_pair_r[1] - pre_wrapped_pair_r[1]) / self.dt  # previous step ligand velocity
                vk = pre_vel - pre_vcom
                rk = pre_wrapped_pair_r[1] - pre_rcom 

                rcom = self._get_rcom(mol)
                self._chk_pair_sequence(wrapped_pair_r, rcom, i, p)

                angular_vel = np.cross(rk, vk) 
                # angular velocity unnormalized version
                if not normalized:
                    angular_vel = angular_vel / np.dot(rk, rk)

                # angular velocity normalized version
                if normalized:
                    angular_vel = angular_vel / norm(angular_vel) / np.dot(rk, rk)

                angular_vels.append(angular_vel) # (FrameNum * pair_number, 3 )
                #print(angular_vel)

                # update
                pre_rcom = rcom
                pre_wrapped_pair_r = wrapped_pair_r
                pre_wrapped_mol_r = wrapped_mol_r

            # YIELD ANGULAR VELOCITIES FOR A P-S PAIR IN all FRAMES (VelFrameNum, 3)
            yield np.array(angular_vels)

    def get_angular_velocity(self, central_atom, stop=None, normalized=True, dump_path=None, ):
        print('calculating angular velocity...')
        av = np.array([w for w in self._get_angular_velocity(central_atom, stop=stop, normalized=normalized,)])
        if dump_path is None:
            dump_pkl(f'db/wacf/{central_atom}_{self.temperature}K_{self.run_type}_w.pkl', av)
        else:
            dump_pkl(dump_path, av)

    def _get_rcom(self, mol):
        wrapped_coords = self._get_mol_wrapped(mol)
        com = get_com(wrapped_coords, self.mass_lt[list(mol)])
        return com

    def _get_pair_belong(self, pair):
        """find out which molecule the pair belongs to."""
        #print()
        for mol_lt in self.sel_mol_lt:
            if pair[0] in mol_lt:
                return mol_lt

    def _get_vcom(self, pre_wrapped_mol_r, wrapped_mol_r, mol):
        vel = (wrapped_mol_r - pre_wrapped_mol_r) / self.dt
        vcom = get_com(vel, self.mass_lt[list(mol)])
        return vcom

    def _chk_pair_sequence(self, wrapped_pair_coords, com, frame_num, pair_idx):
        """ 
        in our style, wrapped_pair_coords[0] is central atom coordinates which is close to the center of mass of molecule.
        wrapped_pair_coords[1] is ligand.
        """
        b = norm(wrapped_pair_coords[0] - com) < norm(wrapped_pair_coords[1] - com)
        assert b, f'central atom {wrapped_pair_coords[0]}  ligand atom {wrapped_pair_coords[1]} com {com} FrameNum {frame_num}, '\
        f'pair index {pair_idx}'

    def _setup_ps(self, path, atom_symbol, temperature, run_type, omegamax, domega, unit):
        if path is None:
            try:
                vacf = load_pkl(f'db/vacf/{atom_symbol}_{temperature}K_{run_type}.pkl')
            except FileNotFoundError:
                raise FileNotFoundError ("provide specific vacf pkl file path or make sure "
                "db/vacf/{atom_symbol}_{temperature}K_{run_type}.pkl file exists")
        else:
            vacf = load_pkl(path)

        self.THz = np.arange(0, omegamax, domega)
        if (unit is None or unit == 'THz'):
            #by default THz
            pass
        elif unit == 'wave number':
            self.THz *= wavenumber_to_THz
        else:
            raise NotImplementedError(f"{unit} unit not implemented!")
        return vacf

    def get_wacf(self, group_name, correlation_step, interval, dump_name):
        """with ana.XYZL(lat_vec=lat_vec, dt=dt, path=path) as trajl:
            trajl.setup_group({'Mol1': Sequence(P_atom_idx, S1_atom_idx, S2_atom_idx, S3_atom_idx, S4_atom_idx)})
            trajl.get_wacf('Mol1')
        """
        idx_lt = list(self.idx_d[group_name])
        mass_lt = self.mass_lt[idx_lt]

        print('calculating angular velocity...')
        # velocity array generation
        vel = np.array([v for v in self._get_vel(group_name)])
        for v in vel:
            vcom = get_com(v, mass_lt)
            v -= vcom

        # position array generation
        # four lines have been commented out, because we can ignore the central shift in molecule center of mass calculation
        #com_arr = np.zeros((self.FrameNum, 3))
        #for i, f in enumerate(self):
        #    com_arr[i] = get_com(self.coords, self.mass_lt)
        #com_shift = com_arr - self.com0

        #coords = np.zeros((self.FrameNum, len(idx_lt), 3))
        coords = np.zeros((self.FrameNum, len(idx_lt)-1, 3))   # 
        for i, f in enumerate(self):
            #coord = self.coords[idx_lt] - com_shift[i] # central shift correction for your currently inspected molecule.
            coord = self.coords[idx_lt]
            rcom = get_com(coord, mass_lt)
            coord = coord - rcom  # now we have coord under the Molecule Center of Mass.
            coords[i,:,:] = coord

        # angular velocity generation
        coords = coords[1:-1]
        vel = np.cross(coords, vel) # dreamy numpy broadcast.
        #vel = np.delete(vel,0,1) # generation only for the ligand atom.

        div = False
        if div:
            self._get_central_ligand_dist(coords)
        
        def _get_central_ligand_dist(self, coords):
            d = np.zeros(self.FrameNum - 2)
            for m in coords:
                cc = m[0]    # central atom coordinates
                lc = m[1:]    # ligand atom coordinates


        #dump_pkl(dump_name, angular_vels)
        print('calculating wacf...')
        FN = vel.shape[0]
        WACF = []
        for cs in range(correlation_step):
            v0v0 = 0
            v0vt = 0
            for i in range(0,FN,interval):
                if (i+cs >= FN):
                    break
                velocities_t0 = vel[i]  # ensemble velocities (SeletedAtomNum, 3)
                velocities_t = vel[i+cs] 
                v0v0 += np.sum(velocities_t0 * velocities_t0)
                v0vt += np.sum(velocities_t0 * velocities_t)
            WACF.append(v0vt / v0v0)

        wacf = np.array(WACF)
        dump_pkl(dump_name, wacf)
    
    def get_com_vacf2(self, group_lt, pkl_name, ):
        """ensemble averaged and pickle-style com vacf."""
        coords = load_pkl(pkl_name)
        FrameNum = coords.shape[0]
        for count, coord in coords:
            if (count % 5000 == 0):
                print(count // FrameNum * 100, '%')
            for mol in group_lt:
                pass

    def get_wacf(self, group_names, correlation_step, interval, dump_name, drift_correction=True,):
        """with ana.XYZL(lat_vec=lat_vec, dt=dt, path=path) as trajl:
            trajl.setup_group({'Mol1': Sequence(P_atom_idx, S1_atom_idx, S2_atom_idx, S3_atom_idx, S4_atom_idx)})
            trajl.get_wacf('Mol1')
            with ana.XYZL(lat_vec=lat_vec, dt=dt, path=path) as trajl:
                trajl.setup_group({'Mol1': Sequence(P_atom_idx, S1_atom_idx, S2_atom_idx, S3_atom_idx, S4_atom_idx)})
                trajl.setup_group({'Mol2': Sequence(P_atom_idx, S1_atom_idx, S2_atom_idx)})
                trajl.get_wacf(['Mol1','Mol2'])
        """
        unwrap=True
        print(f'drift_correction={drift_correction}')
        print('calculating group average angular velocity...')
        MolNum = len(group_names)
        print(f'MolNum={MolNum}')

        #idx_lt = list(self.idx_d[group_name])
        #mass_lt = self.mass_lt[idx_lt]

        #print(self.coords[idx_lt])
        #print(a:=get_com(self.coords[idx_lt],mass_lt))
        #print(self.coords[idx_lt] - a)
        #exit()
        # velocity array generation
        total_lt=list(chain(*[self.idx_d[gn] for gn in group_names]))
        vel = np.array([v for v in self._get_vel(total_lt,drift_correction)])
        for v in vel:
            vcom = get_com(v, mass_lt)
            v -= vcom
        # position array generation
        coords = np.zeros((self.FrameNum, len(idx_lt)-1, 3))   # for starter we rule the central atom out.
        Dist2 = np.zeros((self.FrameNum, len(idx_lt)-1, 1))  
        for i, f in enumerate(self):
            # \vec(r_k) generation
            coord = self.coords[idx_lt]
            rcom = get_com(coord, mass_lt)
            coord = coord - rcom  # now we have coord under the Molecule Center of Mass.
            lc = coord[1:]
            coords[i,:,:] = lc
            # r_k^2 generation
            for j,c in enumerate(lc):
                Dist2[i,j,0]=np.dot(c,c) 
        # angular velocity generation
        coords = coords[1:-1]
        Dist2 = Dist2[1:-1]
        vel = np.delete(vel,0,1) # generation only for the ligand atom.
        #print(f'coords of ligands (no central shift correction, Molecule com) coords.shape {coords.shape}')
        #print(coords)
        #print(f'Dist2 (from ligands to center of mass) Dist2.shape {Dist2.shape}')
        #print(Dist2)
        #print(f'vel (after central shift correction) vel.shape {vel.shape}')
        #print(vel)
        #exit()
        vel = np.cross(coords, vel) / Dist2# dreamy numpy broadcast.

        #print(vel)
        #exit()
        div = True
        
        #dump_pkl(dump_name, angular_vels)
        print('calculating wacf...')
        FN = vel.shape[0]
        WACF = []
        for cs in range(correlation_step):
            v0v0 = 0
            v0vt = 0
            for i in range(0,FN,interval):
                if (i+cs >= FN):
                    break
                velocities_t0 = vel[i]  # ensemble velocities (SeletedAtomNum, 3)
                velocities_t = vel[i+cs] 
                v0v0 += np.sum(velocities_t0 * velocities_t0)
                v0vt += np.sum(velocities_t0 * velocities_t)
            WACF.append(v0vt / v0v0)

        wacf = np.array(WACF)
        dump_pkl(dump_name, wacf)

    def get_wacf_inertiaPy(self, group_name, correlation_step, interval, dump_name):
        idx_lt = list(self.idx_d[group_name])
        mass_lt = self.mass_lt[idx_lt]
    
        print('calculating total angular velocity...')
        # velocity array generation (of )
        vel = np.array([v for v in self._get_vel(group_name)])
        for v in vel:
            vcom = get_com(v, mass_lt)
            v -= vcom
        #print('vel')
        #print(vel)
        #exit()
        # position array generation
        coords = np.zeros((self.FrameNum, len(idx_lt), 3))   # for starter we rule the central atom out.
        #Dist2 = np.zeros((self.FrameNum, len(idx_lt), 1))  
        for i, f in enumerate(self):
            # \vec(r_k) generation
            coord = self.coords[idx_lt]
            rcom = get_com(coord, mass_lt)
            coord = coord - rcom  # now we have coord under the Molecule Center of Mass.
            coords[i,:,:] = coord 
            #lc = coord[1:]
            ## r_k^2 generation
            #for j,c in enumerate(lc):
            #    Dist2[i,j,0]=np.dot(c,c) 

        # generation2
        coords = coords[1:-1]

        print('calculating angular velocity...')
        angular_vel = np.zeros(vel.shape)
        for i in range(coords.shape[0]):
            # get each frame inertia for a cluster and its inverse.
            coord = coords[i]
            I_inertia=get_inertia(coord, mass_lt)
            I_inv = inv(I_inertia)
            # get each frame angular momentum
            L=np.zeros(3)
            v = vel[i]
            for j in range(coord.shape[0]):
                L += np.cross(coord[j], v[j]) * mass_lt[j]
            angular_vel[i] = np.matmul(I_inv, L)
        
        vel = angular_vel
        print('calculating wacf...')
        FN = vel.shape[0]
        WACF = []
        for cs in range(correlation_step):
            v0v0 = 0
            v0vt = 0
            for i in range(0,FN,interval):
                if (i+cs >= FN):
                    break
                velocities_t0 = vel[i]  # ensemble velocities (SeletedAtomNum, 3)
                velocities_t = vel[i+cs] 
                v0v0 += np.sum(velocities_t0 * velocities_t0)
                v0vt += np.sum(velocities_t0 * velocities_t)
            WACF.append(v0vt / v0v0)

        wacf = np.array(WACF)
        dump_pkl(dump_name, wacf)
from itertools import combinations
from scipy.constants import physical_constants
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from typing import Sequence
from numpy.linalg import norm
from collections import deque
import math

class ShapeError(Exception):
    """indicating wrong shape."""

class RotAnalysis:
    def __init__(self, traj, mol_seq: Sequence, lat_vec: np.ndarray) -> None:
        """Analyze selected one molecule.
        Parameters
        -------------
        traj : trajectory object
        mol_seq : Sequence
            selected molecule index sequence.
    
        Raises
        -------------

        Examples
        -------------
        def usage():
            with Analysis.XYZ(file_path) as traj:
                T = traj[1000:]
                ana_rot = Analysis.RotAnalysis(T, [0,1,2,3])
                theta_arr = ana_rot.get_theta()

            # plotting via Plot class implementation
            Plot.plot_theta(theta_arr)
            
        """
        self.traj = traj
        self.mol_seq = mol_seq
        self.lat_vec = lat_vec
        if lat_vec.shape != (3,3):
            raise ShapeError

    def get_theta(self):
        return self._get_theta().T

    def _get_theta(self):
        theta_arr = []
        for f in self.traj:
            theta_arr.append([])
            host_atom_point = self.traj.coords[self.mol_seq[0]]
            for indice in self.mol_seq[1:]:
                theta_arr[-1].append(self._one_bond_one_frame_theta(host_atom_point, self.traj.coords[indice]))
        return np.array(theta_arr)
    
    def _one_bond_one_frame_theta(self, hap, gap):
        # judge the distance between them
        # scrut - scrutinization
        scrut = [dist > norm(self.lat_vec[i]) / 2 for i, dist in enumerate(abs(hap - gap))]
        if any(scrut):
            self.scrut = scrut
            #in this situation, correction is required.
            gap = self._correct_pos(hap.copy(), gap.copy())
        
        # host_atom_point; guest_atom_point
        # center of mass processing
        atom_vec = np.array(gap) - np.array(hap)

        # z <dotprodcut> atom_vec = |z| | atom_vec| cos(theta)
        # cos(theta) = (z <dotproduct> atom_vec) / |z| |atom_vec|
        cos_theta = np.dot(np.array([0,0,1]), atom_vec) / norm(atom_vec)
        return np.rad2deg(np.arccos(cos_theta))


    def get_phi(self):
        return self._get_phi().T

    def _get_phi(self):
        phi_arr = []
        for f in self.traj:
            phi_arr.append([])
            host_atom_point = self.traj.coords[self.mol_seq[0]]
            for indice in self.mol_seq[1:]:
                phi_arr[-1].append(self._one_bond_one_frame_phi(host_atom_point, self.traj.coords[indice]))
        return np.array(phi_arr)

    def _one_bond_one_frame_phi(self, hap, gap):
        scrut = [dist > norm(self.lat_vec[i]) / 2 for i, dist in enumerate(abs(hap - gap))]
        if any(scrut):
            self.scrut = scrut
            #in this situation, correction is required.
            gap = self._correct_pos(hap.copy(), gap.copy())
        # host_atom_point; guest_atom_point
        # center of mass processing
        atom_vec = np.array(gap) - np.array(hap)

        in_plane = atom_vec[:2]
        x_basis = np.array([1,0])

        # 0 ~ 180 y>0
        if in_plane[1] > 0:
            phi = np.rad2deg(np.arccos(np.dot(x_basis, in_plane) / norm(in_plane)))
            return phi
        # 180 ~ 360 y < 0
        # arccos() -> 180 - rad2deg(arccos()) + 180
        elif in_plane[1] < 0:
            symmetry = np.rad2deg(np.arccos(np.dot(x_basis, in_plane) / norm(in_plane)))
            phi = 180 - symmetry + 180
            return phi
        # 0 or 180 y = 0 
        if in_plane[0] > 0:
            return 0
        else:
            return 180

    def get_bond_angle(self):
        return self._get_bond_angle()

    def _get_bond_angle(self):
        # [1, 2], [1,3] ...
        angle_arr=[]
        for f in self.traj:
            angle_arr.append([])
            for comb in combinations(self.mol_seq[1:], 2):
                angle_arr[-1].append(self._one_angle_one_frame(comb))
        return angle_arr

    def _one_angle_one_frame(self, comb: tuple):
        # center of mass processing
        vec_lt=[]
        host_atom_pos = np.array(self.traj.coords[self.mol_seq[0]])
        for indices in comb:
            guest_atom_pos = self.traj.coords[self.mol_seq[indices]]
            scrut = [dist > norm(self.lat_vec[i]) / 2  for i, dist in enumerate(abs(host_atom_pos - guest_atom_pos))]
            if any(scrut):
                self.scrut = scrut
                guest_atom_pos = self._correct_pos(host_atom_pos, guest_atom_pos)
            vec_lt.append(host_atom_pos - guest_atom_pos)
        # vec_lt -- the container for two com-processed points(vectors)
        bond_angle=np.rad2deg(np.arccos(np.dot(vec_lt[0], vec_lt[1]) / (norm(vec_lt[0]) * norm(vec_lt[1]) )))
        return bond_angle

    def _correct_pos(self, host, guest):
        # but left or right?
        # we need to inspect which direction results in `bond breaking`

        for i, tf in enumerate(self.scrut):
            if tf:
                if host[i] > guest[i]:
                    guest += self.lat_vec[i]
                else:
                    guest -= self.lat_vec[i]
        return guest

    def get_density_plot(self):
        # the theta_arr and phi_arr are regarding the molecule
        return self._get_density_plot()

    def get_density(self):
        return self._get_density()

    def _get_density(self):
        # TA -- theta array; PA -- phi array
        # theta array for the `molecule` meaning that
        # this is not limited to one theta, but all thetas
        # in probed molecule thus TA shaped 
        # ((atom_num - 1), FrameNum)
        
        # natural theta array and phi arr

        # make a `Z` to contain the density
        # I want this grid to be `thin` thus
        # 361 rows and 181 columns.
        # 0-360 degree changing is in fact
        # 361 kinds of change.
        Z: np.ndarray = np.zeros((361, 181))
        theta=self.get_theta()
        phi=self.get_phi()
        # loop over the corresponding `bond and axis angle`
        for i in range(len(self.mol_seq) - 1):
            # this takes out the `theta` and `phi` of each frame.
            for t,p in zip(theta[i], phi[i]):
                # this round takes better than floor
                Z[round(p), round(t)] += 1
        # normalization here for it is density
        Z = Z / norm(Z)
        # round to 2 for the density is good
        return np.round(Z, 2)

    def _get_density_plot(self):
        Z = np.zeros((73, 37))
        theta = self.get_theta()
        phi=self.get_phi()
        for i in range(len(self.mol_seq) - 1):
            for t,p in zip(theta[i], phi[i]):
                Z[round(p/5), round(t/5)] += 1

        # normalization here
        Z = Z / norm(Z)
        return np.round(Z, 2)

    @staticmethod
    def continued_multiply(Sequence):
        total=1
        for ele in Sequence:
            total *= ele
        return total

    def get_Helmholtz(self, T=1200):
        return np.round(self._get_Helmholtz(T), 1)

    
    def _get_Helmholtz(self, T=1200):
        theta=self.get_theta()
        phi=self.get_phi()
        density=self.get_density()
        density = density + 0.001
        kB=physical_constants["Boltzmann constant in eV/K"][0]
        A = -kB * T * np.log(density)
        return A
        pass


class RotPlot:
    @classmethod
    def plot_theta(cls, theta_arr, show=True, save=False, save_path=None):
        fig, ax = plt.subplots()
        for theta in theta_arr:
            ax.plot(theta)
        if show:
            plt.show()
        elif save:
            if save_path is None:
                raise NameError("please specify the figure saving path")
            plt.savefig(save_path, format="png", dpi=1200)

    @classmethod
    def plot_phi(cls, phi_arr, show=True, save=False, save_path=None):
        """
        with Analysis.XYZ(filename) as traj:
            T=traj[100:]
            rot_ana = Analysis.RotAnalysis(T, [188, 189, 190, 191, 192])
            RotPlot.plot_phi(rot_ana.get_phi(T, mol_seq), show=False, save=True, save_path="tmp.png")
            >>>the plot was completed.
        """
        fig, ax = plt.subplots()
        for phi in phi_arr:
            modified_phi=cls._eliminate_jump(phi)
            ax.plot(modified_phi)
        if show:
            plt.show()
        elif save:
            if save_path is None:
                raise NameError("please specify the figure saving path")
            plt.savefig(save_path, format="png", dpi=1200)

    @staticmethod
    def _eliminate_jump(one_phi_multi_frames: np.ndarray):
        # written for phi adjustment
        #alias it
        phis=one_phi_multi_frames
        FrameNum = len(phis)
        offset = [i for i in range(len(phis)-1) if abs(phis[i+1] - phis[i]) > 350]
        
        # this is just the offsets for one phi
        # [.....] 1d
        if offset:
            offset = RotPlot._group_offset(offset)
        # after this `_group_offset` call offset is now 2d
        # single jump is necessary and natural thus excluding it
        offset = [ele for ele in offset if len(ele) > 1]

        for f in offset:
            if abs(f[-1] - FrameNum) > 300:
                end = sum(phis[f[-1]:f[-1] + 100] > 180)
                eend = sum(phis[f[-1]:f[-1] + 100] < 180)
        
                if end > 95:
                    # flip up
                    for i in range(f[0],(f[-1] + 1)):
                        if phis[i] <180:
                            phis[i] += 360
                               
                elif eend > 95:
                    # flip down
                    for i in range(f[0],(f[-1] + 1)):
                        if phis[i] > 180:
                            phis[i] -= 360
            else:
                start = sum(phis[f[0]:f[0]+100] > 180)
                if start > 95:
                    #flip up
                    for i in range(f[0], (f[-1] + 1)):
                        if phis[i] < 180:
                            phis[i] += 360
                else:
                    for i in range(f[0], (f[-1] + 1)):
                        if phis[i] > 180:
                            phis[i] -= 360
        return phis
        
    @staticmethod
    def _group_offset(offset):
        # record the start point of the different jump
        recorder=deque([i+1 for i in range(len(offset) - 1) if abs(offset[i+1] - offset[i]) > 1000])
        recorder.appendleft(0)

        lt=list()
        for i, ele in enumerate(offset):
            if i in recorder:
                lt.append([])
                lt[-1].append(ele)
            else:
                lt[-1].append(ele)
        return lt

    @classmethod
    def plot_density_contourf(cls, Z, show=True, save=False, save_path=None):
        #plt.style.use('_mpl-gallery-nogrid')
        x_linear = np.linspace(0, 180, num=37)
        y_linear = np.linspace(0, 360, num=73)
        X, Y = np.meshgrid(x_linear, y_linear)

        fig, ax = plt.subplots()
        ax.contourf(X, Y, Z)
        
        if show:
            plt.show()
        elif save:
            if save_path is None:
                raise NameError("please specify the figure saving path")
            plt.savefig(save_path, format="png", dpi=1200)
    
    @classmethod
    def plot_density_pcolormesh(cls, Z, show=True, save=False, save_path=None):
        x_linear = np.linspace(0, 180, num=181)
        y_linear = np.linspace(0, 360, num=361)
        X, Y = np.meshgrid(x_linear, y_linear)

        fig, ax = plt.subplots()
        ax.pcolormesh(X, Y, Z)

        if show:
            plt.show()
        elif save:
            if save_path is None:
                raise NameError("please specify the figure saving path")
            plt.savefig(save_path, format="png", dpi=1200)

    @classmethod
    def plot_density_imshow(cls, Z, show=True, save=False, save_path=None):
        #x_linear = np.linspace(0, 180, num=181)
        #y_linear = np.linspace(0, 360, num=361)
        #X, Y = np.meshgrid(x_linear, y_linear)

        fig, ax = plt.subplots()
        ax.imshow(Z)

        if show:
            plt.show()
        elif save:
            if save_path is None:
                raise NameError("please specify the figure saving path")
            plt.savefig(save_path, format="png", dpi=1200)

    #@classmethod
    #def plot_density(cls, Z, show=True, save=False, save_path=None):
    #    plt.style.use('_mpl-gallery-nogrid')
    #    x_linear = np.linspace(0, 180, num=33)
    #    y_linear = np.linspace(0, 360, num=66)
    #    X, Y = np.meshgrid(y_linear, x_linear)

    #    fig, ax = plt.subplots()
    #    ax.contourf(X, Y, Z)
    #    
    #    if show:
    #        plt.show()
    #    elif save:
    #        if save_path is None:
    #            raise NameError("please specify the figure saving path")
    #        plt.savefig(save_path, format="png", dpi=1200)
    #@classmethod
    #def plot_density(cls, Z, show=True, save=False, save_path=None):
    #    plt.style.use('_mpl-gallery-nogrid')
    #    x_linear = np.linspace(0, 180, num=33)
    #    y_linear = np.linspace(0, 360, num=66)
    #    X, Y = np.meshgrid(y_linear, x_linear)

    #    fig, ax = plt.subplots()
    #    ax.contourf(X, Y, Z)
    #    
    #    if show:
    #        plt.show()
    #    elif save:
    #        if save_path is None:
    #            raise NameError("please specify the figure saving path")
    #        plt.savefig(save_path, format="png", dpi=1200)

    @classmethod
    def plot_A(cls, A, show=True, save=False, save_path=None):
        
        x_linear = np.linspace(0, 180, num=181)
        y_linear = np.linspace(0, 360, num=361)
        X, Y = np.meshgrid(x_linear, y_linear)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        surf = ax.plot_surface(X, Y, A, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        if show:
            plt.show()
        elif save:
            if save_path is None:
                raise NameError("please specify the figure saving path")
            plt.savefig(save_path, format="png", dpi=1200)

def cat_to_mol(lt):
    """
    notes
    ---------
    based on common index for a list of bonding atom index to form a `mol_idx`
    """
    ltt = []
    for p in lt:
        IN, idx, which = find_pair_any(p, ltt)
        if IN:
            other = list({0,1} - {which})[0]
            ltt[idx].append(p[other])
        elif not IN:
            ltt.append([])
            ltt[-1].append(p[0])
            ltt[-1].append(p[1])
    return ltt

def find_pair_any(pair, lt):
    IN=False
    idx=None
    which=None
    for i, sub_lt in enumerate(lt):
        for j in range(2):
            if (pair[j] in sub_lt):
                idx = i
                IN=True
                which=j
                break

    return IN, idx, which

    def _check_and_count(self):
        # for particle number fixed system (NPT, NVT, NVE, ...)
        #check every frame by view
        count = 0

        while True:
            count += 1
            line = self.f.readline()
            # eof
            if line == "":
                #self._offsets.append(self.f.tell())
                break
            elif not count % self.LinesPerFrame:
                try:
                    int (line)
                    self._offsets.append(self.f.tell())
                except ValueError:
                    #"input xyz file. Use inspect method to check defect line and Nr. of defect frame")
                    #raise RuntimeError(f":at lineNr{count}: FrameNum and actual traj are "
                    #                    "not in accordance, please check the "
                    #"input xyz file.")
        #print(f"count = {count}")
        FrameNum = count / self.LinesPerFrame
        if FrameNum == int(FrameNum):
            self.FrameNum = int(FrameNum)
        else:
            raise RuntimeError ("error occurs when calc FrameNum of the system!")

    def inspect(self, defect_list=False):
        """ now you have self.AtomNum thus the LinePerFrame.
        Now inspect from each `time` to another `time the framelength"""
        count = 1
        DefectNum=0
        LinesRec = list()
        rec = 0

        self.f.readline()
        while (line := self.f.readline()):
            count += 1
            if re.match(XYZ.atomnum_part, line):
                if (rec != self.AtomNum + 1):
                    DefectNum += 1
                    LinesRec.append(count)
                rec = 0
                continue
            rec += 1

        if defect_list:
            print(DefectNum, LinesRec)
            return DefectNum, LinesRec
        print(DefectNum)
        return DefectNum


class XYZ(FormatBase):
    """read xyz file or zipped xyz file and return 
    the XYZ object representing the trajectory."""

    atomnum_part = r"\s*\d+\s+"
    cmt_part = r"[^\n]*\n"
    coords_part = r"\s*[A-Z][a-z]*\s+[\.0-9Ee\-\+]+\s+[\.0-9Ee\-\+]+\s+[\.0-9Ee\-\+]+\s+"

    atomnum_part_mmap = rb"\s*\d+\s+"
    cmt_part_mmap = rb"[^\n]*\n"
    coords_part_mmap = rb"\s*[A-Z][a-z]*\s+[\.0-9Ee\-\+]+\s+[\.0-9Ee\-\+]+\s+[\.0-9Ee\-\+]+\s+"
    #def __init__(self, path, traj_file_name=None):
    #    super().__init__(path, traj_file_name)
    #    self.flag=False

    def _read_info(self):    
        #coords_part = r"(\s*[A-Z][a-z]*\s+[\.0-9\-]+\s+[\.0-9\-]+\s+[\.0-9\-]+\s+)+"
        # you should assert `AtomNum` on the first line of the file...

        self.Cartesian=True
        self._offsets = []
        self.AtomNum = int(self.f.readline())
        self.f.seek(0)
        self.LinesPerFrame = self.AtomNum + 2

        self._offset = [0,]
        OneFrame = list(islice(self.f, self.LinesPerFrame))
        # from first frame we get the `self.atom_lt`
        self.atom_lt = [line.split()[0]  for line in OneFrame[2:]]

        while OneFrame:
            self._offset.append(len(tuple(chain.from_iterable(OneFrame))))
            OneFrame = list(islice(self.f, self.LinesPerFrame))
        self._offset = list(accumulate(self._offset))
        self.FrameNum = len(self._offset) - 1
         
        #cmt part
        # just the next behind the AtomNum is the cmt line


    def __len__(self):
        return self.FrameNum
    def _get_one_frame(self):
        coords_lt = []
        element_lt = []
        
        while True:
            line = self.f.readline()
            #print(line)
            if self.flag:
                return 
            elif re.match(XYZ.atomnum_part, line):
                self.atom_lt = np.array(element_lt)
                self.coords = np.array(coords_lt).reshape(self.AtomNum, 3).astype(np.float32)
                #if self.mol_div:
                #    self.get_mol_div()
                return self.AtomNum
            elif (match_res := re.match(XYZ.coords_part, line)):
                lt = match_res.group().split()
                coords_lt += lt[1:]
                element_lt += lt[:1]
            elif line == '':
                self.atom_lt = np.array(element_lt)
                self.coords = np.array(coords_lt).reshape(self.AtomNum, 3).astype(np.float32)
                #if self.mol_div:
                #    self.get_mol_div()
                self.flag = True
                return self.AtomNum
            else:
                #print(line)
                self.annotations.append(line)
        
    def _get_one_frame_mmap(self):
        coords_lt = []
        element_lt = []
        
        while True:
            line = self.f.readline()
            #print(line)
            if self.flag:
                return 
            elif re.match(XYZ.atomnum_part_mmap, line):
                self.atom_lt=np.array([ele.decode('utf-8') for ele in element_lt])
                
                self.coords=np.array([coord.decode('utf-8') for coord in coords_lt]).reshape(self.AtomNum,3).astype(np.float32)
                #self.coords = np.array(coords_lt).reshape(self.AtomNum, 3).astype(np.float32)
                #if self.mol_div:
                #    self.get_mol_div()
                return self.AtomNum
            elif (match_res := re.match(XYZ.coords_part_mmap, line)):
                lt = match_res.group().split()
                coords_lt += lt[1:]
                element_lt += lt[:1]
            elif line.decode('utf-8') == '':
                self.atom_lt = np.array(element_lt)
                self.coords = np.array(coords_lt).reshape(self.AtomNum, 3).astype(np.float32)
                #if self.mol_div:
                #    self.get_mol_div()
                self.flag = True
                return self.AtomNum
            else:
                #print(line)
                self.annotations.append(line)

    def _check_and_count(self):
        # calc `FrameNum` based on Nr of `atomnum_part`
        # also modify the self._offsets by your record of `atomnum_part`
        DefectNum = 0
        self.f.seek(self._offsets[0])
        rec = 0
        count = 1
        LinesRec = []
        FrameNum = 1
        FrameNumLt = []
        #count = self._FirstAtomnumLineNr

        while (line := self.f.readline()):
            count += 1
            if re.match(XYZ.atomnum_part, line):
                self._offsets.append(self.f.tell())
                if (rec != self.AtomNum +1):
                    DefectNum += 1
                    LinesRec.append(count+self._offsets[0])
                    FrameNumLt.append(FrameNum)
                FrameNum += 1
                rec = -1
            rec += 1

        self.FrameNum = len(self._offsets)
        if (DefectNum != 0):
            logging.warning(f'Defect Nr of Frame {DefectNum} at line{LinesRec} for Frame {FrameNumLt}')

    def _check_and_count_mmap(self):
        # calc `FrameNum` based on Nr of `atomnum_part`
        # also modify the self._offsets by your record of `atomnum_part`
        DefectNum = 0
        self.f.seek(self._offsets[0])  # works the same for the mmap object
        rec = 0
        count = 1
        LinesRec = []
        FrameNum = 1
        FrameNumLt = []
        #count = self._FirstAtomnumLineNr

        while (line := self.f.readline()):
            count += 1
            if re.match(XYZ.atomnum_part_mmap, line):
                self._offsets.append(self.f.tell())
                if (rec != self.AtomNum +1):
                    DefectNum += 1
                    LinesRec.append(count+self._offsets[0])
                    FrameNumLt.append(FrameNum)
                FrameNum += 1
                rec = -1
            rec += 1

        self.FrameNum = len(self._offsets)
        if (DefectNum != 0):
            logging.warning(f'Defect Nr of Frame {DefectNum} at line{LinesRec} for Frame {FrameNumLt}')

class FormatBase(metaclass=ABCMeta):
    __supported = ( "xyz", "xyzl", "POSCAR")
    axes_dict = {'x': 0, 'y': 1, 'z': 2}

    def __init__(self, 
                 path=None, 
                 temperature=None,
                 run_type=None,
                 system_name=None,
                 flag=False, 
                 slicing=None, 
                 from_stream=False,
                 AtomNum=None,
                 FrameNum=None,
                 coords=None,
                 atom_lt=None,
                 dt=None,
                 MolIdx=None,
                 memmap=False,
                 details=False,
                 ):
        """
        arguments
        ------------
        path :: trajectory file path
        temperature :: system temperature if it is a canonical ensemble
        run_type :: unconstrained or rotc or rot_trsc
        system_name :: 'sodium' for Na11Sn2PS4 'lithium' for LiBH4
        flag :: internal parameter controlling trajectory reading
        slicing :: internal parameter controlling trajectory slicing
        from_stream :: internal parameter controllinng instantiation
        AtomNum :: number of system atom
        FrameNum :: Frame Number
        coords :: coordinates of the atom
        atom_lt :: atom symbol list
        dt :: timestep in ps

        """
        self._from_stream = from_stream
        if from_stream:
            self._from_file = False
            self._setup_from_stream(AtomNum, FrameNum, coords, atom_lt)
            return 

        self._from_file = True
        self._memmap:bool=memmap
        self.flag=flag
        if slicing:
            self.start: None|int = slicing.start
            self.stop: None|int = slicing.stop
            self.step: None|int = slicing.step
            self.count=0
            
            return 
            
        self._setup(path, temperature, run_type, system_name, dt, MolIdx, details)
    
    def _setup(self, path, temperature, run_type, system_name, dt, MolIdx, details):
        """
        run_type = ['unconstrained', 'constrained', 'trs_rot_constrained', 'trans_constraint']
        name = 'sodium' or 'lithium'
        T = 450, 300, 600, 900, 1200
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found, please check again.")
        print(path)
        self.path = path
        self.T = temperature
        self.run_type = run_type
        self.name = system_name
        self.dt = dt
        self.MolIdx = MolIdx
        self.file_suffix = str(path).split('/')[-1].split('.')[-1]
        self.details = details

    @abstractmethod
    def _get_one_frame(self):
        """traj_fo is the trajectory file 
        object waiting to be analyzed."""

    @abstractmethod
    def _read_info(self, ):
        """read trajectory information."""

    @abstractmethod
    def _check_and_count(self):
        """check trajectory FrameNum and LineNr correspondence, read trajectory Frame Number."""

    @abstractmethod
    def _get_one_frame_mmap(self):
        """mmap version."""

    @abstractmethod
    def _check_and_count_mmap(self):
        """mmap version."""

    def __enter__(self):
        if not self.file_suffix in FormatBase.__supported:
            raise NotImplementedError(
                    f"supported formats now are {', ' .join(FormatBase.__supported)}")
        
        f = open(self.path, 'r')  # open the trajectory file for the first time and points to the first record.
        if self._memmap:
            self.f = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
            f.close()
        else:
            self.f = f
        #st=time.time() 
        self._read_info()
        #print('_read_info', time.time() - st)
        self._setup_mass()
        self._setup_group()

        #if details:
        ##st=time.time()
        #if self._memmap:
        #    self._get_one_frame_mmap()
        #else:
        #    self._get_one_frame()
        ##print('_get_one_frame', time.time() - st)
        ## here we already read one frame but in this case in
        ## the `for` loop we start from the second frame.
        ## but we still want to start with the first frame.
        #self.f.seek(self._offsets[0])

        ##if self._chk_traj:
        ##print('counting and checking...')
        ##st = time.time() 
        #if self.details:
        #    if self._memmap:
        #        self._check_and_count_mmap()
        #    else:
        #        self._check_and_count()
        ##print('_check_and_count', time.time() - st)

        #self.f.seek(self._offsets[0]) # anyway, we go back to the first AtomNum line
        #self.flag=False # set to False in case OneFrame scenario.
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()

    def __next__(self):
        # not slicing -- a complete file traj object
        if not hasattr(self, "start"):
            AtomNum=self._get_one_frame()
            #print(AtomNum)
            #print(self.flag)
            if AtomNum is None:
                # to run a trajectory initialization here!
                self.flag = False
                #self.f.seek(0)    
                #self._read_info()
                self.f.seek(self._offsets[0])
                raise StopIteration
        elif hasattr(self, "start"):
            self.flag=False
            if self.start >= self.stop:
                raise IndexError

            which_frame = self.count * self.step + self.start 
            #print(f"self.stop is {self.stop}")
            #print(f"self.start is {self.start}")
            #print(f"self.step is {self.step}")
            #
            #print(f"which frame is {which_frame}")
            if which_frame >= self.stop:
                self.f.seek(self._offsets[self.start])
                self.count=0
                raise StopIteration

            
            res=self.f.seek(self._offsets[which_frame])
            #print(self.f.tell())
            #print(self.count)
            AtomNum=self._get_one_frame()
            self.count += 1
            
            #if AtomNum is None:
            #    self.flag=False
            #    self.f.seek(self._offsets[start])
            #    self._read_info()
            #    raise StopIteration

    def __iter__(self):
        return self

    @classmethod
    def gen(cls, AtomNum, FrameNum, coords, atom_lt, dt=None, slicing=None, cartesian=True):
        #bypass the file-related `trajetory` object generation, automatic generation via 
        #info supplemented.
        return cls(from_stream=True, AtomNum=AtomNum, FrameNum=FrameNum, 
                   coords=coords, atom_lt=atom_lt, dt=dt, slicing=slicing)

    def __getitem__(self, indice):
        """The usage of traj_object is, e.g., 
        `with ReadXYZ(filename) as traj:
            pass
            traj_50 = traj[50] # works
            # this returns a traj object with start, stop and step attribute
            # the implementation of `for` loop will just looping over this slice.
            traj_200_300 = traj[200:300] # works
            for f in traj_200_300:
                pass
            traj_300_2=traj[:300:2] # works
        Traj=traj[:22] # out of the context manager thus fails.
        """
        if isinstance(indice, numbers.Integral):
            if indice >= self.FrameNum:
                raise IndexError("index out of current frame range")
            start_char = self._offsets[indice]
            #end_char = self._offsets[indice+1]
            # this lead to the beginning of the cmt line
            self.f.seek(start_char)            
            if self._memmap:
                self._get_one_frame_mmap()
            else:
                self._get_one_frame()
            traj = self.__class__.gen(self.AtomNum, 1, self.coords, self.atom_lt, self.dt, slicing=indice, cartesian=True)
            if hasattr(self, 'lat_vec'):
                traj.lat_vec = self.lat_vec
            self.f.seek(self._offsets[0])
            if (self.FrameNum -1 == indice):
                self.flag = False
            return traj

        elif isinstance(indice, slice):
            # creating new traj object with marked `start`
            # `stop` and `step` attr
            traj=self.__class__(slicing=indice,)
            # then simply verify the three input indices
            for s in (traj.stop, traj.start, traj.step):
                if not (s is None or isinstance(s, numbers.Integral)):
                    raise TypeError("slice object elements must be None or Integral")

            if traj.stop is not None:
                if traj.stop > self.FrameNum:
                    raise IndexError("index out of range")

            elif (traj.start is not None and traj.start < 0):
                raise ValueError

            elif (traj.step is not None and raj.step < 1):
                raise ValueError
            # now the both pointing to the self.f object 
            # when self.f object is closed, traj.f object is close
            # as well!
            traj.f = self.f
            traj._offsets = self._offsets   
            traj.atom_lt = self.atom_lt
            #traj.FrameNum
            traj.AtomNum = self.AtomNum
            traj.annotations = self.annotations
            traj.dt = self.dt

            traj.start = 0 if traj.start is None else traj.start
            traj.stop = self.FrameNum if traj.stop is None else traj.stop
            traj.step = 1 if traj.step is None else traj.step

            traj.f.seek(traj._offsets[traj.start])
            traj._memmap=self._memmap
            if traj._memmap:
                traj._get_one_frame_memmap
            else:
                traj._get_one_frame()

            res=traj.f.seek(traj._offsets[traj.start])
            # (((stop - 1) - start) + 1) / step -> floor
            traj.FrameNum = math.floor((traj.stop - traj.start) / traj.step)
            if hasattr(self, 'lat_vec'):
                traj.lat_vec = self.lat_vec
            return traj
        else:
            raise TypeError("input must be of class 'slice' or 'int'")

    def _setup_from_stream(self, AtomNum, FrameNum, coords, atom_lt):
        if AtomNum is None:
            raise ValueError ("must supply atom number in the trajectory")
        if FrameNum is None:
            raise ValueError ("must supply frame number in the trajectory")
        if coords is None:
            raise ValueError ("must supply atom coordinates in the trajectory")
        if atom_lt is None:
            raise ValueError ("must supply atom coordinates in the trajectory")
        self.AtomNum = AtomNum
        self.FrameNum = FrameNum
        self.coords = coords
        self.atom_lt = atom_lt
        self._setup_group()

    def write_to_xyz(self, atom_labels: Sequence, filepath, digit=8):
        """select the frame and wanted atoms you want to write to the xyz file.
        examples
        might with the utility of reducing digit
        ------------
        atom_labels: [0, 1, 3] this three atoms.
        with Analysis.XYZ(file_name) as traj:
            T = traj[200:]
            T.write_to_xyz([0, 1, 3])
        digit :: how many digit you want to have for coordinates
            
        """
        if isinstance(atom_labels, int):
            assert (atom_labels == self.AtomNum)
            atom_labels = np.array(range(atom_labels))
    
        with open(filepath, "w") as file:
            if self._from_file:
                for frame in self:
                    file.write(f"    {str(len(atom_labels))}\n")
                    file.write(self.annotations[-1])
                    for label in atom_labels:
                        file.write(f"{self.atom_lt[label]}    ")
                        coords=str(np.round(self.coords[label],digit)).strip("[] ")
                        file.write(coords)
                        file.write("\n")
            elif self._from_stream:
                #3d (self.FrameNum, self.AtomNum, 3)
                for i in range(self.FrameNum):
                    file.write(f"    {str(len(atom_labels))}\n")
                    file.write(self.annotations[-1])
                    for label in atom_labels:
                        file.write(f"{self.atom_lt[label]}    ")
                        coord=str(self.coords[i][label]).strip("[] ").split() # ['x', 'y', 'z']
                        for c in coord:
                            c = '{:.8f}'.format(float(c))
                            file.write('{:<17}'.format(c))
                        file.write("\n")

            #elif self._from_stream:
            #    #2d coordinates array 
            #    for i in range(self.FrameNum):
            #        new_labels = i * self.AtomNum + atom_labels
            #        file.write(f"    {str(len(atom_labels))}\n")
            #        file.write(f"{cmt}\n")
            #        for label in new_labels:
            #            file.write(f"{self.atom_lt[label]}    ")
            #            coords=str(self.coords[label]).strip("[] ")
            #            file.write(coords)
            #            file.write("\n")

    def _setup_mass(self):
        self.mass_lt = np.array([atomic_masses[chemical_symbols.index(symbol)] for symbol in self.atom_lt])
        #self.com0 = get_com(self.coords, self.mass_lt) 
        # note that in this case we first calculate com and unwrap/wrap the atom

    def _setup_group(self):
        idx_atom_lt = [(atom, i) for i, atom in enumerate(self.atom_lt)]
        self.idx_d = defaultdict(list)
        for k, v in idx_atom_lt:
            self.idx_d[k].append(v)
        self.type_lt = np.array([k for k in self.idx_d.keys()])

    def setup_group(self, idx_d):
        """
        with ana.XYZL(path=path, lat_vec=lat_vec) as trajl:
            trajl.setup_group({'S1': [1,2,3], 'S2':[4,5,6]})
        """
        for k, v in idx_d.items():
            if k in self.idx_d:
                raise ValueError('errors happened when trying to modify original system-self-generated group dictionary')
            self.idx_d[k] = v

    def write_to_poscar(self, dump_name=None, Cartesian=True, frame=0):
        if (not hasattr(self, 'lat_vec')):
            raise AttributeError('need lattice info to write to POSCAR file')
        if dump_name is None:
            if Path('POSCAR').exists():
                raise RuntimeError('POSCAR already exists, you can pass in new name you want to dump')
            dump_name = 'POSCAR'

        poscar_dict=same_cat_pair(self.atom_lt)
        if self.file_suffix.lower() != 'poscar': 
            set_lt=[]
            for ele in self.atom_lt:
                if ele in set_lt:
                    continue
                set_lt.append(ele)
                idx = []
                for ele in set_lt:
                    for i, eele in enumerate(self.atom_lt):
                        if ele == eele:
                            idx.append(i)

        file = open(dump_name, 'w')
        if self.name is None:
            self.name = ' '
        file.write(f'{self.name.strip()} POSCAR\n')
        file.write('1.0\n') # scale factor we here set as 1.0
        for vec in self.lat_vec:
            vec_str=str(vec).strip('[] ')
            file.write(f'\t{vec_str}\n')
        for k in poscar_dict.keys():
            file.write(f'{k}  ')
        file.write('\n')
        for v in poscar_dict.values():
            file.write(f'{v}  ')
        file.write('\n')
        if self.file_suffix.upper() == 'POSCAR':
            idx_coords = self.coords 
        else:
            idx_coords = self[frame].coords[idx] # correctly sorted for POSCAR output
        if Cartesian:
            file.write('Cartesian\n')
        else:
            file.write('Direct\n')
            idx_coords = cc_to_fc(idx_coords, lat_vec=self.lat_vec)
        # poscar is one frame object
        for coords in idx_coords:
            coord = str(coords).strip('[] ').split()
            file.write('\t')
            for c in coord:
                c = '{:.8f}'.format(float(c))
                file.write('{:<15}'.format(c))
            file.write('\n')
    
