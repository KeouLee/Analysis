"""
frequency.py

Compute orientational reorientation frequencies for polyhedral rotors
(here: PS4 tetrahedra in Na11Sn2PS12) from MD trajectories.

Workflow (two stages):

  1. compute_correlations(traj_num)
        Read a DCD trajectory, identify P-S bond unit vectors, and
        compute the first- and second-order Legendre reorientation
        correlation functions C1(t), C2(t) via FFT.
        Saves C1_traj{n}.npy and C2_traj{n}.npy in the current dir.

  2. Extract the reorientation frequency by ONE of:

        fast_rotor()
            For rapidly reorienting groups whose C(t) decays quickly.
            Integrate C(t) up to the first index where it drops below
            FAST_CUTOFF; frequency = 1/tau. Done independently for
            C1 and C2.

        slow_rotor()
            For slowly reorienting groups. Jointly fit C1 and C2 with
            the Ivanov jump-diffusion model to extract residence time
            tau0 and jump angle delta.

Both methods compute SEM across trajectories from per-trajectory fits.

Configure USER PARAMETERS below, then edit the __main__ block at the
bottom to pick which step to run.
"""

import numpy as np
from scipy.special import legendre
from scipy.optimize import least_squares
from scipy.stats import sem

from Analysis import DCD, XYZ


# ===================== USER PARAMETERS =====================

# Trajectory indices to process.
TRAJ_LIST = [1, 2, 3]

# Time step between successive frames of the saved C1/C2 arrays, in fs.
# This is (MD timestep) * (DCD stride) * (additional stride used here).
# Typical values:
#   fast rotors (dense sampling)  : 1 fs
#   slow rotors (coarse sampling) : 200 fs
DT_FS = 200.0

# --- fast_rotor parameters ---
# Integrate C(t) from t=0 up to the first frame at which C(t) < FAST_CUTOFF.
FAST_CUTOFF = 0.02

# --- slow_rotor (Ivanov fit) parameters ---
# Number of frames of C1/C2 to use in the fit.
SLOW_FIT_FRAMES = 20000
# Initial guess for residence time, fs (will be optimized).
IVANOV_TAU0_INIT_FS = 10e6
# Initial guess for jump angle, degrees (109.0 = tetrahedral).
IVANOV_DELTA_INIT_DEG = 109.0

# --- compute_correlations: trajectory I/O ---
# Path to a frame used to identify PS4 connectivity.
INIT_FRAME_XYZ = (
    'home/Na11Sn2PS12/'
    'unconstrained/1200K/2/init_frame.xyz'
)
# DCD file path; {traj} is replaced by trajectory number.
DCD_TEMPLATE = '{traj}/Na11Sn2PS12-pos-1.dcd'
# Stride used when reading the DCD file (additional subsampling).
DCD_STRIDE = 200
# Distance cutoff for identifying P-S bonds, in Angstrom.
PS_CUTOFF = 2.2

# ===========================================================


# ----------------------- Stage 1: correlations -----------------------

def find_PS4_groups():
    """Return a list of [P_idx, S1, S2, S3, S4] from the initial frame."""
    PS4s = []
    with XYZ(INIT_FRAME_XYZ) as traj:
        for P in traj.idx_d['P']:
            group = [P]
            for S in traj.idx_d['S']:
                if np.linalg.norm(traj.coords[P] - traj.coords[S]) < PS_CUTOFF:
                    group.append(S)
            PS4s.append(group)
    return PS4s


def autocorr_C1_fft(r):
    """C1(t) = <r(0).r(t)> via FFT. r has shape (N_frames, 3)."""
    N = len(r)
    C = np.zeros(N)
    for k in range(3):
        x = r[:, k]
        f = np.fft.fft(x, n=2 * N)
        acf = np.fft.ifft(f * np.conj(f))[:N].real
        acf /= (N - np.arange(N))  # normalize by effective sample count
        C += acf
    return C


def autocorr_C2_fft(r):
    """C2(t) = <P2(r(0).r(t))> via traceless rank-2 tensor autocorr."""
    N = len(r)
    # Five independent components of the traceless symmetric tensor
    # Q_ab = r_a r_b - (1/3) delta_ab.
    Q = np.zeros((N, 5))
    Q[:, 0] = r[:, 0] * r[:, 0] - 1.0 / 3.0
    Q[:, 1] = r[:, 1] * r[:, 1] - 1.0 / 3.0
    Q[:, 2] = r[:, 0] * r[:, 1]
    Q[:, 3] = r[:, 0] * r[:, 2]
    Q[:, 4] = r[:, 1] * r[:, 2]
    # Weights: diagonal 1, off-diagonal 2 (Q_xy = Q_yx).
    w = np.array([1, 1, 2, 2, 2])

    C = np.zeros(N)
    for k in range(5):
        x = Q[:, k]
        f = np.fft.fft(x, n=2 * N)
        acf = np.fft.ifft(f * np.conj(f))[:N].real
        acf /= (N - np.arange(N))
        C += w[k] * acf
    # Add the zz component (Q_zz = -Q_xx - Q_yy from tracelessness).
    zz = -Q[:, 0] - Q[:, 1]
    f = np.fft.fft(zz, n=2 * N)
    acf = np.fft.ifft(f * np.conj(f))[:N].real
    acf /= (N - np.arange(N))
    C += acf
    # Identity: C2 = (3/2) * sum_{ab} <Q_ab(0) Q_ab(t)>.
    return 1.5 * C


def compute_correlations(traj_num):
    """Compute C1 and C2 for one trajectory and save as .npy."""
    PS4s = find_PS4_groups()
    dcd = DCD(DCD_TEMPLATE.format(traj=traj_num), DCD_STRIDE)
    print(f'traj {traj_num} coords shape: {dcd.coords.shape}')

    # Collect unit vectors of all P-S bonds across all PS4 units.
    unit_vecs = []
    for ps4 in PS4s:
        P = dcd.coords[ps4[0]]                  # (N_frames, 3)
        for s in ps4[1:]:
            r = dcd.coords[s] - P               # (N_frames, 3)
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            unit_vecs.append(r)
    unit_vecs = np.array(unit_vecs)             # (N_bonds, N_frames, 3)

    # Average over all bonds.
    C1 = np.mean([autocorr_C1_fft(r) for r in unit_vecs], axis=0)
    C2 = np.mean([autocorr_C2_fft(r) for r in unit_vecs], axis=0)

    np.save(f'C1_traj{traj_num}.npy', C1)
    np.save(f'C2_traj{traj_num}.npy', C2)
    print(f'saved C1_traj{traj_num}.npy, C2_traj{traj_num}.npy')


# ----------------------- Stage 2a: fast rotor -----------------------

def fast_rotor():
    """
    For fast rotors: tau = integral of C(t) up to first crossing of
    FAST_CUTOFF; frequency = 1/tau. Reports C1 and C2 results
    separately. SEM is taken across per-trajectory tau values.
    """
    n_traj = len(TRAJ_LIST)
    for label in ['C1', 'C2']:
        tau_per_traj = np.zeros(n_traj)
        C_sum = None

        for idx, i in enumerate(TRAJ_LIST):
            C_i = np.load(f'{label}_traj{i}.npy')

            if C_sum is None:
                C_sum = C_i.copy()
            else:
                C_sum += C_i

            cut = np.argmax(C_i < FAST_CUTOFF)
            t = np.arange(cut) * DT_FS
            tau_per_traj[idx] = np.trapz(C_i[:cut], t)

        C_avg = C_sum / n_traj
        cut = np.argmax(C_avg < FAST_CUTOFF)
        t = np.arange(cut) * DT_FS
        tau = np.trapz(C_avg[:cut], t)

        sub = label[1]
        print(f'tau{sub}      = {tau:.4g} fs')
        print(f'error bar  = {sem(tau_per_traj):.4g} fs')
        print(f'freq{sub}     = {1e6 / tau:.4g} ns-1')
        print()


# ----------------------- Stage 2b: slow rotor (Ivanov) -----------------------

def _C_ell_ivanov(t, tau0, cos_d, ell):
    Pl = legendre(ell)(cos_d)
    return np.exp(-(1 - Pl) / tau0 * t)


def _ivanov_residuals(params, t, C1_data, C2_data):
    tau0, delta_deg = params
    cos_d = np.cos(np.radians(delta_deg))
    r1 = C1_data - _C_ell_ivanov(t, tau0, cos_d, 1)
    r2 = C2_data - _C_ell_ivanov(t, tau0, cos_d, 2)
    return np.concatenate([r1, r2])


def slow_rotor():
    """
    For slow rotors: jointly fit C1 and C2 to the Ivanov jump-diffusion
    model to extract residence time tau0 and jump angle delta. SEM is
    taken across per-trajectory fits; the reported central value comes
    from fitting the trajectory-averaged C1, C2.
    """
    n_traj = len(TRAJ_LIST)
    tau_per_traj = []
    delta_per_traj = []
    C1_sum = None
    C2_sum = None
    t_fit = None

    for traj_num in TRAJ_LIST:
        C1 = np.load(f'C1_traj{traj_num}.npy')[:SLOW_FIT_FRAMES]
        C2 = np.load(f'C2_traj{traj_num}.npy')[:SLOW_FIT_FRAMES]

        if C1_sum is None:
            C1_sum = C1.copy()
            C2_sum = C2.copy()
            t_fit = np.arange(len(C1)) * DT_FS
        else:
            C1_sum += C1
            C2_sum += C2

        res = least_squares(
            _ivanov_residuals,
            x0=[IVANOV_TAU0_INIT_FS, IVANOV_DELTA_INIT_DEG],
            bounds=([0, 0], [np.inf, 180]),
            args=(t_fit, C1, C2),
        )
        tau0, delta = res.x
        tau_per_traj.append(tau0)
        delta_per_traj.append(delta)

    eb_freq = round(sem(1e6 / np.array(tau_per_traj)), 2)
    eb_tau = round(sem(tau_per_traj) / 1e6, 2)
    eb_delta = round(sem(delta_per_traj), 2)

    # Central value: fit on trajectory-averaged C1, C2.
    res = least_squares(
        _ivanov_residuals,
        x0=[IVANOV_TAU0_INIT_FS, IVANOV_DELTA_INIT_DEG],
        bounds=([0, 0], [np.inf, 180]),
        args=(t_fit, C1_sum / n_traj, C2_sum / n_traj),
    )
    tau0, delta = res.x

    print(f'tau0  = {round(tau0 / 1e6, 2)}±{eb_tau} ns')
    print(f'Delta = {round(delta, 2)}±{eb_delta}°')
    print(f'nu    = {round(1e6 / tau0, 2)}±{eb_freq} /ns')


# ============================ MAIN ============================
# Edit the lines below to choose which step to run.

if __name__ == '__main__':

    # --- Step 1: compute C1, C2 for each trajectory ---
    # Comment out once the .npy files exist.
    for traj_num in TRAJ_LIST:
        compute_correlations(traj_num)

    # --- Step 2: extract frequency. Pick ONE of the following ---
    # fast_rotor()
    # slow_rotor()
