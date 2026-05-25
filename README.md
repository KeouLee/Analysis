## Analysis
Code reads MD trajectories in DCD (CP2K) or XYZ format and then calculates the free energy curve/surface, reorientation
frequency, and the time-correlation function of angular velocity, etc.

## frequency.py
Compute rotor reorientation frequencies from MD trajectories, either by direct integration (fast rotors) or by an Ivanov model fit (slow rotors).
- Demo instructions

Instructions to run on data:
Edit the user parameters in frequency.py, including TRAJ_LIST, DT_FS, INIT_FRAME_XYZ, DCD_TEMPLATE, DCD_STRIDE, and PS_CUTOFF, so that they point to the input XYZ frame and DCD trajectory files. Then run:

python frequency.py

By default, the script computes the first- and second-order orientational correlation functions for each trajectory. After the C1_traj*.npy and C2_traj*.npy files have been generated, edit the __main__ block to call either fast_rotor() or slow_rotor() to extract the reorientation frequency.

Expected output:
The correlation step generates C1_traj{n}.npy and C2_traj{n}.npy files for each trajectory index n in TRAJ_LIST. The frequency-analysis step prints the extracted relaxation time and reorientation frequency to the terminal. For slow rotors, the script reports tau0, jump angle Delta, and frequency nu, including SEM-based error estimates across trajectories.

Expected run time:
The run time depends on the number and length of DCD trajectories. For a DCD trajectory of approximately 4 GB, the analysis is expected to finish within a few minutes.

## data
The computed mean square (dipole) displacements of six solid electrolytes, along with the script used to compute them.

## defect
The VASP input files for point defect calculations for Na3OBH4, using a sodium vacancy with a charge of -2 as an example.

## metadynamics
The CP2K input files for well-tempered metadynamics simulations of LiBH4, together with the post-processing scripts.

Repo tested with Python 3.10.
