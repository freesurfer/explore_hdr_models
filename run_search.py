"""
Run grid search to find best fitting delta, tau, and alpha using the canonical hemodynamic model:
    h(t>delta)  = ((t-delta)/tau)^alpha * exp(-(t-delta)/tau)
    h(t<=delta) = 0

TODO:
* run on purely synthetic data
* noise-effect testing
* get p-value for best
* pick out rois
"""
import math
import numpy as np
import os
import pandas as pd
import surfa as sf
import time

from feeg_fmri_sync import SEARCH_TYPES
from feeg_fmri_sync.constants import MLSC_ROOT_DIR, URSA_ROOT_DIR, PROJECT_DIR, EEG_DIR, FMRI_DIR, NUMBER_OF_TASKS
from feeg_fmri_sync.models import EEGData, fMRIData
from feeg_fmri_sync.search import search_voxel


### Inputs
hemisphere = 'l'
network = 'DAN5'
run = '1'
subject = 's06_137'
in_mlsc = True
write_dir = 'HDRmodeling'
job_number = 5
verbose = True
interactive = False

### Code starts
root_dir = MLSC_ROOT_DIR if in_mlsc else URSA_ROOT_DIR
par_f = os.path.join(root_dir, PROJECT_DIR, EEG_DIR, network, f'{subject}-r{run}.par')
print(par_f)
nii_f = os.path.join(root_dir, PROJECT_DIR, FMRI_DIR, subject, 'rest', f'fsrest_{hemisphere}h_native', 'res', f'res-00{run}.nii.gz')
print(nii_f)

eeg_data = np.fromfile(par_f, sep='\n')
eeg = EEGData(eeg_data, 20)  # 20Hz

fmri_data = sf.load_volume(nii_f)
TR_fmri = 800
fmri = EEGData(fmri_data, TR_fmri)
#eeg_sample_freq = 20  # 20Hz


n_voxels = fmri_data.shape[0]
start = math.ceil(n_voxels/NUMBER_OF_TASKS)*job_number
end = min([math.ceil(n_voxels/NUMBER_OF_TASKS)*(job_number+1), n_voxels])

if end < start:
    raise ValueError(f"Job number {job_number} too large - no data to evaluate")

out_file = os.path.join(root_dir, PROJECT_DIR, write_dir, f'{subject}-{network}-{hemisphere}h-r{run}-j{job_number}-voxels{start}_{end}.csv')
print(out_file)
if not os.path.exists(os.path.dirname(out_file)):
    os.makedirs(os.path.dirname(out_file))

descriptions = []
tstart = time.time()
for voxel in range(start, end):
    tend = time.time()
    if verbose:
        print(f'Voxel: {voxel} ({(voxel-start)/(end-start)*100:.2f}%). Last voxel took {tend-tstart:.2f} seconds')
    tstart = time.time()
    data = []
    fmri_voxel_data = fmri_data[voxel,:,:,:].data.squeeze()
    models = {
        search_type: SEARCH_TYPES[search_type](eeg, 
                                               fMRIData(fmri_voxel_data, TR_fmri),
                                               plot=interactive) 
        for search_type in ['classic_hemodynamic']}
    descriptions.extend(
        search_voxel(
            voxel, 
            models, 
            np.arange(1, 3.2, 0.2),  # 1-3s, 11 pts
            np.arange(0.75, 1.8, 0.1),  # 0.75-1.75s, 11pts
            np.arange(1.75, 2.30, 0.05)  # 1.75-2.25s, 11pts
        )
    )

with open(out_file, 'w') as f:
    pd.DataFrame(descriptions).to_csv(f)