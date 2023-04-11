"""
Run grid search to find best fitting delta, tau, and alpha using the canonical hemodynamic model:
    h(t>delta)  = ((t-delta)/tau)^alpha * exp(-(t-delta)/tau)
    h(t<=delta) = 0

python run_search.py --par_file /autofs/space/ursa_004/users/HDRmodeling/EEGspikeTrains/DAN5/s06_137-r1.par --nii-file /autofs/space/ursa_004/users/HDRmodeling/HDRshape/s06_137/rest/fsrest_lh_native/res/res-001.nii.gz --out-file

"""
import argparse
import math
import numpy as np
import os
import pandas as pd
import surfa as sf
import time

from feeg_fmri_sync import SEARCH_TYPES, VALID_KWARGS
from feeg_fmri_sync.constants import NUMBER_OF_TASKS
from feeg_fmri_sync.models import EEGData, fMRIData

print(VALID_KWARGS)

parser = argparse.ArgumentParser()
parser.add_argument('--par-file', required=True, help=f'par file path')

parser.add_argument('--nii-file', required=True, help=f'nii file path')
parser.add_argument('--job-number', type=int, required=True, help='Job number assigned by submit')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--out-file', required=True, help=f'out file path')
parser.add_argument('--search-type', nargs='+', default=['classic_hemodynamic'], choices=SEARCH_TYPES.keys())
parser.add_argument('search-kwargs', nargs='*', default=[])
parser.add_argument('--eeg-sample-frequency', type=int, default=20)
parser.add_argument('--tr', type=int, default=800)

args = parser.parse_args()



### Code starts
eeg_data = np.fromfile(args.par_file, sep='\n')
eeg = EEGData(eeg_data, args.eeg_sample_frequency)

fmri_data = sf.load_volume(args.nii_file)

n_voxels = fmri_data.shape[0]
start = math.ceil(n_voxels/NUMBER_OF_TASKS)*args.job_number
end = min([math.ceil(n_voxels/NUMBER_OF_TASKS)*(args.job_number+1), n_voxels])

if not os.path.exists(os.path.dirname(args.out_file)):
    os.makedirs(os.path.dirname(args.out_file))

descriptions = []
tstart = time.time()
for voxel in range(start, end):
    tend = time.time()
    if args.verbose:
        print(f'Voxel: {voxel} ({(voxel-start)/(end-start)*100:.2f}%). '
              f'Last voxel took {tend-tstart:.2f} seconds')
    tstart = time.time()
    data = []
    fmri_voxel_data = fmri_data[voxel,:,:,:].data.squeeze()
    models = {
        search_type: SEARCH_TYPES[search_type](eeg, 
                                               fMRIData(fmri_voxel_data, args.tr),
                                               plot=args.verbose,
                                               ) 
        for search_type in args.search_type}
    for delta in np.arange(1, 3.2, 0.2): #1-3s, 11 pts
        for tau in np.arange(0.75, 1.8, 0.1): #0.75-1.75s, 11pts
            for alpha in np.arange(1.75, 2.30, 0.05): #1.75-2.25s, 11pts
                scores = []
                for model in models.values():
                    scores.append(model.score(delta, tau, alpha))
                data.append([delta, tau, alpha]+scores)
    df = pd.DataFrame(data, columns=['delta', 'tau', 'alpha'] + [k for k in models.keys()])
    for model_name in models.keys():
        description = df[model_name].describe()
        description.name = f'{model_name}_{voxel}'
        for column_name in ['delta', 'tau', 'alpha']:
            min_val = df[df[model_name] == df[model_name].min()][column_name].iloc[0]
            description[column_name] = min_val
        descriptions.append(description)

with open(args.out_file, 'w') as f:
    pd.DataFrame(descriptions).to_csv(f)