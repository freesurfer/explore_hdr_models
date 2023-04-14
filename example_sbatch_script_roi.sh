#!/bin/bash
#SBATCH --account=mobius
#SBATCH --partition=basic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:15:00
#SBATCH -o /cluster/batch/sg871/slurm-%A_%a.out
#SBATCH -e /cluster/batch/sg871/slurm-%A_%a.err

# run with `sbatch example_sbatch_script.sh`

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/pubsw/packages/python/anaconda3-2019.03/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/pubsw/packages/python/anaconda3-2019.03/etc/profile.d/conda.sh" ]; then
        . "/usr/pubsw/packages/python/anaconda3-2019.03/etc/profile.d/conda.sh"
    else
        export PATH="/usr/pubsw/packages/python/anaconda3-2019.03/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate feeg_fmri
time python ./run_search_on_roi_gamma_model.py \
    --par-file=/autofs/space/ursa_004/users/HDRmodeling/EEGspikeTrains/DAN5/s06_137-r1.par \
    --mat-file=/autofs/space/ursa_004/users/HDRmodeling/ROItcs_fMRI/mat4HMMregWindu.mat \
    --sub-and-run-i=1 \
    -v \
    --out-dir=/autofs/space/ursa_004/users/HDRmodeling/HDRmodeling/DAN/s06_137-r1/ \
    --out-name=DAN_s06_137_r1 \
    --tau-start=0.25 \
    --tau-end=3.0 \
    --delta-end=5.0 \
    --alpha-start=1.5 \
    --alpha-end=3.0
