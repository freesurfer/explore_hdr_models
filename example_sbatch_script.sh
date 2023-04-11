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
time python ./run_search.py