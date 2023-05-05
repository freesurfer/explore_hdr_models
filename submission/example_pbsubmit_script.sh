#!/bin/bash

# >>> conda initialize >>>
__conda_setup="$('/usr/pubsw/packages/python/anaconda3-2019.03/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
	eval "$__conda_setup"
else
	if [ -f "/usr/pubsw/packages/python/anaconda3-2019.03/etc/profile.d/conda.sh" ]; then
		. "/usr/pubsw/packages/python/anaconda3-2019.03/etc/profile.d/conda.sh"	else
		export PATH="/usr/pubsw/packages/python/anaconda3-2019.03/bin:$PATH"
	fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate feeg_fmri
time python /autofs/space/ursa_004/users/mark11/COVID/code/feeg_fmri_sync/run_search_on_roi_gamma_model.py \
	--par-file=/autofs/space/ursa_004/users/HDRmodeling/EEGspikeTrains/DAN5/s13_151-r1.par \
	--out-dir=/autofs/space/ursa_004/users/HDRmodeling/HDRmodelingOutputs/DAN5/s13_151/1/ \
	--mat-file=/autofs/space/ursa_004/users/HDRmodeling/ROItcs_fMRI/mat4HMMregWindu.mat \
	--sub-and-run-i=15 \
	--search-type=classic_hemodynamic \
	--eeg-sample-frequency=20 \
	--tr=800 \
	--num-trs-skipped-at-beginning=1 \
	--hdr-window=30 \
	--delta-start=0.5 \
	--delta-end=1.5 \
	--delta-step=0.05 \
	--tau-start=0.75 \
	--tau-end=1.75 \
	--tau-step=0.05 \
	--alpha-start=1 \
	--alpha-end=5.5 \
	--alpha-step=0.05 \
	--verbose \
	--out-name=s13_151_r1_DAN5
