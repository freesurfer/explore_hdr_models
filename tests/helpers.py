import os

import numpy as np
import scipy.io

from feeg_fmri_sync.constants import EEGData, fMRIData


def load_test_eeg_with_nans(sample_frequency=20):
    eeg_data = np.fromfile(f'{os.path.dirname(__file__)}/data/eeg.par', sep='\n')
    return EEGData(eeg_data, sample_frequency)


def load_test_eeg_without_nans(sample_frequency=20):
    eeg_data = np.fromfile(f'{os.path.dirname(__file__)}/data/eeg.par', sep='\n')
    eeg_data[np.isnan(eeg_data)] = 0
    return EEGData(eeg_data, sample_frequency)


def load_expected_hdr():
    exp_hdr = scipy.io.loadmat(f'{os.path.dirname(__file__)}/data/expected_hdr.mat')
    return exp_hdr['hrf'].squeeze()


def load_expected_hdr_for_eeg():
    exp_hdr = scipy.io.loadmat(f'{os.path.dirname(__file__)}/data/expected_hdr_for_eeg.mat')
    return exp_hdr['hrfeeg'].squeeze()


def load_expected_downsampled_hdr_for_eeg():
    exp_ds_hdr = scipy.io.loadmat(f'{os.path.dirname(__file__)}/data/expected_downsampled_hdr.mat')
    return exp_ds_hdr['hrfeeg_fmri'].squeeze()


def load_simulated_raw_fmri_data():
    sim_fmri = scipy.io.loadmat(f'{os.path.dirname(__file__)}/data/simulated_fmri.mat')
    return sim_fmri['fmri'].squeeze()


def load_simulated_raw_fmri(tr=800):
    return fMRIData(load_simulated_raw_fmri_data(), tr)



