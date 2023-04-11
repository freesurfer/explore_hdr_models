import matplotlib.pyplot as plt
import numpy as np

from feeg_fmri_sync.plotting import plot_hdr_for_eeg
from feeg_fmri_sync.utils import (
    downsample_hdr_for_eeg,
    get_est_hemodynamic_response, 
    get_hdr_for_eeg, 
    get_ratio_eeg_freq_to_fmri_freq,
)
from tests.helpers import (
    load_expected_downsampled_hdr_for_eeg,
    load_expected_hdr_for_eeg,
    load_test_eeg_without_nans,
)
from tests.data.hdr import HDR

PLOT = True

class TestGetEstHemodynamicResponse:
    time_steps = np.arange(stop=4, step=0.5)
    delta = 2.25
    tau = 1.25
    alpha = 2

    def test_all_zero(self):
        delta = 4
        est_hemodynamic_response = get_est_hemodynamic_response(
            self.time_steps, delta=delta, tau=self.tau, alpha=self.alpha
        )
        assert est_hemodynamic_response.size == self.time_steps.size, f'est_hemodynamic_response returns an array with a different shape {est_hemodynamic_response.size} than input time steps {self.time_steps.size}'
        assert np.count_nonzero(est_hemodynamic_response) == 0, f'est_hemodynamic_response: ({est_hemodynamic_response}) should be all zeros if delta ({delta}) > max(time_steps) ({np.max(self.time_steps)})'

    def test_non_zero_defaults(self):
        hdr = get_est_hemodynamic_response(
            self.time_steps, delta=self.delta, tau=self.tau, alpha=self.alpha
        )
        expected_hdr = np.array([0, 0, 0, 0, 0, 0.0605, 0.3650, 0.6796])
        assert hdr.size == self.time_steps.size, f'est_hemodynamic_response returns an array with a different shape {hdr.size} than input time steps {self.time_steps.size}'
        assert np.allclose(hdr, expected_hdr, rtol=0.001, atol=0.001), f'est_hemodynamic_response: ({hdr}) should be equal to {expected_hdr}'

    def test_non_zero_odd_alpha(self):
        hdr = get_est_hemodynamic_response(
            self.time_steps, delta=self.delta, tau=self.tau, alpha=3
        )
        expected_hdr = np.array([0, 0, 0, 0, 0, 0.0049, 0.0882, 0.2737])
        assert hdr.size == self.time_steps.size, f'est_hemodynamic_response returns an array with a different shape {hdr.size} than input time steps {self.time_steps.size}'
        assert np.allclose(hdr, expected_hdr, rtol=0.001, atol=0.001), f'est_hemodynamic_response: ({hdr}) should be equal to {expected_hdr}'


def test_get_hdr_for_eeg():
    eeg_sample_freq = 20
    eeg_data = load_test_eeg_without_nans(sample_frequency=eeg_sample_freq)
    hdr_for_eeg = get_hdr_for_eeg(eeg_data.data, HDR)
    exp_hdr = load_expected_hdr_for_eeg()
    assert hdr_for_eeg.size == eeg_data.data.size
    assert np.isclose(hdr_for_eeg, exp_hdr).all()
    if PLOT:
        plot_hdr_for_eeg(eeg_data, hdr_for_eeg)
        plot_hdr_for_eeg(eeg_data, exp_hdr)
    

def test_downsampled_hdr_for_eeg():
    eeg_sample_freq = 20
    tr = 800
    r_fmri = get_ratio_eeg_freq_to_fmri_freq(eeg_sample_freq, tr)
    hdr_for_eeg = load_expected_hdr_for_eeg()
    expected_ds_hdr = load_expected_downsampled_hdr_for_eeg()
    ds_hdr = downsample_hdr_for_eeg(r_fmri, hdr_for_eeg)
    assert np.isclose(ds_hdr, expected_ds_hdr).all()
    if PLOT:
        eeg_data = load_test_eeg_without_nans(sample_frequency=eeg_sample_freq)
        plot_hdr_for_eeg(eeg_data, hdr_for_eeg, tr, ds_hdr)
        plot_hdr_for_eeg(eeg_data, hdr_for_eeg, tr, expected_ds_hdr)


def test_helper_simulate_fmri():
    eeg_sample_freq = 20
    eeg_data = load_test_eeg_without_nans(sample_frequency=eeg_sample_freq)
