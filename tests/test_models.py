import numpy as np

from feeg_fmri_sync.models import CanonicalHemodynamicModel
from feeg_fmri_sync.plotting import plot_hdr
from tests.helpers import (
    load_expected_hdr,
    load_test_eeg_with_nans, 
    load_test_eeg_without_nans,
    load_simulated_raw_fmri,
)

PLOT = False


class TestHemodynamicResponseWithoutNans:

    model = CanonicalHemodynamicModel(eeg=load_test_eeg_without_nans(sample_frequency=20),
                                      fmri=load_simulated_raw_fmri(tr=800), name='test_name',
                                      hemodynamic_response_window=30, display_plot=False)
    delta = 2.25
    tau = 1.25
    alpha = 2

    def test_get_time_steps(self):
        time_steps = self.model.get_time_steps()
        assert time_steps.size == 601
    
    def test_get_hemodynamic_response(self):
        hdr = self.model.get_est_hemodynamic_response(self.delta, self.tau, self.alpha)
        expected_hdr = load_expected_hdr()
        assert np.isclose(hdr, expected_hdr).all()
    
    def test_plot(self):
        if PLOT:
            time_steps = self.model.get_time_steps()
            hdr = self.model.get_est_hemodynamic_response(self.delta, self.tau, self.alpha)
            plot_hdr(time_steps, hdr)


class TestHemodynamicResponseWithNans(TestHemodynamicResponseWithoutNans):

    model = CanonicalHemodynamicModel(eeg=load_test_eeg_with_nans(sample_frequency=20),
                                      fmri=load_simulated_raw_fmri(tr=800), name='test_hdr',
                                      hemodynamic_response_window=30, display_plot=False)
