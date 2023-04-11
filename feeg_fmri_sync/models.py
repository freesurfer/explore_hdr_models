import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from typing import Callable, Optional, Tuple

from feeg_fmri_sync.constants import (
    EEGData,
    fMRIData
)
from feeg_fmri_sync.utils import (
    downsample_hdr_for_eeg,
    get_est_hemodynamic_response,
    get_hdr_for_eeg,
    get_ratio_eeg_freq_to_fmri_freq,
    sum_hdr_for_eeg,
)


class GridSearch:
    def __init__(self, param_names):
        self.params = param_names


class HemodynamicModel:    
    def __init__(self, eeg: EEGData, fmri: fMRIData, name: str, n_tr_skip_beg: int = 1, 
                 hemodynamic_response_window: int = 30, plot: bool = True):
        # Data
        self.eeg: EEGData = eeg
        self.fmri: fMRIData = fMRIData(data=fmri.data.squeeze(), TR=fmri.TR)
        #print(f'Num eeg nans: {np.count_nonzero(np.isnan(self.eeg.data))}')

        # Ratio of eeg freq to fMRI frequency
        self.r_fmri: float = get_ratio_eeg_freq_to_fmri_freq(self.eeg.sample_frequency, self.fmri.TR)
        # Number of TRs skipped at beginning of EEG
        self.n_tr_skip_beg: int = n_tr_skip_beg
        
        # Tunable parameters
        self.hemodynamic_response_window:float  =  hemodynamic_response_window  # seconds
        
        # Configuration parameters
        self.name = name
        self.plot = plot        

        # internal tracking for better performance
        self.transform_est_fmri: Optional[Callable] = None
        self.transform_actual_fmri: Optional[Callable] = None
        self.est_fmri_size: Optional[int] = None

    def __str__(self):
        return self.__name__

    def plot_hdr_for_eeg(self, hdr: npt.NDArray, fmri_hdr: npt.NDArray) -> None:
        plt.cla()
        time_steps_for_eeg = np.arange(len(self.eeg.data)) / self.eeg.sample_frequency
        time_steps_for_fmri = time_steps_for_eeg[::self.r_fmri]
        plt.plot(time_steps_for_fmri, fmri_hdr, '.-', label='HDR-fMRI')
        plt.plot(time_steps_for_eeg, hdr, label='HDR-EEG')
        plt.plot(time_steps_for_eeg, self.eeg.data, label='EEG spikes')
        plt.title(f'Estimated hemodynamic response (EEG and fMRI time scales) from EEG spikes using model={self.name}')
        plt.legend()
        plt.show()

    def compare_est_fmri_with_actual(
        self,
        est_fmri: npt.NDArray, 
        actual_fmri: npt.NDArray, 
        residual: npt.NDArray,
        actual_fmri_name: Optional[str] = None):
        plt.cla()
        time_steps_for_eeg = np.arange(len(self.eeg.data)) / self.eeg.sample_frequency
        time_steps_for_fmri = time_steps_for_eeg[::self.r_fmri]
        if residual is not None:
            X_nan = np.isnan(est_fmri)
            plt.plot(time_steps_for_fmri[~X_nan], residual, '.', label='Residual')
        plt.plot(time_steps_for_fmri, est_fmri, label='Estimated fMRI')
        plt.plot(time_steps_for_fmri, actual_fmri, label='Actual fMRI')
        plt.plot(time_steps_for_eeg, self.eeg.data, label='EEG spikes')
        if not actual_fmri_name:
            title = f'Estimated fMRI HDR from EEG spikes compared with actual fMRI using model={self.name}'
        else:
            title = f'Estimated fMRI HDR from EEG spikes compared with actual fMRI ({actual_fmri_name}) using model={self.name}'
        plt.title(title)
        plt.legend()
        plt.show()

    def get_time_steps(self):
        return np.arange(self.hemodynamic_response_window*self.eeg.sample_frequency + 1) / self.eeg.sample_frequency

    def get_est_hemodynamic_response(self, delta: float, tau: float, alpha: float) -> npt.NDArray:
        """
        Time sampled at EEG rate at a long enough window (hemodynamic response window, default 30sec) 
        to capture the hemodynamic response
        """
        return get_est_hemodynamic_response(self.get_time_steps(), delta, tau, alpha)

    def get_est_fmri_hemodynamic_response(self, est_hemodynamic_response: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        hemodynamic_response_to_eeg = get_hdr_for_eeg(self.eeg.data, est_hemodynamic_response)
        est_fmri = downsample_hdr_for_eeg(self.r_fmri, hemodynamic_response_to_eeg)
        return hemodynamic_response_to_eeg, est_fmri

    def get_transformation_functions(self, est_fmri):
        if not self.transform_est_fmri or not self.transform_actual_fmri or self.est_fmri_size != est_fmri.size:
            if self.est_fmri_size and self.est_fmri_size != est_fmri.size:
                print(f'WARNING: estimated fMRI size changed! {self.est_fmri_size} -> {est_fmri.size}')
            self.est_fmri_size = est_fmri.size
            self.transform_est_fmri = lambda x: x
            self.transform_actual_fmri = lambda x: x[self.n_tr_skip_beg:]
            if est_fmri.size > self.fmri.data.size - self.n_tr_skip_beg:
                if self.plot:
                    print(f'Estimated fMRI is larger than actual fMRI (-# skipped TRs at beginning of EEG): {est_fmri.size} : {self.fmri.data.size} - {self.n_tr_skip_beg}')
                self.transform_est_fmri = lambda x: x[:self.fmri.data.size]
            if est_fmri.size < self.fmri.data.size - self.n_tr_skip_beg:
                if self.plot:
                    print(f'Estimated fMRI is smaller than actual fMRI (-# skipped TRs at beginning of EEG): {est_fmri.size} : {self.fmri.data.size} - {self.n_tr_skip_beg}')
                self.transform_actual_fmri = lambda x: x[self.n_tr_skip_beg:est_fmri.size + self.n_tr_skip_beg]
        return self.transform_est_fmri, self.transform_actual_fmri

    def fit_glm(self, est_fmri, actual_fmri):
        """
        Fit this time course to raw fMRI to waveform with a GLM:
            beta = inv(X'*X)*X'*fmri
            yhat = X*beta
            residual = fmri-yat
            residual variance = std(residual)
        Same effect as doing the correlation but residual variance is a cost we want to minimize
        rather than a correlation to maximize
        """
        if self.plot:
            print(f'Is est_fmri close to actual_fmri? {np.isclose(est_fmri, actual_fmri)}')
        X_nan = np.isnan(est_fmri)
        X_drop_nans = np.extract(~X_nan, est_fmri)
        y_drop_nans_t = np.extract(~X_nan, actual_fmri)
        # ones not necessary here
        X_t = np.array([X_drop_nans, np.ones(X_drop_nans.size)])
        beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_t, X_t.T)), X_t), y_drop_nans_t.T)
        y_hat = np.matmul(X_t.T, beta)
        residual = np.subtract(y_drop_nans_t, y_hat)
        degrees_of_freedom = X_t.shape[1] - X_t.shape[0]
        residual_variance = np.sum(residual**2) / degrees_of_freedom
        return beta, residual, residual_variance, degrees_of_freedom

    def score_from_hemodynamic_response(self, est_hemodynamic_response):
        hemodynamic_response_to_eeg, est_fmri = self.get_est_fmri_hemodynamic_response(est_hemodynamic_response)
        if self.plot:
            #print(f'Num nans in hemodynamic response to eeg: {np.count_nonzero(np.isnan(hemodynamic_response_to_eeg))}')
            #print(f'length of eeg: {self.eeg.data.size}')
            #print(f'r_fmri {self.r_fmri}, length of hemodynamic_response: {hemodynamic_response_to_eeg.size}, '
            #      f'hemodynamic_response/r_fmri: {hemodynamic_response_to_eeg.size / self.r_fmri} length of fmri data: {self.fmri.data.size}')
            self.plot_hdr_for_eeg(hemodynamic_response_to_eeg, est_fmri)
        
        est_fmri_transform, actual_fmri_transform = self.get_transformation_functions(est_fmri)
        est_fmri = est_fmri_transform(est_fmri)
        actual_fmri = actual_fmri_transform(self.fmri.data)
        beta, residual, residual_variance, degrees_of_freedom = self.fit_glm(est_fmri, actual_fmri)
        if self.plot:
            self.compare_est_fmri_with_actual(est_fmri, actual_fmri, residual)
        return residual_variance

    def score(self, delta: float, tau: float, alpha: float):
        """
        delta: delay
        tau: 
        alpha: exponent
        """
        if self.plot:
            print(f"Scoring {self.name} delta={delta}, tau={tau}, alpha={alpha}")
        est_hemodynamic_response = self.get_est_hemodynamic_response(delta, tau, alpha)
        return self.score_from_hemodynamic_response(est_hemodynamic_response)


class SumEEGHemodynamicModel(HemodynamicModel):

    def get_est_fmri_hemodynamic_response(self, est_hemodynamic_response):
        hemodynamic_response_to_eeg = get_hdr_for_eeg(self.eeg.data, est_hemodynamic_response)
        est_fmri = sum_hdr_for_eeg(self.r_fmri, hemodynamic_response_to_eeg)
        return hemodynamic_response_to_eeg, est_fmri