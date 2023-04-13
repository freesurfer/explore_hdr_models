import numpy as np
import numpy.typing as npt
import warnings

from typing import Callable, Optional, Tuple

from feeg_fmri_sync.constants import (
    EEGData,
    fMRIData
)
from feeg_fmri_sync.models import HemodynamicModel
from feeg_fmri_sync.utils import (
    get_hdr_for_eeg,
    sum_hdr_for_eeg,
)


class VectorizedHemodynamicModel(HemodynamicModel):    
    def __init__(self, eeg: EEGData, fmri: fMRIData, name: str, n_tr_skip_beg: int = 1,
                 hemodynamic_response_window: float = 30, plot: bool = True):
        super().__init__(eeg, fmri, name, n_tr_skip_beg, hemodynamic_response_window, plot)

        # Un-squeeze the fmri data
        self.fmri: fMRIData = fmri

        # Configuration parameters
        voxel_name, _ = self.fmri.get_voxel_i(0)
        self.plot_voxels = [voxel_name]
            
        # internal tracking for better performance
        self.est_fmri_n_trs: Optional[int] = None

    def set_plot_voxels(self, voxel_names_to_plot):
        self.plot_voxels = voxel_names_to_plot

    def get_transformation_functions(self, est_fmri: fMRIData) -> Tuple[Callable, Callable]:
        if not self.transform_est_fmri or not self.transform_actual_fmri or self.est_fmri_n_trs != est_fmri.get_n_trs():
            if self.est_fmri_n_trs and self.est_fmri_n_trs != est_fmri.get_n_trs():
                print(f'WARNING: estimated fMRI size changed! {self.est_fmri_n_trs} -> {est_fmri.get_n_trs()}')
            tr_axis = self.fmri.get_tr_axis()
            actual_fmri_compression_mask = np.arange(self.fmri.data.shape[tr_axis]) >= self.n_tr_skip_beg
            self.est_fmri_n_trs = est_fmri.get_n_trs()
            self.transform_est_fmri = lambda x: x
            self.transform_actual_fmri = lambda x: x.compress(actual_fmri_compression_mask, axis=tr_axis)
            if est_fmri.get_n_trs() > self.fmri.get_n_trs() - self.n_tr_skip_beg:
                if self.plot:
                    print(f'Estimated fMRI is larger than actual fMRI '
                          f'(-# skipped TRs at beginning of EEG): '
                          f'{est_fmri.get_n_trs()} : {self.fmri.get_n_trs()} - '
                          f'{self.n_tr_skip_beg}')
                est_fmri_compression_mask = np.arange(self.fmri.data.shape[tr_axis]) < self.fmri.get_n_trs()
                self.transform_est_fmri = lambda x: x.compress(est_fmri_compression_mask, axis=tr_axis)
            if est_fmri.get_n_trs() < self.fmri.get_n_trs() - self.n_tr_skip_beg:
                if self.plot:
                    print(f'Estimated fMRI is smaller than actual fMRI '
                    f'(-# skipped TRs at beginning of EEG): '
                    f'{est_fmri.get_n_trs()} : {self.fmri.get_n_trs()} '
                    f'- {self.n_tr_skip_beg}')
                actual_fmri_compression_mask = np.logical_and(
                    actual_fmri_compression_mask, 
                    np.arange(self.fmri.data.shape[tr_axis]) < (est_fmri.get_n_trs() + self.n_tr_skip_beg)
                )
                self.transform_actual_fmri = lambda x: x.compress(actual_fmri_compression_mask, axis=tr_axis)
        return self.transform_est_fmri, self.transform_actual_fmri

    def fit_glm(self, est_fmri: fMRIData, actual_fmri: fMRIData) -> Tuple[
        npt.NDArray, npt.NDArray, npt.NDArray, int
    ]:
        """
        Fit this time course to raw fMRI to waveform with a GLM:
            beta = inv(X'*X)*X'*fmri
            yhat = X*beta
            residual = fmri-yat
            residual variance = std(residual)
        Same effect as doing the correlation but residual variance is a cost we want to minimize
        rather than a correlation to maximize
        """
        if not est_fmri.is_single_voxel():
            warnings.warn(f'Estimated fMRI is multiple voxels. Model has not been tested')
        X_nan = np.isnan(est_fmri.data)
        X_drop_nans = np.extract(~X_nan, est_fmri.data)
        Y_nan = np.tile(X_nan, actual_fmri.get_n_voxels()).reshape(
            (actual_fmri.get_n_voxels(), est_fmri.get_n_trs()))
        # np.extract flattens the array
        y_drop_nans_t = np.extract(~Y_nan, actual_fmri.data).reshape(
            (actual_fmri.get_n_voxels(), X_drop_nans.shape[0]))
        # ones not necessary here
        X_t = np.array([X_drop_nans, np.ones(X_drop_nans.shape[0])])
        beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_t, X_t.T)), X_t), y_drop_nans_t.T)
        y_hat = np.matmul(X_t.T, beta)
        residual = np.subtract(y_drop_nans_t.T, y_hat).T
        degrees_of_freedom = X_t.shape[1] - X_t.shape[0]
        residual_variance = np.sum(residual**2, axis=actual_fmri.get_tr_axis()) / degrees_of_freedom
        return beta, residual, residual_variance, degrees_of_freedom

    def score_from_hemodynamic_response(self, est_hemodynamic_response: npt.NDArray) -> npt.NDArray:
        hemodynamic_response_to_eeg, est_fmri = self.get_est_fmri_hemodynamic_response(
            est_hemodynamic_response
        )
        if self.plot:
            #print(f'Num nans in hemodynamic response to eeg: '
            #      f'{np.count_nonzero(np.isnan(hemodynamic_response_to_eeg))}')
            #print(f'length of eeg: {self.eeg.data.size}')
            #print(f'r_fmri {self.r_fmri}, '
            #      f'length of hemodynamic_response: {hemodynamic_response_to_eeg.size}, '
            #      f'hemodynamic_response/r_fmri: {hemodynamic_response_to_eeg.size / self.r_fmri} '
            #      f'shape of fmri data: {self.fmri.data.shape}')
            self.plot_hdr_for_eeg(hemodynamic_response_to_eeg, est_fmri)
        
        est_fmri_transform, actual_fmri_transform = self.get_transformation_functions(
            fMRIData(est_fmri, self.fmri.TR)
        )
        est_fmri = fMRIData(est_fmri_transform(est_fmri), self.fmri.TR)
        actual_fmri = fMRIData(
            actual_fmri_transform(self.fmri.data),
            self.fmri.TR,
            voxel_names=self.fmri.voxel_names
        )
        beta, residual, residual_variance, degrees_of_freedom = self.fit_glm(
            est_fmri, 
            actual_fmri
        )
        residual = fMRIData(
            residual,
            TR=self.fmri.TR,
            voxel_names=self.fmri.voxel_names
        )
        if self.plot:
            for voxel_name in self.plot_voxels:
                i, voxel_data = actual_fmri.get_voxel_by_name(voxel_name)
                _, residual_data = residual.get_voxel_by_name(voxel_name)
                self.compare_est_fmri_with_actual(est_fmri.data, voxel_data, residual_data, actual_fmri_name=voxel_name)
                print(f'Residual Variance is {residual_variance[i].item():.6f}')
        return residual_variance



class VectorizedSumEEGHemodynamicModel(VectorizedHemodynamicModel):

    def get_est_fmri_hemodynamic_response(self, est_hemodynamic_response):
        hemodynamic_response_to_eeg = get_hdr_for_eeg(self.eeg.data, est_hemodynamic_response)
        est_fmri = sum_hdr_for_eeg(self.r_fmri, hemodynamic_response_to_eeg)
        return hemodynamic_response_to_eeg, est_fmri