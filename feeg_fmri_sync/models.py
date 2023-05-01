import os

import numpy as np
import numpy.typing as npt

from typing import Callable, Optional, Tuple

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from feeg_fmri_sync.constants import (
    EEGData,
    fMRIData
)
from feeg_fmri_sync.plotting import generate_latex_label
from feeg_fmri_sync.utils import (
    get_hdr_for_eeg,
    sum_hdr_for_eeg, get_ratio_eeg_freq_to_fmri_freq, get_est_hemodynamic_response, downsample_hdr_for_eeg, fit_glm,
    get_contrast_matrix,
)


class CanonicalHemodynamicModel:
    save_plot_dir: Optional[str] = None
    delta: Optional[float] = None
    tau: Optional[float] = None
    alpha: Optional[float] = None

    def __init__(self, eeg: EEGData, fmri: fMRIData, name: str, n_tr_skip_beg: int = 1,
                 hemodynamic_response_window: float = 30, display_plot: bool = True,
                 save_plot_dir: Optional[str] = None):
        # Data
        self.eeg: EEGData = eeg
        self.fmri: fMRIData = fmri

        # Ratio of eeg freq to fMRI frequency
        self.r_fmri: float = get_ratio_eeg_freq_to_fmri_freq(self.eeg.sample_frequency, self.fmri.TR)
        # Number of TRs skipped at beginning of EEG
        self.n_tr_skip_beg: int = n_tr_skip_beg

        # Tunable parameters
        self.hemodynamic_response_window: float = hemodynamic_response_window  # seconds

        # Plotting parameters
        self.name: str = name
        self.display_plot: bool = display_plot
        self.set_save_plot_dir(save_plot_dir)
        voxel_name, _ = self.fmri.get_voxel_i(0)
        self.plot_voxels = [voxel_name]
        self.figures_to_plot = []

        # internal tracking for better performance
        self.transform_est_fmri: Optional[Callable] = None
        self.transform_actual_fmri: Optional[Callable] = None
        self.est_fmri_size: Optional[int] = None
        self.est_fmri_n_trs: Optional[int] = None

    def set_save_plot_dir(self, save_plot_dir: Optional[str]):
        if save_plot_dir:
            if not os.path.exists(save_plot_dir):
                os.makedirs(save_plot_dir)
        self.save_plot_dir = save_plot_dir

    def set_params(self, delta, tau, alpha):
        self.delta = delta
        self.tau = tau
        self.alpha = alpha

    def unset_params(self):
        self.delta = None
        self.tau = None
        self.alpha = None

    def __str__(self):
        return self.__name__

    def set_plot_voxels(self, voxel_names_to_plot):
        self.plot_voxels = voxel_names_to_plot

    def plot_hdr_for_eeg(self, hdr: npt.NDArray, fmri_hdr: npt.NDArray) -> None:
        plt.figure()
        time_steps_for_eeg = np.arange(len(self.eeg.data)) / self.eeg.sample_frequency
        time_steps_for_fmri = time_steps_for_eeg[::self.r_fmri]
        plt.plot(time_steps_for_eeg, self.eeg.data, label='EEG spikes', alpha=0.25)
        plt.plot(time_steps_for_fmri, fmri_hdr, '.-', label='HDR-fMRI')
        plt.plot(time_steps_for_eeg, hdr, label='HDR-EEG')
        plt.title(f'Estimated hemodynamic response (EEG and fMRI time scales) from EEG spikes using model={self.name}')
        plt.legend()
        self.figures_to_plot.append(plt.gcf())

    def compare_est_fmri_with_actual(
            self,
            est_fmri: npt.NDArray,
            actual_fmri: npt.NDArray,
            residual: npt.NDArray,
            residual_variance: float,
            actual_fmri_name: Optional[str] = None):
        plt.figure()
        time_steps_for_eeg = np.arange(len(self.eeg.data)) / self.eeg.sample_frequency
        time_steps_for_fmri = time_steps_for_eeg[::self.r_fmri]
        plt.plot(time_steps_for_eeg, self.eeg.data, label='EEG spikes', alpha=0.25)
        if residual is not None:
            x_nan = np.isnan(est_fmri)
            plt.plot(time_steps_for_fmri[~x_nan], residual, '.', label='Residual')
            plt.xlabel(f'Residual Variance is {residual_variance:.6f}')
        plt.plot(time_steps_for_fmri, actual_fmri, label='Actual fMRI')
        plt.plot(time_steps_for_fmri, est_fmri, label='Estimated fMRI')
        if not actual_fmri_name:
            title = f'Estimated fMRI HDR from EEG spikes compared with actual fMRI using model={self.name}'
        else:
            title = f'Estimated fMRI HDR from EEG spikes compared with actual fMRI ({actual_fmri_name}) \n' \
                    f'using model={self.name}, {generate_latex_label("delta")}={self.delta:.4f}, ' \
                    f'{generate_latex_label("tau")}={self.tau:.4f}, {generate_latex_label("alpha")}={self.alpha:.4f},'
        plt.title(title)
        plt.legend()
        self.figures_to_plot.append(plt.gcf())

    def get_time_steps(self):
        return np.arange(self.hemodynamic_response_window*self.eeg.sample_frequency + 1) / self.eeg.sample_frequency

    def get_est_hemodynamic_response(self, delta: float, tau: float, alpha: float) -> npt.NDArray:
        """
        Time sampled at EEG rate at a long enough window (hemodynamic response window, default 30sec)
        to capture the hemodynamic response
        """
        return get_est_hemodynamic_response(self.get_time_steps(), delta, tau, alpha)

    def get_est_fmri_hemodynamic_response(self,
                                          est_hemodynamic_response: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        hemodynamic_response_to_eeg = get_hdr_for_eeg(self.eeg.data, est_hemodynamic_response)
        est_fmri = downsample_hdr_for_eeg(self.r_fmri, hemodynamic_response_to_eeg)
        return hemodynamic_response_to_eeg, est_fmri

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
                if self.display_plot:
                    print(f'Estimated fMRI is larger than actual fMRI '
                          f'(-# skipped TRs at beginning of EEG): '
                          f'{est_fmri.get_n_trs()} : {self.fmri.get_n_trs()} - '
                          f'{self.n_tr_skip_beg}')
                est_fmri_compression_mask = np.arange(self.fmri.data.shape[tr_axis]) < self.fmri.get_n_trs()
                self.transform_est_fmri = lambda x: x.compress(est_fmri_compression_mask, axis=tr_axis)
            if est_fmri.get_n_trs() < self.fmri.get_n_trs() - self.n_tr_skip_beg:
                if self.display_plot:
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

    def score_from_hemodynamic_response(self, est_hemodynamic_response: npt.NDArray, column: Optional[str]) -> Tuple[
            npt.NDArray, fMRIData, npt.NDArray, float]:
        hemodynamic_response_to_eeg, est_fmri = self.get_est_fmri_hemodynamic_response(est_hemodynamic_response)
        if self.display_plot or self.save_plot_dir:
            self.plot_hdr_for_eeg(hemodynamic_response_to_eeg, est_fmri)

        est_fmri_transform, actual_fmri_transform = self.get_transformation_functions(fMRIData(est_fmri, self.fmri.TR))
        est_fmri = fMRIData(est_fmri_transform(est_fmri), self.fmri.TR)
        actual_fmri = fMRIData(
            actual_fmri_transform(self.fmri.data),
            self.fmri.TR,
            voxel_names=self.fmri.voxel_names
        )
        if column:
            _, voxel_data = actual_fmri.get_voxel_by_name(column)
            actual_fmri = fMRIData(
                voxel_data,
                self.fmri.TR,
                voxel_names=[column]
            )
        beta, residual, residual_variance, degrees_of_freedom = fit_glm(est_fmri, actual_fmri)
        if column:
            residual = fMRIData(residual.squeeze(), TR=self.fmri.TR, voxel_names=[column])
        else:
            residual = fMRIData(residual, TR=self.fmri.TR, voxel_names=self.fmri.voxel_names)
        if self.display_plot or self.save_plot_dir:
            for voxel_name in self.plot_voxels:
                if actual_fmri.is_single_voxel():
                    voxel_data = actual_fmri.data
                    residual_data = residual.data
                    i = 0
                    residual_variance_at_i = residual_variance.item()
                else:
                    i, voxel_data = actual_fmri.get_voxel_by_name(voxel_name)
                    _, residual_data = residual.get_voxel_by_name(voxel_name)
                    residual_variance_at_i = residual_variance[i].item()
                beta_val = np.matmul(get_contrast_matrix(), beta)[i].item()
                self.compare_est_fmri_with_actual(
                    beta_val * est_fmri.data,
                    voxel_data,
                    residual_data,
                    residual_variance_at_i,
                    actual_fmri_name=voxel_name
                )
                if self.display_plot:
                    print(f'Residual Variance is {residual_variance_at_i:.6f}')
        return beta, residual, residual_variance, degrees_of_freedom

    def score(self, delta: float, tau: float, alpha: float):
        """
        delta: delay
        tau:
        alpha: exponent
        """
        _, _, residual_variance, _ = self.score_detailed(delta, tau, alpha)
        return residual_variance

    def score_detailed(self, delta: float, tau: float, alpha: float, column: Optional[str] = None) -> Tuple[
            npt.NDArray, fMRIData, npt.NDArray, float]:
        self.set_params(delta, tau, alpha)
        if self.display_plot:
            name = self.name
            if column:
                name = f'{self.name}, column {column}'
            print(f"Scoring {name}: delta={delta:.4f}, tau={tau:.4f}, alpha={alpha:.4f}")
        est_hemodynamic_response = self.get_est_hemodynamic_response(delta, tau, alpha)
        beta, residual, residual_variance, dof = self.score_from_hemodynamic_response(est_hemodynamic_response, column)
        if self.display_plot:
            plt.show()
        elif self.save_plot_dir:
            with PdfPages(os.path.join(self.save_plot_dir, f'{self.name}_cmp_est_fmri_with_actual.pdf')) as pdf:
                for fig in self.figures_to_plot:
                    pdf.savefig(fig, bbox_inches='tight')
        plt.close('all')
        self.unset_params()
        return beta, residual, residual_variance, dof


class VectorizedSumEEGHemodynamicModel(CanonicalHemodynamicModel):

    def get_est_fmri_hemodynamic_response(self, est_hemodynamic_response):
        hemodynamic_response_to_eeg = get_hdr_for_eeg(self.eeg.data, est_hemodynamic_response)
        est_fmri = sum_hdr_for_eeg(self.r_fmri, hemodynamic_response_to_eeg)
        return hemodynamic_response_to_eeg, est_fmri
