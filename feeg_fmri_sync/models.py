import os

import numpy as np
import numpy.typing as npt

from typing import Callable, Optional, Tuple

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import zscore, pearsonr

from feeg_fmri_sync.constants import (
    EEGData,
    fMRIData
)
from feeg_fmri_sync.plotting import generate_latex_label
from feeg_fmri_sync.utils import (
    get_hdr_for_eeg,
    sum_hdr_for_eeg, get_ratio_eeg_freq_to_fmri_freq, get_est_hemodynamic_response, downsample_hdr_for_eeg, fit_glm,
    get_b1_contrast_matrix, get_b0_contrast_matrix,
)


class CanonicalHemodynamicModel:
    save_plot_dir: Optional[str] = None
    delta: Optional[float] = None
    tau: Optional[float] = None
    alpha: Optional[float] = None
    plot_standardization: bool = True

    def __init__(self, eeg: EEGData, fmri: fMRIData, name: str, n_trs_skipped_at_beginning: int = 1,
                 hemodynamic_response_window: float = 30, display_plot: bool = True,
                 save_plot_dir: Optional[str] = None, standardize: bool = False, **kwargs):
        # Data
        self.eeg: EEGData = eeg
        self.raw_fmri: fMRIData = fmri

        # Ratio of eeg freq to fMRI frequency
        self.r_fmri: float = get_ratio_eeg_freq_to_fmri_freq(self.eeg.sample_frequency, fmri.TR)
        # Number of TRs skipped at beginning of EEG
        self.n_trs_skipped_at_beginning: int = n_trs_skipped_at_beginning

        # Tunable parameters
        self.hemodynamic_response_window: float = hemodynamic_response_window  # seconds

        # Plotting parameters
        self.name: str = name
        self.display_plot: bool = display_plot
        self.set_save_plot_dir(save_plot_dir)
        voxel_name, _ = fmri.get_voxel_i(0)
        self.plot_voxels = [voxel_name]
        self.figures_to_plot = []

        # internal tracking for better performance
        self.transform_est_fmri: Optional[Callable] = None
        self.transform_actual_fmri: Optional[Callable] = None
        self.est_fmri_size: Optional[int] = None
        self.est_fmri_n_trs: Optional[int] = None

        # z-scoring and filtering setup
        self.standardize = standardize
        self.fmri = self.standardize_actual_fmri(fmri.data)

    def standardize_actual_fmri(self, input_data: npt.NDArray) -> fMRIData:
        if self.standardize:
            standardized_data = zscore(input_data, axis=self.raw_fmri.get_tr_axis(), nan_policy='omit')
            standardized_fmri = fMRIData(standardized_data, self.raw_fmri.TR, self.raw_fmri.voxel_names)
            if self.plot_standardization:
                if self.display_plot or self.save_plot_dir:
                    for voxel_name in self.plot_voxels:
                        if self.raw_fmri.is_single_voxel():
                            transformed_voxel_data = standardized_fmri.data
                            original_voxel_data = input_data
                        else:
                            i, transformed_voxel_data = standardized_fmri.get_voxel_by_name(voxel_name)
                            original_voxel_data = input_data.take(i, axis=standardized_fmri.get_voxel_axis()).squeeze()
                        self.compare_transformed_fmri_with_actual(transformed_voxel_data, original_voxel_data,
                                                                  transformation_str='Z-scored fMRI')
                if self.display_plot:
                    plt.show()
                elif self.save_plot_dir:
                    with PdfPages(
                            os.path.join(self.save_plot_dir, f'{self.name}_cmp_transformed_fmri_with_actual.pdf')) as pdf:
                        for fig in self.figures_to_plot:
                            pdf.savefig(fig, bbox_inches='tight')
                plt.close('all')
            return standardized_fmri
        return fMRIData(input_data, self.raw_fmri.TR, self.raw_fmri.voxel_names)

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

    def get_param_str(self) -> str:
        return f'{generate_latex_label("delta")} = {self.delta: .4f}, ' \
               f'{generate_latex_label("tau")} = {self.tau: .4f}, {generate_latex_label("alpha")} = {self.alpha: .4f}, '

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
        plt.title(f'Estimated hemodynamic response (EEG and fMRI time scales) from EEG spikes using \n'
                  f'model={self.name}, {self.get_param_str()}')
        plt.legend()
        self.figures_to_plot.append(plt.gcf())

    def compare_transformed_est_eeg_with_est_eeg(
            self,
            transformed_est_fmri: npt.NDArray,
            est_fmri: npt.NDArray):
        plt.figure()
        time_steps_for_eeg = np.arange(len(self.eeg.data)) / self.eeg.sample_frequency
        plt.plot(time_steps_for_eeg, self.eeg.data, label='EEG spikes', alpha=0.25)
        plt.plot(time_steps_for_eeg, est_fmri, label='Original estimated fMRI', alpha=0.5)
        plt.plot(time_steps_for_eeg, transformed_est_fmri, label='Transformed estimated fMRI', alpha=0.5)
        plt.title(f'Estimated fMRI HDR from EEG spikes before and after transformation \n'
                  f'using model={self.name}, {self.get_param_str()}')
        plt.legend()
        self.figures_to_plot.append(plt.gcf())

    def compare_transformed_fmri_with_actual(
            self,
            transformed_fmri: npt.NDArray,
            actual_fmri: npt.NDArray,
            actual_fmri_name: Optional[str] = None,
            transformation_str: Optional[str] = None):
        plt.figure()
        if not transformation_str:
            transformation_str = 'Transformed fMRI'
        time_steps_for_eeg = np.arange(len(self.eeg.data)) / self.eeg.sample_frequency
        time_steps_for_fmri = time_steps_for_eeg[::self.r_fmri]
        if time_steps_for_fmri.size != actual_fmri.size:
            plt.plot(np.arange(actual_fmri.size), actual_fmri, label='Actual fMRI', alpha=0.5)
            plt.plot(np.arange(actual_fmri.size), transformed_fmri, label=transformation_str, alpha=0.5)
        else:
            plt.plot(time_steps_for_fmri, actual_fmri, label='Actual fMRI', alpha=0.5)
            plt.plot(time_steps_for_fmri, transformed_fmri, label=transformation_str, alpha=0.5)

        if not actual_fmri_name:
            title = f'{transformation_str} compared with actual fMRI using \n' \
                    f'model={self.name}'
        else:
            title = f'{transformation_str} compared with actual fMRI ({actual_fmri_name}) \n' \
                    f'using model={self.name}'
        plt.title(title)
        plt.legend()
        self.figures_to_plot.append(plt.gcf())

    def compare_est_fmri_with_actual(self, est_fmri: npt.NDArray, actual_fmri: npt.NDArray, residual: npt.NDArray,
                                     residual_variance: float, actual_fmri_name: Optional[str] = None, corr_coef=None,
                                     pearsons_stat=None, p_value=None):
        plt.figure()
        time_steps_for_eeg = np.arange(len(self.eeg.data)) / self.eeg.sample_frequency
        time_steps_for_fmri = time_steps_for_eeg[::self.r_fmri]
        plt.plot(time_steps_for_eeg, self.eeg.data, label='EEG spikes', alpha=0.25)
        if residual is not None:
            x_nan = np.isnan(est_fmri)
            plt.plot(time_steps_for_fmri[~x_nan], residual, '.', label='Residual')
            xlabel = f'Residual Variance is {residual_variance:.6f}'
            if corr_coef and not np.isnan(corr_coef):
                xlabel = f'{xlabel}, Correlation Coef={corr_coef:.6f}'
            if pearsons_stat and not np.isnan(pearsons_stat):
                xlabel = f'{xlabel}, Pearsons Stat={pearsons_stat:.6f}'
            if p_value and not np.isnan(p_value):
                xlabel = f'{xlabel}, Pvalue={p_value:.6f}'
            plt.xlabel(xlabel)
        plt.plot(time_steps_for_fmri, actual_fmri, label='Actual fMRI', alpha=0.5)
        plt.plot(time_steps_for_fmri, est_fmri, label='Estimated fMRI', alpha=0.5)

        if not actual_fmri_name:
            title = f'Estimated fMRI HDR from EEG spikes compared with actual fMRI using \n' \
                    f'model={self.name}, {self.get_param_str()}'
        else:
            title = f'Estimated fMRI HDR from EEG spikes compared with actual fMRI ({actual_fmri_name}) \n' \
                    f'using model={self.name}, {self.get_param_str()}'
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
        hdr_to_eeg = get_hdr_for_eeg(self.eeg.data, est_hemodynamic_response)
        if self.standardize:
            transformed_hdr_to_eeg = zscore(hdr_to_eeg, nan_policy='omit')
            est_fmri = downsample_hdr_for_eeg(self.r_fmri, transformed_hdr_to_eeg)
            if self.display_plot or self.save_plot_dir:
                self.compare_transformed_est_eeg_with_est_eeg(transformed_hdr_to_eeg, hdr_to_eeg)
            return transformed_hdr_to_eeg, est_fmri
        est_fmri = downsample_hdr_for_eeg(self.r_fmri, hdr_to_eeg)
        return hdr_to_eeg, est_fmri

    def get_transformation_functions(self, est_fmri: fMRIData) -> Tuple[Callable, Callable]:
        if not self.transform_est_fmri or not self.transform_actual_fmri or self.est_fmri_n_trs != est_fmri.get_n_trs():
            if self.est_fmri_n_trs and self.est_fmri_n_trs != est_fmri.get_n_trs():
                print(f'WARNING: estimated fMRI size changed! {self.est_fmri_n_trs} -> {est_fmri.get_n_trs()}')
            tr_axis = self.fmri.get_tr_axis()
            actual_fmri_compression_mask = np.arange(self.fmri.data.shape[tr_axis]) >= self.n_trs_skipped_at_beginning
            self.est_fmri_n_trs = est_fmri.get_n_trs()
            self.transform_est_fmri = lambda x: x
            self.transform_actual_fmri = lambda x: x.compress(actual_fmri_compression_mask, axis=tr_axis)
            if est_fmri.get_n_trs() > self.fmri.get_n_trs() - self.n_trs_skipped_at_beginning:
                if self.display_plot:
                    print(f'Estimated fMRI is larger than actual fMRI '
                          f'(-# skipped TRs at beginning of EEG): '
                          f'{est_fmri.get_n_trs()} : {self.fmri.get_n_trs()} - '
                          f'{self.n_trs_skipped_at_beginning}')
                est_fmri_compression_mask = np.arange(self.fmri.data.shape[tr_axis]) < self.fmri.get_n_trs()
                self.transform_est_fmri = lambda x: x.compress(est_fmri_compression_mask, axis=tr_axis)
            if est_fmri.get_n_trs() < self.fmri.get_n_trs() - self.n_trs_skipped_at_beginning:
                if self.display_plot:
                    print(f'Estimated fMRI is smaller than actual fMRI '
                          f'(-# skipped TRs at beginning of EEG): '
                          f'{est_fmri.get_n_trs()} : {self.fmri.get_n_trs()} '
                          f'- {self.n_trs_skipped_at_beginning}')
                actual_fmri_compression_mask = np.logical_and(
                    actual_fmri_compression_mask,
                    np.arange(self.fmri.data.shape[tr_axis]) < (est_fmri.get_n_trs() + self.n_trs_skipped_at_beginning)
                )
                self.transform_actual_fmri = lambda x: x.compress(actual_fmri_compression_mask, axis=tr_axis)
        return self.transform_est_fmri, self.transform_actual_fmri

    def score_from_hemodynamic_response(self, est_hemodynamic_response: npt.NDArray, column: Optional[str],
                                        get_corr_coef: bool = False, get_significance: bool = False) -> Tuple[
            npt.NDArray, fMRIData, npt.NDArray, float, npt.NDArray, npt.NDArray, npt.NDArray]:
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
        if get_corr_coef:
            all_fmri = np.concatenate([np.atleast_2d(est_fmri.data), actual_fmri.data])
            nas = np.any(np.isnan(all_fmri), axis=actual_fmri.get_voxel_axis())
            r = np.corrcoef(all_fmri[:, ~nas])[0, 1:]
        else:
            r = np.empty(residual_variance.shape)
            r.fill(np.nan)
        pearsons_statistic = np.empty(residual_variance.shape)
        pearsons_pvalue = np.empty(residual_variance.shape)
        if get_significance:
            for voxel_name in actual_fmri.voxel_names:
                i, voxel_data = actual_fmri.get_voxel_by_name(voxel_name)
                nas = np.logical_or(np.isnan(est_fmri.data), np.isnan(voxel_data))
                res = pearsonr(est_fmri.data[~nas], voxel_data[~nas])
                pearsons_statistic[i] = res.statistic
                pearsons_pvalue[i] = res.pvalue
        else:
            pearsons_statistic.fill(np.nan)
            pearsons_pvalue.fill(np.nan)
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
                    corr_coef = r.item()
                    pearsons_stat = pearsons_statistic.item()
                    p_value = pearsons_pvalue.item()
                else:
                    i, voxel_data = actual_fmri.get_voxel_by_name(voxel_name)
                    _, residual_data = residual.get_voxel_by_name(voxel_name)
                    residual_variance_at_i = residual_variance[i].item()
                    corr_coef = r[i].item()
                    pearsons_stat = pearsons_statistic[i].item()
                    p_value = pearsons_pvalue[i].item()
                beta_1_val = np.matmul(get_b1_contrast_matrix(), beta)[i].item()
                beta_0_val = np.matmul(get_b0_contrast_matrix(), beta)[i].item()
                self.compare_est_fmri_with_actual(beta_0_val + beta_1_val * est_fmri.data, voxel_data, residual_data,
                                                  residual_variance_at_i, actual_fmri_name=voxel_name,
                                                  corr_coef=corr_coef, pearsons_stat=pearsons_stat, p_value=p_value)
                if self.display_plot:
                    print(f'Residual Variance is {residual_variance_at_i:.6f}, Corr coef={corr_coef:.6f}, '
                          f'Pearsons Stat={pearsons_stat:.6f}, pvalue={p_value:.6f}')
        return beta, residual, residual_variance, degrees_of_freedom, r, pearsons_statistic, pearsons_pvalue

    def score(self, delta: float, tau: float, alpha: float):
        """
        delta: delay
        tau:
        alpha: exponent
        """
        _, _, residual_variance, _, _, _, _ = self.score_detailed(delta, tau, alpha)
        return residual_variance

    def score_detailed(self, delta: float, tau: float, alpha: float, column: Optional[str] = None,
                       get_significance: bool = False) -> Tuple[
            npt.NDArray, fMRIData, npt.NDArray, float, npt.NDArray, npt.NDArray, npt.NDArray]:
        self.set_params(delta, tau, alpha)
        if self.display_plot:
            name = self.name
            if column:
                name = f'{self.name}, column {column}'
            print(f"Scoring {name}: delta={delta:.4f}, tau={tau:.4f}, alpha={alpha:.4f}")
        est_hemodynamic_response = self.get_est_hemodynamic_response(delta, tau, alpha)
        info = self.score_from_hemodynamic_response(est_hemodynamic_response, column,
                                                    get_corr_coef=True, get_significance=get_significance)
        beta, residual, residual_variance, dof, r, pearsons_statistic, pearsons_pvalue = info
        if self.display_plot:
            plt.show()
        elif self.save_plot_dir:
            with PdfPages(os.path.join(self.save_plot_dir, f'{self.name}_cmp_est_fmri_with_actual.pdf')) as pdf:
                for fig in self.figures_to_plot:
                    pdf.savefig(fig, bbox_inches='tight')
        plt.close('all')
        self.unset_params()
        return beta, residual, residual_variance, dof, r, pearsons_statistic, pearsons_pvalue


class HemodynamicModelSumEEG(CanonicalHemodynamicModel):

    def get_est_fmri_hemodynamic_response(self, est_hemodynamic_response):
        hemodynamic_response_to_eeg = get_hdr_for_eeg(self.eeg.data, est_hemodynamic_response)
        est_fmri = sum_hdr_for_eeg(self.r_fmri, hemodynamic_response_to_eeg)
        return hemodynamic_response_to_eeg, est_fmri


class SavgolFilterHemodynamicModel(CanonicalHemodynamicModel):
    def __init__(self, eeg: EEGData, fmri: fMRIData, name: str, n_trs_skipped_at_beginning: int = 1,
                 hemodynamic_response_window: float = 30, display_plot: bool = True,
                 save_plot_dir: Optional[str] = None, standardize: bool = False, savgol_filter_window_length: int = 5,
                 savgol_filter_polyorder: int = 5, **kwargs):
        self.plot_standardization = False
        super().__init__(eeg, fmri, name, n_trs_skipped_at_beginning, hemodynamic_response_window, display_plot,
                         save_plot_dir, standardize)
        self.plot_standardization = True
        self.savgol_filter_window_length = savgol_filter_window_length
        self.savgol_filter_polyorder = savgol_filter_polyorder
        self.savgol_filter_kwargs = {}
        for kwarg_name, kwarg_val in kwargs.items():
            if kwarg_name in ['deriv', 'delta', 'mode', 'cval']:
                self.savgol_filter_kwargs[kwarg_name] = kwarg_val

        filtered_data = savgol_filter(self.raw_fmri.data, self.savgol_filter_window_length, self.savgol_filter_polyorder,
                                      axis=self.raw_fmri.get_tr_axis(), **self.savgol_filter_kwargs)
        if self.display_plot or self.save_plot_dir:
            for voxel_name in self.plot_voxels:
                if self.raw_fmri.is_single_voxel():
                    original_voxel_data = self.raw_fmri.data
                    transformed_voxel_data = filtered_data
                else:
                    i, original_voxel_data = self.raw_fmri.get_voxel_by_name(voxel_name)
                    transformed_voxel_data = filtered_data.take(i, axis=self.raw_fmri.get_voxel_axis()).squeeze()
                self.compare_transformed_fmri_with_actual(transformed_voxel_data, original_voxel_data,
                                                          transformation_str='Savgol Filtered fMRI')

        if self.display_plot:
            plt.show()
        elif self.save_plot_dir:
            with PdfPages(
                    os.path.join(self.save_plot_dir, f'{self.name}_cmp_savgol_filtered_fmri_with_actual.pdf')) as pdf:
                for fig in self.figures_to_plot:
                    pdf.savefig(fig, bbox_inches='tight')
        plt.close('all')
        self.fmri = self.standardize_actual_fmri(filtered_data)


class GaussianFilterHemodynamicModel(CanonicalHemodynamicModel):
    def __init__(self, eeg: EEGData, fmri: fMRIData, name: str, n_trs_skipped_at_beginning: int = 1,
                 hemodynamic_response_window: float = 30, display_plot: bool = True,
                 save_plot_dir: Optional[str] = None, standardize: bool = False, gaussian_filter_sigma: float = 5,
                 **kwargs):
        self.plot_standardization = False
        super().__init__(eeg, fmri, name, n_trs_skipped_at_beginning, hemodynamic_response_window, display_plot,
                         save_plot_dir, standardize)
        self.plot_standardization = True
        self.gaussian_filter_sigma = gaussian_filter_sigma
        self.gaussian_filter_kwargs = {}
        for kwarg_name, kwarg_val in kwargs.items():
            if kwarg_name in ['order', 'mode', 'cval', 'truncate', 'radius']:
                self.gaussian_filter_kwargs[kwarg_name] = kwarg_val
        filtered_data = gaussian_filter1d(self.raw_fmri.data, self.gaussian_filter_sigma,
                                          axis=self.raw_fmri.get_tr_axis(), **self.gaussian_filter_kwargs)
        if self.display_plot or self.save_plot_dir:
            for voxel_name in self.plot_voxels:
                if self.raw_fmri.is_single_voxel():
                    original_voxel_data = self.raw_fmri.data
                    transformed_voxel_data = filtered_data
                else:
                    i, original_voxel_data = self.raw_fmri.get_voxel_by_name(voxel_name)
                    transformed_voxel_data = filtered_data.take(i, axis=self.raw_fmri.get_voxel_axis()).squeeze()
                self.compare_transformed_fmri_with_actual(transformed_voxel_data, original_voxel_data,
                                                          transformation_str='Gaussian filtered fMRI')
        if self.display_plot:
            plt.show()
        elif self.save_plot_dir:
            with PdfPages(
                    os.path.join(self.save_plot_dir, f'{self.name}_cmp_gaussian_filtered_fmri_with_actual.pdf')) as pdf:
                for fig in self.figures_to_plot:
                    pdf.savefig(fig, bbox_inches='tight')
        plt.close('all')
        self.fmri = self.standardize_actual_fmri(filtered_data)
