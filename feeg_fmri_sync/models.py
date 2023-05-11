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
    """
    Canonical hemodynamic response function (HRF) model using a single Gamma Distribution

    h(t>delta)  = ((t-delta)/tau)^alpha * exp(-(t-delta)/tau)
    h(t<=delta) = 0;

    The return value is scaled so that the continuous-time peak = 1.0,
    though the peak of the sampled waveform may not be 1.0.
    """
    save_plot_dir: Optional[str] = None
    delta: Optional[float] = None
    tau: Optional[float] = None
    alpha: Optional[float] = None
    # plot_de_meaned indicates whether a plot should be generated to compare the de-meaned data with the original
    #   data. Useful for other models that filter the raw_data before de-meaning
    plot_de_meaned: bool = True

    def __init__(self, eeg: EEGData, fmri: fMRIData, name: str, n_trs_skipped_at_beginning: int = 1,
                 hemodynamic_response_window: float = 30, display_plot: bool = True,
                 save_plot_dir: Optional[str] = None, de_mean_est_fmri: bool = False,
                 de_mean_input_fmri: bool = False, **kwargs):
        """
        :param eeg: EEG data
        :param fmri: fMRI data
        :param name: name of model
        :param n_trs_skipped_at_beginning: number of TRs skipped at beginning of EEG (number of TRs to drop from
            beginning of fMRI)
        :param hemodynamic_response_window: window of time (in seconds) to use for estimating hemodynamic response
        :param display_plot: whether to display plots (for use in Jupyter notebooks)
        :param save_plot_dir: directory to save plots to (for use in scripts)
        :param de_mean_est_fmri: whether to de-mean (take the z-score) the estimated fMRI data (the data that is
            generated by the hemodynamic model)
        :param de_mean_input_fmri: whether to de-mean (take the z-score) the input fMRI data
        :param kwargs: additional keyword arguments for other models (that support filters) (ignored here)
        """
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
        self.de_mean_input_fmri = de_mean_input_fmri
        self.de_mean_est_fmri = de_mean_est_fmri
        self.fmri = self.de_mean_actual_fmri(fmri.data)

    def de_mean_actual_fmri(self, input_data: npt.NDArray) -> fMRIData:
        """
        If de_mean_input_fmri is True, de-mean the input fMRI data. Otherwise, return the input data unchanged.
        If plot_de_meaned is True, plot the de-meaned data against the original data.
        """
        if self.de_mean_input_fmri:
            de_meaned_data = zscore(input_data, axis=self.raw_fmri.get_tr_axis(), nan_policy='omit')
            de_meaned_fmri = fMRIData(de_meaned_data, self.raw_fmri.TR, self.raw_fmri.voxel_names)
            if self.plot_de_meaned:
                if self.display_plot or self.save_plot_dir:
                    for voxel_name in self.plot_voxels:
                        if self.raw_fmri.is_single_voxel():
                            transformed_voxel_data = de_meaned_fmri.data
                            original_voxel_data = input_data
                        else:
                            i, transformed_voxel_data = de_meaned_fmri.get_voxel_by_name(voxel_name)
                            original_voxel_data = input_data.take(i, axis=de_meaned_fmri.get_voxel_axis()).squeeze()
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
            return de_meaned_fmri
        return fMRIData(input_data, self.raw_fmri.TR, self.raw_fmri.voxel_names)

    def set_save_plot_dir(self, save_plot_dir: Optional[str]) -> None:
        if save_plot_dir:
            if not os.path.exists(save_plot_dir):
                os.makedirs(save_plot_dir)
        self.save_plot_dir = save_plot_dir

    def set_params(self, delta: float, tau: float, alpha: float) -> None:
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

    def __str__(self) -> str:
        return self.__name__

    def set_plot_voxels(self, voxel_names_to_plot: npt.ArrayLike) -> None:
        self.plot_voxels = voxel_names_to_plot

    def plot_hdr_for_eeg(self, hdr: npt.NDArray, fmri_hdr: npt.NDArray) -> None:
        """
        Plot the estimated hemodynamic response from EEG spikes. Includes the fMRI time points
        sampled from the estimated hemodynamic response.
        """
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
            est_fmri: npt.NDArray) -> None:
        """
        Plot the original estimated HDR and the transformed estimated HDR (with filters/de-meaning applied)
        """
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
        """
        Plot the original fMRI and the transformed fMRI (with filters/de-meaning applied)

        actual_fmri_name: name of the actual fMRI data voxel (e.g. '1')
        transformation_str: name of the transformation applied to the fMRI data (e.g. 'Z-scored fMRI')
        """
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
        """
        Plot the estimated fMRI and the actual fMRI. Include the residual values for each time step.

        actual_fmri_name: name of the actual fMRI data voxel (e.g. '1')
        corr_coef: correlation coefficient between the estimated and actual fMRI, if calculated
        pearsons_stat: pearson's statistic between the estimated and actual fMRI, if calculated
        p_value: p-value for the pearson's statistic, if calculated
        """
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

    def get_time_steps(self) -> npt.NDArray:
        """
        Get time steps for Hemodynamic Response Function (HRF) at EEG rate (covers number of seconds in
        hemodynamic response window, default 30sec)
        """
        return np.arange(self.hemodynamic_response_window*self.eeg.sample_frequency + 1) / self.eeg.sample_frequency

    def get_est_hemodynamic_response(self, delta: float, tau: float, alpha: float) -> npt.NDArray:
        """
        Generate estimated hemodynamic response for time steps given parameters delta, tau, alpha
        """
        return get_est_hemodynamic_response(self.get_time_steps(), delta, tau, alpha)

    def get_est_fmri_hemodynamic_response(self,
                                          est_hemodynamic_response: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Get estimated fMRI hemodynamic response from estimated EEG hemodynamic response (de-meaned if specified and
        downsampled to fMRI rate)
        """
        hdr_to_eeg = get_hdr_for_eeg(self.eeg.data, est_hemodynamic_response)
        if self.de_mean_est_fmri:
            transformed_hdr_to_eeg = zscore(hdr_to_eeg, nan_policy='omit')
            est_fmri = downsample_hdr_for_eeg(self.r_fmri, transformed_hdr_to_eeg)
            if self.display_plot or self.save_plot_dir:
                self.compare_transformed_est_eeg_with_est_eeg(transformed_hdr_to_eeg, hdr_to_eeg)
            return transformed_hdr_to_eeg, est_fmri
        est_fmri = downsample_hdr_for_eeg(self.r_fmri, hdr_to_eeg)
        return hdr_to_eeg, est_fmri

    def get_transformation_functions(self, est_fmri: fMRIData) -> Tuple[Callable, Callable]:
        """
        Get transformation functions for estimated and actual fMRI data

        Normally, the EEG spikes had data spanning at least 1 TR removed from its beginning and some number of
        TRs removed from the end, due to noise. This means that the estimated fMRI data will be missing some number
        of TRs at the beginning and end. The transformation functions returned here will transform the estimated
        fMRI data and the actual fMRI data to be the same size and cover the same time period, so they can be
        compared accurately.
        """
        # Only calculate transformation functions if they haven't been calculated yet or if the estimated fMRI has
        # changed size
        if not self.transform_est_fmri or not self.transform_actual_fmri or self.est_fmri_n_trs != est_fmri.get_n_trs():
            if self.est_fmri_n_trs and self.est_fmri_n_trs != est_fmri.get_n_trs():
                print(f'WARNING: estimated fMRI size changed! {self.est_fmri_n_trs} -> {est_fmri.get_n_trs()}')
            tr_axis = self.fmri.get_tr_axis()
            actual_fmri_compression_mask = np.arange(self.fmri.data.shape[tr_axis]) >= self.n_trs_skipped_at_beginning
            self.est_fmri_n_trs = est_fmri.get_n_trs()
            # Default function for estimated fMRI: do nothing
            self.transform_est_fmri = lambda x: x
            # Default function for actual fMRI: remove self.n_trs_skipped_at_beginning from the beginning
            self.transform_actual_fmri = lambda x: x.compress(actual_fmri_compression_mask, axis=tr_axis)
            # If estimated fMRI is larger than the actual fMRI (after the beginning has been truncated)
            # (this isn't expected to happen)
            if est_fmri.get_n_trs() > self.fmri.get_n_trs() - self.n_trs_skipped_at_beginning:
                if self.display_plot:
                    print(f'Estimated fMRI is larger than actual fMRI '
                          f'(-# skipped TRs at beginning of EEG): '
                          f'{est_fmri.get_n_trs()} : {self.fmri.get_n_trs()} - '
                          f'{self.n_trs_skipped_at_beginning}')
                est_fmri_compression_mask = np.arange(self.fmri.data.shape[tr_axis]) < self.fmri.get_n_trs()
                self.transform_est_fmri = lambda x: x.compress(est_fmri_compression_mask, axis=tr_axis)
            # If actual fMRI is larger than the estimated fMRI (after the beginning has been truncated)
            # (this is expected to happen)
            if est_fmri.get_n_trs() < self.fmri.get_n_trs() - self.n_trs_skipped_at_beginning:
                if self.display_plot:
                    print(f'Estimated fMRI is smaller than actual fMRI '
                          f'(-# skipped TRs at beginning of EEG): '
                          f'{est_fmri.get_n_trs()} : {self.fmri.get_n_trs()} '
                          f'- {self.n_trs_skipped_at_beginning}')
                # Combine mask that truncates the beginning with a mask that truncates the end
                actual_fmri_compression_mask = np.logical_and(
                    actual_fmri_compression_mask,
                    np.arange(self.fmri.data.shape[tr_axis]) < (est_fmri.get_n_trs() + self.n_trs_skipped_at_beginning)
                )
                self.transform_actual_fmri = lambda x: x.compress(actual_fmri_compression_mask, axis=tr_axis)
        return self.transform_est_fmri, self.transform_actual_fmri

    def score_from_hemodynamic_response(self, est_hemodynamic_response: npt.NDArray, column: Optional[str],
                                        get_corr_coef: bool = False, get_significance: bool = False) -> Tuple[
            npt.NDArray, fMRIData, npt.NDArray, float, npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Input:
            est_hemodynamic_response: (n_trs) array of estimated hemodynamic response
            column: name of voxel in actual fMRI data to score. If None, will score all voxels

        Output:
            beta: (2 x n_voxels) array of beta values (intercept, slope)
            residual: fMRIData object (n_trs x n_voxels) containing residual values
            residual_variance: (n_voxels) array of residual variance
            degrees_of_freedom: integer
            r (correlation coefficient): (n_voxels) array of correlation coefficients. Will be np.nan if get_corr_coef
                is False
            pearsons_statistic: (n_voxels) array of Pearson's statistic value. Will be np.nan if get_significance
                is False
            pearsons_pvalue: (n_voxels) array of Pearson's p-value. Will be np.nan if get_significance is False
        """
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
            # Only score one voxel
            _, voxel_data = actual_fmri.get_voxel_by_name(column)
            actual_fmri = fMRIData(
                voxel_data,
                self.fmri.TR,
                voxel_names=[column]
            )
        beta, residual, residual_variance, degrees_of_freedom = fit_glm(est_fmri, actual_fmri)
        # Calculate correlation coefficient if get_corr_coef is True
        if get_corr_coef:
            all_fmri = np.concatenate([np.atleast_2d(est_fmri.data), actual_fmri.data])
            nas = np.any(np.isnan(all_fmri), axis=actual_fmri.get_voxel_axis())
            r = np.corrcoef(all_fmri[:, ~nas])[0, 1:]
        else:
            r = np.empty(residual_variance.shape)
            r.fill(np.nan)

        # Calculate Pearson's statistic and p-value if get_significance is True
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

        # Convert residual to fMRIData to allow for easier plotting
        if column:
            residual = fMRIData(residual.squeeze(), TR=self.fmri.TR, voxel_names=[column])
        else:
            residual = fMRIData(residual, TR=self.fmri.TR, voxel_names=self.fmri.voxel_names)
        # Plot results if desired
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
        tau: dispersion
        alpha: exponent

        Output:
            residual_variance: (n_voxels) array of residual variance
        """
        _, _, residual_variance, _, _, _, _ = self.score_detailed(delta, tau, alpha)
        return residual_variance

    def score_detailed(self, delta: float, tau: float, alpha: float, column: Optional[str] = None,
                       get_significance: bool = False) -> Tuple[
            npt.NDArray, fMRIData, npt.NDArray, float, npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Input:
            delta: delay
            tau: dispersion
            alpha: exponent
            column: name of voxel in actual fMRI data to score. If None, will score all voxels

        Output:
            beta: (2 x n_voxels) array of beta values (intercept, slope)
            residual: fMRIData object (n_trs x n_voxels) containing residual values
            residual_variance: (n_voxels) array of residual variance
            degrees_of_freedom: integer
            r (correlation coefficient): (n_voxels) array of correlation coefficients.
            pearsons_statistic: (n_voxels) array of Pearson's statistic value. Will be np.nan if get_significance
                is False
            pearsons_pvalue: (n_voxels) array of Pearson's p-value. Will be np.nan if get_significance is False
        """
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
    """
    Sums the hemodynamic response to EEG data instead of downsampling when converting it to
    fMRI time steps. It is not expected to generate significantly different results than the CanonicalHemodynamicModel
    """
    def get_est_fmri_hemodynamic_response(self, est_hemodynamic_response):
        hemodynamic_response_to_eeg = get_hdr_for_eeg(self.eeg.data, est_hemodynamic_response)
        est_fmri = sum_hdr_for_eeg(self.r_fmri, hemodynamic_response_to_eeg)
        return hemodynamic_response_to_eeg, est_fmri


class SavgolFilterHemodynamicModel(CanonicalHemodynamicModel):
    """
    Applies a Savitzky-Golay filter to the fMRI data hemodynamic response before de-meaning. The filter is not applied
    to the estimated hemodynamic response.
    """
    def __init__(self, eeg: EEGData, fmri: fMRIData, name: str, n_trs_skipped_at_beginning: int = 1,
                 hemodynamic_response_window: float = 30, display_plot: bool = True,
                 save_plot_dir: Optional[str] = None, de_mean_est_fmri: bool = False,
                 de_mean_input_fmri: bool = False, savgol_filter_window_length: int = 5,
                 savgol_filter_polyorder: int = 5, **kwargs):
        self.plot_standardization = False
        super().__init__(eeg, fmri, name, n_trs_skipped_at_beginning, hemodynamic_response_window, display_plot,
                         save_plot_dir, de_mean_est_fmri, de_mean_input_fmri)
        self.plot_standardization = True
        self.savgol_filter_window_length = savgol_filter_window_length
        self.savgol_filter_polyorder = savgol_filter_polyorder
        self.savgol_filter_kwargs = {}
        for kwarg_name, kwarg_val in kwargs.items():
            if kwarg_name in ['deriv', 'delta', 'mode', 'cval']:
                self.savgol_filter_kwargs[kwarg_name] = kwarg_val

        filtered_data = savgol_filter(self.raw_fmri.data, self.savgol_filter_window_length,
                                      self.savgol_filter_polyorder,
                                      axis=self.raw_fmri.get_tr_axis(), **self.savgol_filter_kwargs)
        # Plot if configured
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
        # De-mean the filtered data
        self.fmri = self.de_mean_actual_fmri(filtered_data)


class GaussianFilterHemodynamicModel(CanonicalHemodynamicModel):
    """
    Applies a 1D Gaussian filter to the fMRI data hemodynamic response before de-meaning. The filter is not applied
    to the estimated hemodynamic response.
    """
    def __init__(self, eeg: EEGData, fmri: fMRIData, name: str, n_trs_skipped_at_beginning: int = 1,
                 hemodynamic_response_window: float = 30, display_plot: bool = True,
                 save_plot_dir: Optional[str] = None, de_mean_est_fmri: bool = False,
                 de_mean_input_fmri: bool = False, gaussian_filter_sigma: float = 5,
                 **kwargs):
        self.plot_standardization = False
        super().__init__(eeg, fmri, name, n_trs_skipped_at_beginning, hemodynamic_response_window, display_plot,
                         save_plot_dir, de_mean_est_fmri, de_mean_input_fmri)
        self.plot_standardization = True
        self.gaussian_filter_sigma = gaussian_filter_sigma
        self.gaussian_filter_kwargs = {}
        for kwarg_name, kwarg_val in kwargs.items():
            if kwarg_name in ['order', 'mode', 'cval', 'truncate', 'radius']:
                self.gaussian_filter_kwargs[kwarg_name] = kwarg_val
        filtered_data = gaussian_filter1d(self.raw_fmri.data, self.gaussian_filter_sigma,
                                          axis=self.raw_fmri.get_tr_axis(), **self.gaussian_filter_kwargs)
        # Plot if configured
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
        self.fmri = self.de_mean_actual_fmri(filtered_data)
