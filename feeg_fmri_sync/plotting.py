from datetime import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy
import warnings

from collections import defaultdict
from matplotlib import cm
from typing import Optional, Tuple, Dict, List, Callable

from matplotlib.backends.backend_pdf import PdfPages

from feeg_fmri_sync.constants import EEGData, PLOT_DELTA, PLOT_TAU, PLOT_ALPHA
from feeg_fmri_sync.utils import get_ratio_eeg_freq_to_fmri_freq, get_est_hemodynamic_response, get_hdr_for_eeg, \
    downsample_hdr_for_eeg


def plot_hdr(time_steps, hdr):
    plt.cla()
    plt.plot(time_steps, hdr)
    plt.show()


def plot_hdr_for_eeg(eeg_data: EEGData, hdr: npt.NDArray, tr: Optional[int] = None,
                     fmri_hdr: Optional[npt.NDArray] = None):
    plt.cla()
    time_steps_for_eeg = np.arange(len(eeg_data.data)) / eeg_data.sample_frequency
    if fmri_hdr is not None and tr:
        r_fmri = get_ratio_eeg_freq_to_fmri_freq(eeg_data.sample_frequency, tr)
        time_steps_for_fmri = time_steps_for_eeg[::r_fmri]
        plt.plot(time_steps_for_fmri, fmri_hdr, '.-', label='HDR-fMRI')
    elif fmri_hdr is not None:
        print("Unable to plot fMRI Hemodynamic response - require TR")
    plt.plot(time_steps_for_eeg, hdr, label='HDR-EEG')
    plt.plot(time_steps_for_eeg, eeg_data.data, label='EEG spikes')
    plt.legend()
    plt.show()


def compare_est_fmri_with_actual(
        est_fmri: npt.NDArray,
        actual_fmri: npt.NDArray,
        eeg_data: EEGData,
        tr: float,
        residual: Optional[npt.NDArray] = None,
        title: Optional[str] = None,
        est_fmri_label: str = 'Estimated fMRI',
        actual_fmri_label: str = 'Actual fMRI'):
    """"""
    plt.cla()
    time_steps_for_eeg = np.arange(len(eeg_data.data)) / eeg_data.sample_frequency
    r_fmri = get_ratio_eeg_freq_to_fmri_freq(eeg_data.sample_frequency, tr)
    time_steps_for_fmri = time_steps_for_eeg[::r_fmri]
    if residual is not None:
        X_nan = np.isnan(est_fmri)
        plt.plot(time_steps_for_fmri[~X_nan], residual, '.', label='Residual')
    plt.plot(time_steps_for_fmri, est_fmri, label=est_fmri_label)
    plt.plot(time_steps_for_fmri, actual_fmri, label=actual_fmri_label)
    plt.plot(time_steps_for_eeg, eeg_data.data, label='EEG spikes')
    if title:
        plt.title(title)
    plt.legend()
    plt.show()


def plot_all_search_results_2d(df, separate_by='alpha', save_path: Optional[str] = None):
    """Assumes df was created with
    for d in delta:
        for t in tau:
            for a in alpha:
    And that all lists are ascending
    """
    df = df.astype(float)
    dta = ['delta', 'tau', 'alpha']
    if separate_by not in dta:
        raise ValueError(f'separate_by ({separate_by}) must be in {dta}')
    values_to_plot = df.columns[~np.isin(df.columns, dta)]
    vmin = np.min(df[values_to_plot].min())
    vmax = np.max(df[values_to_plot].max())
    subfigure_separator = np.unique(df[separate_by])
    n_subplot_rows, n_subplot_columns = get_subplot_axes(subfigure_separator)
    dta.remove(separate_by)
    x_label = dta[0]
    y_label = dta[1]
    for column in values_to_plot:
        fig, axs = plt.subplots(n_subplot_rows, n_subplot_columns)
        fig.suptitle(column)
        fig.tight_layout()
        for i, d in enumerate(subfigure_separator):
            if n_subplot_rows == n_subplot_columns == 1:
                ax = axs
            else:
                ax = axs.flatten()[i]
            ax.set_title(f'{separate_by} = {d:.2f}')
            ax.set_xlabel(f'{x_label}')
            ax.set_ylabel(f'{y_label}')
            small_df = df[df[separate_by] == d]
            x_length = len(np.unique(small_df[x_label]))
            y_length = len(np.unique(small_df[y_label]))
            if x_length != y_length:
                warnings.warn(f'Code was not tested on data with different length {x_length} and {y_length}')
            X = np.reshape(small_df[x_label].values, (x_length, y_length))
            if not np.apply_along_axis(lambda x: np.isclose(x, x[0]).all(), 1, X).all():
                raise ValueError('df violates order expectations. Plotting is not safe')
            Y = np.reshape(small_df[y_label].values, (x_length, y_length))
            if not np.isclose(Y, Y[0]).all():
                raise ValueError('df violates order expectations. Plotting is not safe')
            Z = np.reshape(small_df[column].values, (x_length, y_length))
            cf = ax.contourf(X, Y, Z, cmap=cm.gist_earth, vmin=vmin, vmax=vmax)
            ax.set_xlim([np.min(small_df[x_label]), np.max(small_df[x_label])])
            ax.set_ylim([np.min(small_df[y_label]), np.max(small_df[y_label])])
        plt.colorbar(cf, ax=axs.ravel().tolist())
        if not save_path:
            print(
                f'Minimal Cost for {column} = {df[column].min()}; at\n{df[df[column] == df[column].min()][["delta", "tau", "alpha"]]}')
            plt.show()
        else:
            plt.savefig(f'{save_path}_voxel{column}.pdf')
            plt.close()


def generate_latex_label(text, add_dollars=True):
    if text in ['tau', 'alpha']:
        return f'{"$" if add_dollars else ""}\\{text}{"$" if add_dollars else ""}'
    return f'{"$" if add_dollars else ""}\{text}{"$" if add_dollars else ""}'


def plot_all_search_results_2d_on_diff_colormaps(df, separate_by='alpha', verbose=True):
    """Assumes df was created with
    for d in delta:
        for t in tau:
            for a in alpha:
    And that all lists are ascending
    """
    df = df.astype(float)
    dta = ['delta', 'tau', 'alpha']
    if separate_by not in dta:
        raise ValueError(f'separate_by ({separate_by}) must be in {dta}')
    values_to_plot = df.columns[~np.isin(df.columns, dta)]

    subfigure_separator = np.unique(df[separate_by])
    n_subplot_rows, n_subplot_columns = get_subplot_axes(subfigure_separator)
    dta.remove(separate_by)
    x_label = dta[0]
    y_label = dta[1]
    figs = []
    for column in values_to_plot:
        fig, axs = plt.subplots(n_subplot_rows, n_subplot_columns)
        fig.suptitle(column)
        fig.tight_layout()
        vmin = np.min(df[column].min())
        vmax = np.max(df[column].max())
        for i, d in enumerate(subfigure_separator):
            if n_subplot_rows == n_subplot_columns == 1:
                ax = axs
            else:
                ax = axs.flatten()[i]
            ax.set_title(f'{generate_latex_label(separate_by)} = {d:.2f}')
            if i / n_subplot_columns in range(n_subplot_rows):
                ax.set_ylabel(generate_latex_label(y_label))
            if i / n_subplot_columns == n_subplot_rows - 1:
                ax.set_xlabel(generate_latex_label(x_label))
            small_df = df[df[separate_by] == d]
            x_length = len(np.unique(small_df[x_label]))
            y_length = len(np.unique(small_df[y_label]))
            if x_length != y_length:
                warnings.warn(f'Code was not tested on data with different length {x_length} and {y_length}')
            X = np.reshape(small_df[x_label].values, (x_length, y_length))
            if not np.apply_along_axis(lambda x: np.isclose(x, x[0]).all(), 1, X).all():
                raise ValueError('df violates order expectations. Plotting is not safe')
            Y = np.reshape(small_df[y_label].values, (x_length, y_length))
            if not np.isclose(Y, Y[0]).all():
                raise ValueError('df violates order expectations. Plotting is not safe')
            Z = np.reshape(small_df[column].values, (x_length, y_length))
            cf = ax.contourf(X, Y, Z, cmap=cm.gist_earth, vmin=vmin, vmax=vmax)
            ax.set_xlim([np.min(small_df[x_label]), np.max(small_df[x_label])])
            ax.set_ylim([np.min(small_df[y_label]), np.max(small_df[y_label])])
        plt.colorbar(cf, ax=axs.ravel().tolist())
        figs.append(fig)
        if verbose:
            print(
                f'Minimal Cost for {column} = {df[column].min()}; at\n{df[df[column] == df[column].min()][["delta", "tau", "alpha"]]}')
    return figs


def plot_gradient_2d(grad: npt.NDArray,
                     pts: npt.NDArray,
                     grad_label: str,
                     pts_labels: Tuple[str] = ('delta', 'tau', 'alpha'),
                     z_index: int = 2):
    if len(pts_labels) != 3:
        raise ValueError(f"Expect 3 points labels, not {len(pts_labels)} - is the data malformed?")
    x_index = 0 if z_index > 0 else 1
    y_index = x_index + 2 if z_index == 1 else x_index + 1
    z_label = pts_labels[z_index]
    z_unique_values = np.unique(pts[z_index])
    n_subplot_rows, n_subplot_columns = get_subplot_axes(
        z_unique_values
    )
    x_label = pts_labels[x_index]
    y_label = pts_labels[y_index]
    fig, axs = plt.subplots(n_subplot_rows, n_subplot_columns)
    fig.suptitle(grad_label)
    fig.tight_layout()
    for i, d in enumerate(z_unique_values):
        if n_subplot_rows == n_subplot_columns == 1:
            ax = axs
        else:
            ax = axs.flatten()[i]
        ax.set_title(f'{z_label} = {d:.2f}')
        ax.set_xlabel(f'{x_label}')
        ax.set_ylabel(f'{y_label}')
        X = np.take(pts[x_index], i, axis=z_index)
        Y = np.take(pts[y_index], i, axis=z_index)
        grad_at_z = np.take(grad, i, axis=z_index)
        cf = ax.contourf(
            X,
            Y,
            grad_at_z,
            cmap=cm.gist_earth,
            vmin=np.min(grad),
            vmax=np.max(grad),
        )
        fig.colorbar(cf, ax=ax)
    plt.show()


def plot_local_minima(df):
    local_minima_for_model_column = defaultdict(list)
    for model in df['model_name'].unique():
        portion_df = df[df['model_name'] == model].drop(columns='model_name')
        delta_unique_values = portion_df['delta'].unique()
        tau_unique_values = portion_df['tau'].unique()
        alpha_unique_values = portion_df['alpha'].unique()
        new_shape = (delta_unique_values.size, tau_unique_values.size, alpha_unique_values.size)
        delta_pts = np.reshape(portion_df['delta'].values, new_shape)
        tau_pts = np.reshape(portion_df['tau'].values, new_shape)
        alpha_pts = np.reshape(portion_df['alpha'].values, new_shape)
        for column in portion_df.columns.drop(['delta', 'tau', 'alpha']):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_title(column)
            ax.set_xlabel(f'delta')
            ax.set_ylabel(f'tau')
            ax.set_zlabel('alpha')
            ax.set_xlim([np.min(delta_pts), np.max(delta_pts)])
            ax.set_ylim([np.min(tau_pts), np.max(tau_pts)])
            ax.set_zlim([np.min(alpha_pts), np.max(alpha_pts)])
            m = np.reshape(
                portion_df[column].values,
                new_shape
            )
            f1 = np.ones((3, 3, 3))
            f1[1, 1, 1] = 0
            is_minima = m < scipy.ndimage.minimum_filter(m, footprint=f1, mode='constant', cval=np.inf)
            for point in zip(*np.where(is_minima)):
                local_minima_for_model_column[f'{model}_{column}'].append(
                    (delta_pts[point], tau_pts[point], alpha_pts[point]))
                ax.scatter(delta_pts[point], tau_pts[point], alpha_pts[point], label=f'Cost={m[point]}')
            ax.legend()
            plt.show()

    return local_minima_for_model_column


def get_subplot_axes(delta_range):
    n_subplot_rows = 1
    n_subplot_columns = 1
    smallest_number_possible = 1
    while len(delta_range) > smallest_number_possible:
        if n_subplot_rows == n_subplot_columns:
            n_subplot_rows += 1
        else:
            n_subplot_columns += 1
        smallest_number_possible = n_subplot_rows * n_subplot_columns
    return n_subplot_rows, n_subplot_columns


def plot_eeg_hdrs_across_range(
        x_label: str,
        x_range: npt.ArrayLike,
        y_label: str,
        y_range: npt.ArrayLike,
        z_label: str,
        z_range: npt.ArrayLike,
        fmri_hdr_lookup: Dict[Tuple, npt.ArrayLike],
        fmri_time_steps: npt.ArrayLike,
        x_start: int = 100,
        x_length: int = 10) -> plt.Figure:
    # Zoom in since change is very small
    fig, axs = plt.subplots(3, 3, sharex='all', sharey='all', figsize=(14, 8))
    fig.supxlabel(f'${generate_latex_label(x_label, add_dollars=False)} \longrightarrow$', fontsize=24)
    fig.supylabel(f'${generate_latex_label(y_label, add_dollars=False)} \longrightarrow$', fontsize=24)
    for i, tau in enumerate(reversed(y_range)):
        for j, alpha in enumerate(x_range):
            ax = axs[i][j]
            ax.set_title(f'${generate_latex_label(y_label, add_dollars=False)}={tau:.2f}, '
                         f'{generate_latex_label(x_label, add_dollars=False)}={alpha:.2f}$')
            for delta in z_range:
                fmri_hdr_for_eeg = fmri_hdr_lookup[(delta, tau, alpha)]
                ax.plot(
                    fmri_time_steps,
                    fmri_hdr_for_eeg,
                    label=f'${generate_latex_label(z_label, add_dollars=False)}={delta:.2f}$',
                    linewidth=0.5,
                )
            ax.set_xlim(x_start, x_start+x_length)
            if i == 2:
                ax.set_xlabel('Time')
            if j == 0:
                ax.set_ylabel('Expected fMRI signal')
    handles, labels = axs[2][2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    return fig


def plot_eeg_hdr_across_delta_tau_alpha_range(eeg: EEGData, hdr_window: float, tr: float,
                                              delta_range: npt.ArrayLike, tau_range: npt.ArrayLike,
                                              alpha_range: npt.ArrayLike, verbose: bool = True) -> List[plt.Figure]:
    delta_range = np.round(delta_range, 4)
    tau_range = np.round(tau_range, 4)
    alpha_range = np.round(alpha_range, 4)
    sparse_delta_range = [delta_range[0], PLOT_DELTA, delta_range[-1]]
    sparse_tau_range = [tau_range[0], PLOT_TAU, tau_range[-1]]
    sparse_alpha_range = [alpha_range[0], PLOT_ALPHA, alpha_range[-1]]
    time_steps = np.arange(hdr_window * eeg.sample_frequency + 1) / eeg.sample_frequency
    r_fmri = get_ratio_eeg_freq_to_fmri_freq(eeg.sample_frequency, tr)
    time_steps_for_eeg = np.arange(len(eeg.data)) / eeg.sample_frequency
    time_steps_for_fmri = time_steps_for_eeg[::r_fmri]
    fmri_hdr_lookup = {}
    pregen_delta_range = delta_range if PLOT_DELTA in delta_range else np.concatenate([delta_range, np.array([PLOT_DELTA])])
    pregen_tau_range = tau_range if PLOT_TAU in tau_range else np.concatenate([tau_range, np.array([PLOT_TAU])])
    pregen_alpha_range = alpha_range if PLOT_ALPHA in alpha_range else np.concatenate([alpha_range, np.array([PLOT_ALPHA])])
    tstart = time.time()
    for i, delta in enumerate(pregen_delta_range):
        tend = time.time()
        if verbose and i > 0:
            print(f'Delta: {delta:.5f} ({i / len(delta_range) * 100:.2f}%). '
                  f'Last tau/alpha search took {tend - tstart:.2f} seconds')
        elif verbose:
            print(f'Pregenerating eeg responses.\nDelta: {delta:.4f} ({i / len(delta_range) * 100:.2f}%).')
        for tau in pregen_tau_range:
            for alpha in pregen_alpha_range:
                hrf = get_est_hemodynamic_response(time_steps, delta, tau, alpha)
                hdr_for_eeg = get_hdr_for_eeg(eeg.data, hrf)
                fmri_hdr_for_eeg = downsample_hdr_for_eeg(r_fmri, hdr_for_eeg)
                fmri_hdr_lookup[(delta, tau, alpha)] = fmri_hdr_for_eeg

    x_start = 25
    x_length = 10
    try:
        while np.isnan(eeg.data[x_start]):
            x_start += x_length
    except IndexError:
        raise ValueError('Cannot find a section without NANs in eeg data')

    # Zoom in since change is very small
    return [
        plot_eeg_hdrs_across_range(
            'alpha', sparse_alpha_range,
            'tau', sparse_tau_range,
            'delta', delta_range,
            fmri_hdr_lookup,
            time_steps_for_fmri,
            x_start,
            x_length
        ), plot_eeg_hdrs_across_range(
            'delta', sparse_delta_range,
            'alpha', sparse_alpha_range,
            'tau', tau_range,
            fmri_hdr_lookup,
            time_steps_for_fmri,
            x_start,
            x_length
        ), plot_eeg_hdrs_across_range(
            'delta', sparse_delta_range,
            'tau', sparse_tau_range,
            'alpha', alpha_range,
            fmri_hdr_lookup,
            time_steps_for_fmri,
            x_start,
            x_length
        )
    ]


def display_plot(plot_fn: Callable, *args, **kwargs) -> None:
    plot_fn(*args, **kwargs)
    plt.show()
    plt.close()


def save_plot(save_to: str, plot_fn: Callable, *args, **kwargs):
    figs = plot_fn(*args, **kwargs)
    with PdfPages(f'{save_to}.pdf') as pdf:
        for fig in figs:
            pdf.savefig(fig)
    plt.close()
