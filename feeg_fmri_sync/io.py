import math
import os
import re
from collections import defaultdict
from typing import Optional, Tuple, List, Dict

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import surfa as sf

from feeg_fmri_sync.constants import Indexer


def load_from_nii(
        nii_file: str,
        batch_number: Optional[int] = None,
        number_of_tasks: Optional[int] = None) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Load the fMRI data from the given nii file
    """
    fmri_data = sf.load_volume(nii_file)
    if batch_number:
        if not number_of_tasks:
            raise ValueError("Must specify number of tasks if batch_number is specified")
        n_voxels = fmri_data.shape[0]
        start = math.ceil(n_voxels / number_of_tasks) * batch_number
        end = min([math.ceil(n_voxels / number_of_tasks) * (batch_number + 1), n_voxels])
        return fmri_data[start:end, :, :, :].data.squeeze(), np.arange(start, end)
    return fmri_data[:, :, :, :].data.squeeze(), np.arange(fmri_data.shape[0])


def load_roi_from_mat(mat_file: str, subj_and_run_i: int) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Load the ROI fMRI data from the given mat file and the index of the subject and run
    """
    mat_data = scipy.io.loadmat(mat_file)
    if 'X' not in mat_data:
        raise ValueError(f'Expect variable "X" to be defined in {mat_file}')
    if 'subIndx' not in mat_data:
        raise ValueError(f'Expect variable "subIndx" to be defined in {mat_file}')
    fmri_data = mat_data['X']
    ind = mat_data['subIndx'].squeeze()
    return fmri_data[ind == subj_and_run_i, :].T, np.arange(1, fmri_data.shape[1]+1)


def update_data(data: npt.NDArray, start_index: int, end_index: int, curr_indexer: Indexer,
                future_indexers: List[Indexer], variable_arrays: List[npt.NDArray], n_recursions: int) -> None:
    """
    Recursively update the data matrix with the given indexers and variable arrays
    """

    # End of recursion!
    if len(future_indexers) == 1:
        data[start_index:end_index, future_indexers[0].index] = future_indexers[0].values
        for i, array in enumerate(variable_arrays):
            data[start_index:end_index, future_indexers[0].index + 1 + i] = array
        return

    # We're creating a 2-D data matrix from an N-dimensional array
    # For each indexer, we will add N rows, where N is the size of the product of the sizes of the remaining indexers
    # The 2-D data matrix will have a column for each indexer (this is indexed by curr_indexer.index)
    size_of_curr_indexer_chunk = np.prod([len(indexer.values) for indexer in future_indexers])
    for i, value in enumerate(curr_indexer.values):
        # First, go to the starting index of the previous indexers, then skip N rows i times
        new_start_index = start_index + i * size_of_curr_indexer_chunk
        # We'll end after another chunk of data
        new_end_index = start_index + (i + 1) * size_of_curr_indexer_chunk
        # Set value data for the appropriate rows in the column associated with the current indexer.
        #   errors occur if the value is not a float, so we use the index instead of the strings
        try:
            data[new_start_index:new_end_index, curr_indexer.index] = value
        except ValueError:
            data[new_start_index:new_end_index, curr_indexer.index] = i
        # Reduce the variable arrays along the current indexer
        new_variable_to_array = [array.take(i, axis=(curr_indexer.index - n_recursions)) for array in variable_arrays]
        # Update the data for the remaining indexers
        update_data(data, new_start_index, new_end_index, future_indexers[0],
                    future_indexers[1:], new_variable_to_array, n_recursions + 1)


def load_df_from_mat_and_csv(mat_files: List[str], csv_files: List[str], input_data_name: str):
    """
    Generate a pandas dataframe from the output of run_search_on_gamma_model.py
    """
    # Load csv files - these are the labels for the rows of the data
    models_to_indexer: Dict[str, Dict[str, Indexer]] = defaultdict(dict)
    models_to_variables: Dict[str, Dict[str, npt.NDArray]] = defaultdict(dict)

    csv_regex = re.compile(rf'(?P<i>\d+)_key_(?P<indexer_name>.+)_for_{input_data_name}_(?P<out_name>.+).csv')
    for csv_file in csv_files:
        match = csv_regex.match(os.path.basename(csv_file))
        if not match:
            raise ValueError(f'Could not parse {csv_file}')
        columns_to_possible_values = models_to_indexer[match.group('out_name')]
        indexer_name = match.group('indexer_name')
        try:
            values = np.loadtxt(csv_file, delimiter=',')
        except ValueError:
            values = np.loadtxt(csv_file, delimiter=',', dtype=str)
        values = np.atleast_1d(values)
        if len(values.shape) != 1:
            raise ValueError(f'Expected {csv_file} to have a single column')
        columns_to_possible_values[indexer_name] = Indexer(values=values, index=int(match.group('i')))

    # Load mat files - these are the data
    mat_regex = re.compile(rf'(?P<variable_name>.+)_search_on_{input_data_name}_(?P<out_name>.+).mat')
    for mat_file in mat_files:
        match = mat_regex.match(os.path.basename(mat_file))
        if not match:
            raise ValueError(f'Could not parse {mat_file}')
        columns_to_possible_values = models_to_variables[match.group('out_name')]
        indexer_name = match.group('variable_name')
        columns_to_possible_values[indexer_name] = scipy.io.loadmat(mat_file)[indexer_name]

    for meta_model_name, columns_to_possible_values in models_to_indexer.items():
        # Check that each meta-model in the csv files has a corresponding meta-model in the mat files
        if meta_model_name not in models_to_variables:
            raise ValueError(f'Could not find meta-model {meta_model_name} in mat files')
        # Check that the mat file data has the correct shape according to the indexers in the csv files
        expected_shape = np.empty(len(columns_to_possible_values), dtype=int)
        for indexer_data in columns_to_possible_values.values():
            expected_shape[indexer_data.index] = indexer_data.values.size
        for variable_name in models_to_variables[meta_model_name]:
            if not np.all(np.equal(models_to_variables[meta_model_name][variable_name].shape, expected_shape)):
                raise ValueError(f'Expected shape {expected_shape} for {meta_model_name}:{variable_name}, '
                                 f'but got {models_to_variables[meta_model_name][variable_name].shape}')

    frames = []
    for meta_model_name, columns_to_possible_values in models_to_indexer.items():
        sorted_columns = sorted(columns_to_possible_values.keys(), key=lambda x: columns_to_possible_values[x].index)
        indexers = sorted([indexer_data for indexer_data in columns_to_possible_values.values()], key=lambda x: x.index)
        variable_names = sorted(models_to_variables[meta_model_name].keys())
        size_0 = np.prod([indexer_data.values.size for indexer_data in columns_to_possible_values.values()])
        data = np.empty((size_0, len(models_to_variables[meta_model_name]) + len(columns_to_possible_values)))

        update_data(data, 0, 0, indexers[0], indexers[1:],
                    [models_to_variables[meta_model_name][variable_name] for variable_name in variable_names], 0)
        df = pd.DataFrame(data, columns=sorted_columns + variable_names)
        # Now that we're operating in DataFrames, we can convert the indexers to their values
        # (dataframe supports strings)
        for column in sorted_columns:
            indexer_data = columns_to_possible_values[column]
            try:
                indexer_data.values.astype(float)
            except ValueError:
                for i, value in enumerate(indexer_data.values):
                    df.loc[df[column] == i, column] = value
        frames.append(df)
    df = pd.concat(frames)
    return df.apply(pd.to_numeric, errors='ignore')

