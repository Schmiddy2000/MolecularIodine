# Imports
from turtledemo.penrose import start

import numpy as np
from copy import copy
from typing import List, Tuple, Optional, Union


def channel_to_wave_length(channel_array: np.array) -> np.array:
    I = 395.164
    C1 = 0.189753
    C2 = -1.118880e-05
    C3 = -1.240474e-09

    return I + C1 * channel_array + C2 * channel_array ** 2 + C3 * channel_array ** 3


def averager(counts_array: np.array):
    averaged_counts_array = copy(counts_array)
    last_true_value = counts_array[0]

    for i in range(len(counts_array) - 2):
        new_last_true_value = averaged_counts_array[i + 1]
        averaged_counts_array[i + 1] = np.mean([last_true_value, averaged_counts_array[i + 1],
                                                averaged_counts_array[i + 2]])
        last_true_value = new_last_true_value

    return averaged_counts_array


def slice_measurement_data(x_data: np.array,
                           y_data: np.array,
                           start: Optional[float],
                           stop: Optional[float]
                           ) -> Tuple[np.array, np.array]:
    """
    Brute force method that returns array sections of x and y data. Uses the values for
    start and stop to determine the upper and lower bound for the slicing by looping over x_data.
    """
    start_index = 0
    stop_index = len(x_data) - 1

    while x_data[start_index] < start:
        start_index += 1

    while x_data[stop_index] > stop:
        stop_index -= 1

    return x_data[start_index:stop_index], y_data[start_index:stop_index]


def get_slice_indices(data: Union[np.array, List],
                      min_value: Union[int, float],
                      max_value: Union[int, float]) -> Tuple[int, int]:

    start_index = 0
    stop_index = len(data) - 1

    while data[start_index] < min_value:
        start_index += 1

    while data[stop_index] > max_value:
        stop_index -= 1

    return start_index, stop_index
