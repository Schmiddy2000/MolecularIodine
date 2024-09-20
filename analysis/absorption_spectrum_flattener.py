# Imports
import numpy as np
from matplotlib import pyplot as plt

from tools.reader import extract_data
from tools.transformers import channel_to_wave_length

# Questions:
# Would it make sense to fit another curve to the already straightened curve with dips? This could
# then proper linearize the curve, but might introduce issues with dip and valley interpretability...
# Does this generally not really work when the fit would remain around y = 0?


def linearize_count_dips(use_background: bool):
    channels, counts = extract_data('../data/channel_data/halo_7ms_250avg_absorption_50grad.txt')
    background_channels, background_counts = extract_data('../data/channel_data/halo_8ms_250avg_noabsorption.txt')
    wave_lengths = channel_to_wave_length(channels)

    start_index = 550
    stop_index = len(wave_lengths) - 450

    wave_lengths = wave_lengths[start_index:stop_index]
    counts = counts[start_index:stop_index]
    background_counts = background_counts[start_index:stop_index]

    if use_background:
        counts = counts - background_counts

    degree = 11  # Quadratic polynomial

    coefficients = np.polyfit(wave_lengths, counts, degree)

    polynomial = np.poly1d(coefficients)

    y_fit = polynomial(wave_lengths)

    degree_2 = 3

    coefficients_2 = np.polyfit(wave_lengths, counts - y_fit, degree_2)

    polynomial_2 = np.poly1d(coefficients_2)

    y_fit_2 = polynomial_2(wave_lengths)

    plt.figure(figsize=(10, 6))
    plt.scatter(wave_lengths, counts, label='Data', color='blue', s=15)
    plt.plot(wave_lengths, y_fit, label=f'Polynomial Fit (degree {degree})', color='red', linewidth=2)
    plt.plot(wave_lengths, counts - y_fit, label='first adjustment')
    plt.plot(wave_lengths, y_fit_2, label='second fit')
    # plt.plot(wave_lengths, y_fit - y_fit_2, label='second adjustment')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynomial Fit using np.polyfit')
    plt.legend()
    plt.show()


linearize_count_dips(True)
