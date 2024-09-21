# Imports
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from tools.reader import extract_data
from tools.transformers import channel_to_wave_length, get_slice_indices, averager
from typing import Tuple, Optional

# Questions:
# Would it make sense to fit another curve to the already straightened curve with dips? This could
# then proper linearize the curve, but might introduce issues with dip and valley interpretability...
# Does this generally not really work when the fit would remain around y = 0?


def linearize_count_dips_halogen_lamp(degree: int, plot: bool = True) -> Optional[Tuple[np.array, np.array]]:
    channels, counts = extract_data('../data/channel_data/halo_7ms_250avg_absorption_50grad.txt')
    background_channels, background_counts = extract_data('../data/channel_data/halo_8ms_250avg_noabsorption.txt')
    wave_lengths = channel_to_wave_length(channels)

    start_index, stop_index = get_slice_indices(wave_lengths, 500, 650)

    wave_lengths = wave_lengths[start_index:stop_index]
    counts = counts[start_index:stop_index]
    background_counts = background_counts[start_index:stop_index]

    difference_counts = counts - background_counts

    # degree = 51  # Quadratic polynomial
    coefficients = np.polyfit(wave_lengths, difference_counts, degree)
    polynomial = np.poly1d(coefficients)
    y_fit = polynomial(wave_lengths)

    # Normalize the count values
    max_counts = np.max(counts)
    counts = counts / max_counts
    background_counts = background_counts / max_counts
    difference_counts = difference_counts / max_counts
    y_fit = y_fit / max_counts

    if plot:
        plt.figure(figsize=(12, 5))
        plt.title('Isolation of the absorption spectrum for the halogen lamp', fontsize=16)
        plt.xlabel('Wave length in [nm]', fontsize=13)
        plt.ylabel('Normalized intensity', fontsize=13)

        plt.plot(wave_lengths, counts, label='Absorption spectrum')
        plt.plot(wave_lengths, background_counts, label='Lamp spectrum')
        plt.plot(wave_lengths, difference_counts, label='Spectrum difference', color='blue')
        plt.plot(wave_lengths, y_fit, label=f'Polynomial Fit (degree {degree})', color='red', lw=1.25, ls='--')
        plt.plot(wave_lengths, difference_counts - y_fit, label='Isolated absorption spectrum')

        plt.legend()
        plt.tight_layout()
        plt.savefig('Isolation_absorption_spectrum_halogen_lamp.png', dpi=200)
        plt.show()
    else:
        return wave_lengths,  (difference_counts - y_fit) * max_counts


# linearize_count_dips_halogen_lamp(23, True)


def linearize_count_dips_led_lamp(degree: int, plot: bool = True) -> Optional[Tuple[np.array, np.array]]:
    channels, counts = extract_data('../data/channel_data/LED_5ms_500avg_absorption_50grad.txt')
    background_channels, background_counts = extract_data('../data/channel_data/LED_5ms_100avg_lamp.txt')
    wave_lengths = channel_to_wave_length(channels)

    start_index, stop_index = get_slice_indices(wave_lengths, 500, 650)

    wave_lengths = wave_lengths[start_index:stop_index]
    counts = counts[start_index:stop_index]
    background_counts = background_counts[start_index:stop_index]

    # Adjust intensity
    max_absorption_counts = np.max(counts)
    max_background_counts = np.max(background_counts)

    difference_counts = counts - background_counts # (max_absorption_counts / max_background_counts) * background_counts

    # degree = 23  # Quadratic polynomial
    coefficients = np.polyfit(wave_lengths, difference_counts, degree)
    polynomial = np.poly1d(coefficients)
    y_fit = polynomial(wave_lengths)

    # Normalize the count values
    max_counts = np.max(counts)
    counts = counts / max_counts
    background_counts = background_counts / max_counts
    difference_counts = difference_counts / max_counts
    y_fit = y_fit / max_counts

    if plot:
        plt.figure(figsize=(12, 5))
        plt.title('Isolation of the absorption spectrum for the LED lamp', fontsize=16)
        plt.xlabel('Wave length in [nm]', fontsize=13)
        plt.ylabel('Normalized intensity', fontsize=13)

        plt.plot(wave_lengths, counts, label='Absorption spectrum')
        plt.plot(wave_lengths, background_counts, label='Lamp spectrum')
        plt.plot(wave_lengths, difference_counts, label='Spectrum difference', color='blue')
        plt.plot(wave_lengths, y_fit, label=f'Polynomial Fit (degree {degree})', color='red', lw=1.25, ls='--')
        plt.plot(wave_lengths, difference_counts - y_fit, label='Isolated absorption spectrum')

        plt.legend()
        plt.tight_layout()
        plt.savefig('Isolation_absorption_spectrum_led_lamp.png', dpi=200)
        plt.show()
    else:
        return wave_lengths, (difference_counts - y_fit) * max_counts


# linearize_count_dips_led_lamp(23, True)


def linearize_count_dips_both_lamps(degree: int, plot: bool = True) -> Optional[Tuple[np.array, np.array]]:
    channels_led, counts_led = extract_data('../data/channel_data/LED_5ms_500avg_absorption_50grad.txt')
    _, background_counts_led = extract_data('../data/channel_data/LED_5ms_100avg_lamp.txt')
    _, counts_halogen = extract_data('../data/channel_data/halo_7ms_250avg_absorption_50grad.txt')
    _, background_counts_halogen = extract_data('../data/channel_data/halo_8ms_250avg_noabsorption.txt')
    
    wave_lengths = channel_to_wave_length(channels_led)

    start_index, stop_index = get_slice_indices(wave_lengths, 500, 650)

    counts = counts_led + counts_halogen
    background_counts = background_counts_led + background_counts_halogen

    wave_lengths = wave_lengths[start_index:stop_index]
    counts = counts[start_index:stop_index]
    background_counts = background_counts[start_index:stop_index]

    # Adjust intensity
    max_absorption_counts = np.max(counts)
    max_background_counts = np.max(background_counts)

    difference_counts = counts - background_counts # (max_absorption_counts / max_background_counts) * background_counts

    # degree = 23  # Quadratic polynomial
    coefficients = np.polyfit(wave_lengths, difference_counts, degree)
    polynomial = np.poly1d(coefficients)
    y_fit = polynomial(wave_lengths)

    # Normalize the count values
    max_counts = np.max(counts)
    counts = counts / max_counts
    background_counts = background_counts / max_counts
    difference_counts = difference_counts / max_counts
    y_fit = y_fit / max_counts

    if plot:
        plt.figure(figsize=(12, 5))
        plt.title('Isolation of the absorption spectrum for the LED lamp', fontsize=16)
        plt.xlabel('Wave length in [nm]', fontsize=13)
        plt.ylabel('Normalized intensity', fontsize=13)

        plt.plot(wave_lengths, counts, label='Absorption spectrum')
        plt.plot(wave_lengths, background_counts, label='Lamp spectrum')
        plt.plot(wave_lengths, difference_counts, label='Spectrum difference', color='blue')
        plt.plot(wave_lengths, y_fit, label=f'Polynomial Fit (degree {degree})', color='red', lw=1.25, ls='--')
        plt.plot(wave_lengths, difference_counts - y_fit, label='Isolated absorption spectrum')

        plt.legend()
        plt.tight_layout()
        plt.savefig('Isolation_absorption_spectrum_led_lamp.png', dpi=200)
        plt.show()
    else:
        return wave_lengths, (difference_counts - y_fit) * max_counts


# linearize_count_dips_both_lamps(23, True)


def combined_spectrum_plot():
    wave_length_array, halogen_isolation = linearize_count_dips_halogen_lamp(23, False)
    _, led_isolation = linearize_count_dips_led_lamp(23, False)
    _, combined_isolation = linearize_count_dips_both_lamps(23, False)

    normalization_factor = np.max(combined_isolation) - np.min(combined_isolation)

    plt.figure(figsize=(12, 3))
    plt.title('Comparison of the isolated absorption spectra', fontsize=16)
    plt.xlabel('Wave length in [nm]', fontsize=13)
    plt.ylabel('Normalized intensity', fontsize=13)

    plt.plot(wave_length_array, (led_isolation + abs(min(combined_isolation))) / normalization_factor,
             label='LED lamp')
    plt.plot(wave_length_array, (halogen_isolation + abs(min(combined_isolation))) / normalization_factor,
             label='Halogen lamp')
    plt.plot(wave_length_array, (combined_isolation + abs(min(combined_isolation))) / normalization_factor,
             label='Combined', c='black', alpha=0.3)

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('Isolated_absorption_spectrum_comparison.png', dpi=200)
    plt.show()

    return None


# combined_spectrum_plot()


def dip_analysis():
    # Lists of peaks and dips starting at 520 nm and using combined spectrum (gray).
    # Additional thoughts:
    # - We seem to have two series of peaks with slightly different spacing in between. The ones starting to appear
    # around 445 nm will be right peak, the others only peak since this is the initially visible series.
    # - dip at 574.4 is questionable but fits the pattern
    # Uncertainties are:
    # around ± 0.1 from 520 nm to xxx nm
    peak_positions = 500 + np.array([20.56, 22.08])
    dip_positions = 500 + np.array([21.08, 22.61, 24.13, 25.75, 27.41, 29.19, 31.09, 32.94, 34.85, 36.98, 39.1, 41.28,
                                    43.52, 45.92, 48.38, 50.88, 53.49, 56.23, 59.03, 61.94, 64.94, 67.96, 71.11, 74.4])
    right_dip_positions = 500 + np.array([45.28, 47.63, 50.05, 52.41, 54.92, 57.45, 60.1, 62.96, 65.74, 68.73, 71.78,
                                          74.93])

    dip_positions = dip_positions[12:]

    dip_arange = np.arange(len(dip_positions) - 1)
    right_dip_arange = np.arange(len(dip_positions) - len(right_dip_positions), len(dip_positions) - 1)

    dip_differences = np.diff(dip_positions, prepend=dip_positions[0])[1:]
    right_dip_differences = np.diff(right_dip_positions, prepend=dip_positions[0])[1:]
    x_lin = np.linspace(-10, 50, 5)

    print(len(dip_positions))
    print(len(right_dip_positions))
    print(len(dip_positions) - len(right_dip_positions) - 1)
    print(dip_arange)
    print(right_dip_arange)

    # Perform linear regression
    dip_arange = np.arange(len(dip_positions) - 1)
    right_dip_arange = np.arange(len(dip_positions) - len(right_dip_positions) + 1, len(dip_positions))
    slope_dip, intercept_dip, r_value_dip, p_value_dip, std_err_dip = stats.linregress(dip_arange, dip_differences)
    slope_right_dip, intercept_right_dip, r_value_right_dip, p_value_right_dip, std_err_righ_dip = stats.linregress(
        right_dip_arange, right_dip_differences)

    # Regression line
    def lin_func(x, a, b):
        return a * x + b

    y_pred_dip = slope_dip * dip_arange + intercept_dip

    # Calculate residuals and standard deviation of residuals (sigma)
    residuals_dip = dip_differences - y_pred_dip
    sigma_dip = np.std(residuals_dip)

    y_pred_right_dip = slope_right_dip * right_dip_arange + intercept_right_dip

    # Calculate residuals and standard deviation of residuals (sigma)
    residuals_right_dip = right_dip_differences - y_pred_right_dip
    sigma_right_dip = np.std(residuals_right_dip)

    print(slope_dip, slope_right_dip)

    plt.plot(x_lin, lin_func(x_lin, slope_dip, intercept_dip))
    plt.plot(x_lin, lin_func(x_lin, slope_dip, intercept_dip) + sigma_dip, c='b', ls='--')
    plt.plot(x_lin, lin_func(x_lin, slope_dip, intercept_dip) - sigma_dip, c='b', ls='--')

    plt.plot(x_lin, lin_func(x_lin, slope_right_dip, intercept_right_dip))
    plt.plot(x_lin, lin_func(x_lin, slope_right_dip, intercept_right_dip) + sigma_right_dip, c='b', ls='--')
    plt.plot(x_lin, lin_func(x_lin, slope_right_dip, intercept_right_dip) - sigma_right_dip, c='b', ls='--')
    
    plt.scatter(dip_arange, dip_differences)
    plt.scatter(right_dip_arange, right_dip_differences)

    plt.show()

    return None


# dip_analysis()


def automated_dip_analysis():
    wave_lengths, initial_counts = linearize_count_dips_both_lamps(23, False)

    start_index, stop_index = get_slice_indices(wave_lengths, 510, 630)

    counts = averager(initial_counts)

    wave_lengths = wave_lengths[start_index:stop_index]
    counts = counts[start_index:stop_index]

    dips = []
    dip_counts = []
    peaks = []
    peak_counts = []

    for i in range(1, len(wave_lengths) - 2):
        if counts[i + 1] > counts[i] and counts[i - 1] > counts[i]:
            dips.append(wave_lengths[i])
            dip_counts.append(counts[i])

        if counts[i + 1] < counts[i] and counts[i - 1] < counts[i]:
            peaks.append(wave_lengths[i])
            peak_counts.append(counts[i])

    dip_differences = np.diff(dips, prepend=dips[0])[1:]
    peak_differences = np.diff(peaks, prepend=dips[0])[1:]

    dip_arange = np.arange(len(dip_differences))
    peak_arange = np.arange(len(peak_differences))

    delete_indices = [i for i in range(23, 34)] + [38, 39] + [50, 51, 52, 53]

    dip_differences = np.delete(dip_differences, delete_indices)
    peak_differences = np.delete(peak_differences, delete_indices)
    dip_arange = np.delete(dip_arange, delete_indices)
    peak_arange = np.delete(peak_arange, delete_indices)

    dip_slope, dip_intercept, dip_r_value, dip_p_value, dip_std_err = stats.linregress(dip_arange, dip_differences)
    peak_slope, peak_intercept, peak_r_value, peak_p_value, peak_std_err = stats.linregress(peak_arange,
                                                                                            peak_differences)

    # Regression line
    def lin_func(x, a, b):
        return a * x + b

    x_lin = np.linspace(-10, 60, 5)

    plt.figure()

    # plt.plot(wave_lengths, counts, c='k', label='average')
    # # plt.plot(wave_lengths, counts, c='y', label='no average', ls='--')
    # plt.scatter(dips, dip_counts, label='dips')
    # plt.scatter(peaks, peak_counts, label='peaks')

    plt.plot(x_lin, lin_func(x_lin, dip_slope, dip_intercept))
    plt.plot(x_lin, lin_func(x_lin, peak_slope, peak_intercept))

    plt.scatter(dip_arange, dip_differences)
    plt.scatter(peak_arange, peak_differences)

    plt.legend()
    plt.show()


automated_dip_analysis()
