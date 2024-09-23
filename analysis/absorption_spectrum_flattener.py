# Imports
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.constants import c, h
from scipy.odr import ODR, Model, RealData

from tools.reader import extract_data
from tools.transformers import channel_to_wave_length, get_slice_indices, averager
from calibration import calibrate_pixel
from typing import Tuple, Optional
from copy import copy


# Questions:
# Would it make sense to fit another curve to the already straightened curve with dips? This could
# then proper linearize the curve, but might introduce issues with dip and valley interpretability...
# Does this generally not really work when the fit would remain around y = 0?


def linearize_count_dips_halogen_lamp(degree: int, plot: bool = True) -> Optional[Tuple[np.array, np.array]]:
    channels, counts = extract_data('../data/channel_data/halo_7ms_250avg_absorption_50grad.txt')
    background_channels, background_counts = extract_data('../data/channel_data/halo_8ms_250avg_noabsorption.txt')
    # wave_lengths, wave_length_uncertainties = calibrate_pixel(channels)
    wave_lengths = [calibrate_pixel(channel)[0] for channel in channels]
    wave_length_uncertainties = [calibrate_pixel(channel)[1] for channel in channels]

    start_index, stop_index = get_slice_indices(wave_lengths, 500, 650)

    print(start_index, stop_index)

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
        return wave_lengths, (difference_counts - y_fit) * max_counts


# linearize_count_dips_halogen_lamp(23, True)


def linearize_count_dips_led_lamp(degree: int, plot: bool = True) -> Optional[Tuple[np.array, np.array]]:
    channels, counts = extract_data('../data/channel_data/LED_5ms_500avg_absorption_50grad.txt')
    background_channels, background_counts = extract_data('../data/channel_data/LED_5ms_100avg_lamp.txt')

    return_channels = channels - background_channels

    # wave_lengths, wave_length_uncertainties = calibrate_pixel(channels)
    wave_lengths = [calibrate_pixel(channel)[0] for channel in channels]
    wave_length_uncertainties = [calibrate_pixel(channel)[1] for channel in channels]

    start_index, stop_index = get_slice_indices(wave_lengths, 500, 650)

    wave_lengths = wave_lengths[start_index:stop_index]
    counts = counts[start_index:stop_index]
    background_counts = background_counts[start_index:stop_index]

    # Adjust intensity
    max_absorption_counts = np.max(counts)
    max_background_counts = np.max(background_counts)

    difference_counts = counts - background_counts
    # (max_absorption_counts / max_background_counts) * background_counts

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
        return return_channels, (difference_counts - y_fit) * max_counts


# linearize_count_dips_led_lamp(23, True)


def linearize_count_dips_both_lamps(degree: int, plot: bool = True) -> Optional[Tuple[np.array, np.array]]:
    channels_led, counts_led = extract_data('../data/channel_data/LED_5ms_500avg_absorption_50grad.txt')
    _, background_counts_led = extract_data('../data/channel_data/LED_5ms_100avg_lamp.txt')
    _, counts_halogen = extract_data('../data/channel_data/halo_7ms_250avg_absorption_50grad.txt')
    _, background_counts_halogen = extract_data('../data/channel_data/halo_8ms_250avg_noabsorption.txt')

    return_channels = copy(channels_led)

    # wave_lengths, wave_length_uncertainties = calibrate_pixel(channels_led)
    wave_lengths = [calibrate_pixel(channel)[0] for channel in channels_led]
    wave_length_uncertainties = [calibrate_pixel(channel)[1] for channel in channels_led]

    start_index, stop_index = get_slice_indices(wave_lengths, 500, 650)

    counts = counts_led + counts_halogen
    background_counts = background_counts_led + background_counts_halogen

    wave_lengths = wave_lengths[start_index:stop_index]
    counts = counts[start_index:stop_index]
    background_counts = background_counts[start_index:stop_index]

    # Adjust intensity
    max_absorption_counts = np.max(counts)
    max_background_counts = np.max(background_counts)

    difference_counts = counts - background_counts
    # (max_absorption_counts / max_background_counts) * background_counts

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
        print('dims:')
        print(len(return_channels))
        print(len((difference_counts - y_fit) * max_counts))
        return return_channels[start_index:stop_index], (difference_counts - y_fit) * max_counts


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


def automated_dip_analysis(channels, counts):  # , start, stop):
    # wave_lengths, initial_counts = linearize_count_dips_both_lamps(23, False)

    # start_index, stop_index = get_slice_indices(wave_lengths, 510, 630)

    counts = averager(counts)

    # channels = channels[start:stop]
    # counts = counts[start:stop]

    dips = []
    dip_counts = []
    peaks = []
    peak_counts = []

    for i in range(1, len(channels) - 2):
        if counts[i + 1] > counts[i] and counts[i - 1] > counts[i]:
            dips.append(channels[i])
            dip_counts.append(counts[i])

        if counts[i + 1] < counts[i] and counts[i - 1] < counts[i]:
            peaks.append(channels[i])
            peak_counts.append(counts[i])

    return channels, counts, dips, dip_counts, peaks, peak_counts


def dip_sequence_analysis():
    my_channels, my_counts = linearize_count_dips_both_lamps(23, False)
    my_wave_lengths = [calibrate_pixel(channel)[0] for channel in my_channels]

    start_index, stop_index = get_slice_indices(my_wave_lengths, 508, 635)
    my_channels = my_channels[start_index:stop_index]
    my_counts = my_counts[start_index:stop_index]

    # print(my_channels, my_counts)
    wave_lengths, counts, dips, dip_counts, peaks, peak_counts = automated_dip_analysis(my_channels, my_counts)
    wave_lengths = [calibrate_pixel(channel)[0] for channel in my_channels]

    print(min(wave_lengths), max(wave_lengths))

    # start_index, stop_index = get_slice_indices(my_wave_lengths, 509, 620)

    # wave_lengths = [calibrate_pixel(channel)[0] for channel in channels]
    # wave_length_uncertainties = [calibrate_pixel(channel)[1] for channel in channels]

    print(dips, calibrate_pixel(55))
    print(min(wave_lengths), max(wave_lengths))

    dips = [calibrate_pixel(dip)[0] for dip in dips]
    dip_position_uncertainties = [calibrate_pixel(dip)[1] for dip in dips]
    peaks = [calibrate_pixel(peak)[0] for peak in peaks]

    print(dips)

    # Convert dip differences to energy differences
    dips = 1e-9 * np.array(dips[::-1])
    # dips = 1e-9 * np.array(dips)
    dip_energies = 6.242e18 * h * c / dips
    dip_frequencies = 6.242e18 * h * c / dips  # * 1e-13

    # Compute energy differences
    # dip_differences = np.diff(dip_energies, prepend=dips[0])[1:]
    dip_differences = np.diff(dip_frequencies, prepend=dips[0])[1:]
    dip_differences = np.array([abs(val) for val in dip_differences])
    # peak_differences = np.diff(peaks, prepend=dips[0])[1:]

    dip_uncertainties = [np.sqrt(1e-20 + val ** 2) for val in dip_position_uncertainties]
    dip_frequency_uncertainties = [np.sqrt(dip_uncertainties[i] ** 2 + dip_uncertainties[i + 1]) for i in
                                   range(len(dip_uncertainties) - 2)] + [1e-18]

    # print('Dip positions in nm\n', dips)
    # print('Dip position differences in nm\n', dip_differences)

    # Convert dip differences to energy differences
    # dip_differences = 1e-9 * dip_differences
    # dip_differences = 6.242e18 * h * c / dip_differences

    # Generate array of vibrational state indices
    offset = 2
    dip_arange = np.arange(len(dip_differences)) + offset
    # peak_arange = np.arange(len(peak_differences))

    # Indices of problematic dip positions and removing them from all dependent arrays
    # delete_indices = [i for i in range(23, 34)] + [38, 39] + [50, 51, 52, 53]
    delete_indices = np.array([2, 3, 4] + [16] + [i for i in range(22, 33)])

    bad_dips = list(dip_differences[2:5]) + [dip_differences[16]] + list(dip_differences[22:33])

    dip_differences = np.delete(dip_differences, delete_indices)
    dip_arange = np.delete(dip_arange, delete_indices)
    dip_frequency_uncertainties = np.delete(dip_frequency_uncertainties, delete_indices)

    print(dip_differences[20:30])
    print(dip_frequency_uncertainties[20:30])

    # dip_differences = dip_differences[:22]
    # dip_arange = dip_arange[:22]

    # peak_differences = np.delete(peak_differences, delete_indices)
    # peak_arange = np.delete(peak_arange, delete_indices)

    # dip_slope, dip_intercept, dip_r_value, dip_p_value, dip_std_err = stats.linregress(dip_arange, dip_differences)
    # peak_slope, peak_intercept, peak_r_value, peak_p_value, peak_std_err = stats.linregress(peak_arange,
    #                                                                                         peak_differences)

    # Regression line
    def lin_func(x, a, b):
        return a * x + b

    def energy_difference(v, w_0, x_0):
        return w_0 - 2 * w_0 * x_0 * (v + 1)

    # Define the model function
    def energy_difference_odr(w_0_x, v):
        w_0, x_0 = w_0_x  # Unpack parameters (w_0, x_0)
        return w_0 - 2 * w_0 * x_0 * (v + 1)

    # Create a model object for ODR
    model = Model(energy_difference_odr)

    # Prepare data (dip_arange, dip_differences)
    data = RealData(dip_arange, dip_differences)
    # sy=dip_frequency_uncertainties * 1e12)# ,
    # sx=np.ones(len(dip_differences)))

    # Set initial guess for parameters (w_0, x_0)
    initial_guess = [0, -0.02]

    # Set up ODR with the model and data
    odr_instance = ODR(data, model, beta0=initial_guess)

    # Run the ODR fitting
    output = odr_instance.run()

    # Extract fitted parameters and standard errors
    w_0_fitted, x_0_fitted = output.beta
    w_0_error, x_0_error = output.sd_beta

    print(output.cov_beta)

    # Print fitted parameters and their errors
    print(f"Fitted parameters: w_0 = {w_0_fitted}, x_0 = {x_0_fitted}")
    print(f"Errors: w_0_error = {w_0_error}, x_0_error = {x_0_error}")

    # Initial guess for the parameters (w_0, x_0)
    initial_guess = [0, -0.02]

    print(len(dip_differences))

    # Perform the curve fit
    params, covariance = curve_fit(energy_difference, dip_arange, dip_differences, p0=initial_guess, maxfev=10000)

    # # Extract the fitted parameters
    # w_0_fitted, x_0_fitted = params
    # w_0_error, x_0_error = np.sqrt(np.diag(covariance))

    # w_0_fitted *= 8065.5
    # w_0_error *= 8065.5
    # x_0_fitted *= 8065.5
    # x_0_error *= 8065.5

    print(f"Fitted parameters: w_0 = {round(w_0_fitted, 5)} ± {round(w_0_error, 5)},"
          f" x_0 = {round(x_0_fitted, 5)} ± {round(x_0_error, 5)}")

    x_lin = np.linspace(-10, 60, 5)

    plt.figure(figsize=(12, 5))
    plt.title('Birge-Sponer plot of the absorption spectrum', fontsize=16)
    plt.ylabel(r"$\Delta G_{\nu^{''}}$ in [cm$^{-1}$]", fontsize=13)
    plt.xlabel(r"Vibrational quantum number $\nu^{''}$ + 1", fontsize=13)

    # plt.title('Illustration dip counter-propagation', fontsize=16)
    # plt.ylabel(r"Counts", fontsize=13)
    # plt.xlabel(r"Wave length in [nm]", fontsize=13)

    # count_range = max(counts) - min(counts)
    # counts = counts / count_range
    # dips = dips / count_range
    # peaks = np.array(peaks) / count_range

    # min()

    plt.plot(wave_lengths, counts, c='k', label='average')
    # plt.plot(wave_lengths, counts, c='y', label='no average', ls='--')
    plt.scatter(np.array(1e9 * dips), dip_counts[::-1], label='dips')
    plt.scatter(np.array(peaks), peak_counts, label='peaks')

    # plt.fill_between(x_lin, 8065 * energy_difference(x_lin, w_0_fitted - w_0_error, x_0_fitted + x_0_error),
    #                  8065 * energy_difference(x_lin, w_0_fitted + w_0_error, x_0_fitted - x_0_error), ls='--', lw=0.5,
    #                  color='r', alpha=0.2, label=r'1-$\sigma$ confidence band')
    # plt.plot(x_lin, 8065 * energy_difference(x_lin, w_0_fitted, x_0_fitted), c='k', lw=1,
    #          label='Fit with' + '\n' + fr'$\omega_0$ = {8065.5 * w_0_fitted:.2f} ± {8065.5 * w_0_error:.2f}' +
    #                '\n' + fr'$x_0$ = {x_0_fitted:.4f} ± {x_0_error:.4f}')
    # plt.scatter(dip_arange, 8065 * dip_differences, label='Data points')
    # plt.scatter(delete_indices + offset, 8065 * np.array(bad_dips), label='Excluded dips')
    # plt.errorbar(dip_arange, 8065 * dip_differences, yerr=dip_frequency_uncertainties, label='Errors', fmt='none',
    #              ecolor='k')

    plt.tight_layout()
    plt.legend()
    plt.xlim(0, 60)
    # plt.xlim(540, 575)
    # plt.savefig('dip_counter_propagation.png', dpi=200)
    # plt.savefig('Birge_sponer_absorption.png', dpi=200)
    plt.show()


dip_sequence_analysis()


def emission():
    channels, counts = extract_data('../data/all_data/laser_254_1000ms_150avg_vertical_15.txt')
    _, counts_1 = extract_data('../data/all_data/laser_254_800ms_80avg_vertical_11.txt')
    _, counts_2 = extract_data('../data/all_data/laser_254_800ms_80avg_vertical_12.txt')
    _, counts_3 = extract_data('../data/all_data/laser_254_800ms_80avg_vertical_13.txt')
    _, counts_4 = extract_data('../data/all_data/laser_254_800ms_80avg_vertical_10.txt')
    _, counts_5 = extract_data('../data/all_data/laser_254_1000ms_150avg_vertical_14.txt')
    _, counts_6 = extract_data('../data/all_data/laser_254_1000ms_150avg_vertical_14.txt')

    # plt.plot(channels, counts)
    start_index, stop_index = 725, 2040
    channels = channels[start_index:stop_index]
    summed_counts = counts_1 + counts_2 + counts_3 + counts_4 + counts_5 + counts_6
    summed_counts = averager(summed_counts)
    summed_counts = summed_counts[start_index:stop_index]

    wave_lengths = [calibrate_pixel(s_c)[0] for s_c in channels]
    wave_length_uncertainties = [calibrate_pixel(s_c)[1] for s_c in channels]

    v_line_positions = np.array([759.55, 796.2, 832.5, 870, 908.8, 948.22, 988.26, 1030.05, 1072.2, 1116.5, 1206.9,
                                 1302.1, 1401.9, 1454.7])
    position_errors = np.array([0.5, 2, 1, 1, 0.75, 1, 0.5, 0.75, 2, 1, 1, 2, 2, 3]) * 0.15

    v_line_positions = np.array([calibrate_pixel(v_l_p)[0] for v_l_p in v_line_positions])
    v_line_positions_errors = np.array([calibrate_pixel(v_l_p)[1] for v_l_p in v_line_positions])
    position_errors = np.array([np.sqrt(position_errors[i] ** 2 + v_line_positions_errors[i] ** 2) for i in
                       range(len(v_line_positions))])

    print(len(v_line_positions))
    print(list(v_line_positions))
    print(len(position_errors))
    print(list(position_errors))

    my_frequencies = 6.242e18 * h * c / (1e-9 * np.array(v_line_positions))
    my_diff = np.diff(my_frequencies, prepend=my_frequencies[0])[1:]
    my_diff = 8065 * np.array([abs(m_d) for m_d in my_diff])

    # peak_arange = np.concatenate(np.arange(1, len(my_diff) + 1), np.array([]))
    peak_arange = np.array([i for i in range(2, 11)] + [11, 13, 15, 16])

    plt.figure(figsize=(12, 5))

    print(v_line_positions)
    print(my_frequencies)
    print(my_diff)

    my_diff = [d / 2 if d > 300 else d for d in my_diff]

    def energy_difference(v, w_0, x_0):
        return w_0 - 2 * w_0 * x_0 * (v + 1)

    x_lin = np.linspace(-1, 20)

    # Perform the curve fit
    params, covariance = curve_fit(energy_difference, peak_arange, my_diff, p0=[200, 0.1], maxfev=10000)

    # # Extract the fitted parameters
    w_0_fitted, x_0_fitted = params
    w_0_error, x_0_error = np.sqrt(np.diag(covariance))

    print('Cov\n')
    print(covariance)

    # delta_G_errors = [np.sqrt([i] ** 2 + dip_uncertainties[i + 1]) for i in
    #  range(len(dip_uncertainties) - 2)] + [1e-18]

    # Create the figure and two subplots (one above the other)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle('Emission spectrum and Brige-Sponer plot', fontsize=16)

    # First subplot (Fit with 1-sigma confidence band)
    ax1.fill_between(x_lin,
                     energy_difference(x_lin, w_0_fitted - w_0_error, x_0_fitted + x_0_error),
                     energy_difference(x_lin, w_0_fitted + w_0_error, x_0_fitted - x_0_error),
                     ls='--', lw=0.5, color='r', alpha=0.2, label=r'1-$\sigma$ confidence band')
    ax1.plot(x_lin, energy_difference(x_lin, w_0_fitted, x_0_fitted), c='k', lw=1,
             label='Fit with' + '\n' + fr'$\omega_0$ = {w_0_fitted:.2f} ± {w_0_error:.2f}' +
                   '\n' + fr'$x_0$ = {x_0_fitted:.4f} ± {x_0_error:.4f}')
    ax1.scatter(peak_arange, my_diff)
    ax1.set_xlim(0, 18)
    # ax1.errorbar(peak_arange, my_diff, yerr=dip_frequency_uncertainties, label='Errors', fmt='none',
    #                         ecolor='k')
    ax1.legend()

    # Second subplot (Summed counts and vertical lines with errors)
    ax2.plot(wave_lengths, summed_counts - 8150, label='Averaged data')
    for x, err in zip(v_line_positions, position_errors):
        if x == wave_lengths[-1]:
            ax2.axvline(x=x, color='r', linestyle='--', linewidth=0.5, label='Dip center')
            ax2.fill_betweenx(np.linspace(-1000, 60000, 2), x - err, x + err, color='red', alpha=0.2,
                              label=r"1-$\sigma$ confidence band")
        else:
            ax2.axvline(x=x, color='r', linestyle='--', linewidth=0.5)
            ax2.fill_betweenx(np.linspace(-1000, 60000, 2), x - err, x + err, color='red', alpha=0.2)
    ax2.set_ylim(-1000, 60000)

    # Add legends and labels where necessary
    ax2.legend()

    # Set labels, shared x-axis between subplots
    ax2.set_xlabel('Wavelength in [nm]', fontsize=12)
    ax1.set_ylabel(r"$\Delta G_{\nu^{''}}$ in [cm$^{-1}$]", fontsize=12)

    ax1.set_xlabel(r"Vibrational quantum number $\nu^{''}$ + 1", fontsize=12)
    ax2.set_ylabel('Summed Counts', fontsize=12)

    # Tight layout to adjust spacing
    plt.tight_layout()

    # Show the plot
    plt.savefig('emission_plot_with_brige.png', dpi=200)
    plt.show()

    # plt.fill_between(x_lin, energy_difference(x_lin, w_0_fitted - w_0_error, x_0_fitted + x_0_error),
    #                  energy_difference(x_lin, w_0_fitted + w_0_error, x_0_fitted - x_0_error), ls='--', lw=0.5,
    #                  color='r', alpha=0.2, label=r'1-$\sigma$ confidence band')
    # plt.plot(x_lin, energy_difference(x_lin, w_0_fitted, x_0_fitted), c='k', lw=1,
    #          label='Fit with' + '\n' + fr'$\omega_0$ = {w_0_fitted:.2f} ± {w_0_error:.2f}' +
    #                '\n' + fr'$x_0$ = {x_0_fitted:.4f} ± {x_0_error:.4f}')
    #
    # plt.scatter(peak_arange, my_diff)


    # plt.plot(wave_lengths, summed_counts - 8150)
    #
    # for x, err in zip(v_line_positions, position_errors):
    #     plt.axvline(x=x, color='r', linestyle='--', linewidth=0.5)
    #     # Fill between x-error and x+error for vertical lines
    #     plt.fill_betweenx(np.linspace(-1000, 60000, 100), x - err, x + err, color='red', alpha=0.2)
    #
    # plt.ylim(-1000, 60000)


    # plt.legend()
    # plt.show()

    return None


emission()
