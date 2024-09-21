# Imports
import numpy as np
from matplotlib import pyplot as plt

from tools.reader import extract_data
from tools.transformers import channel_to_wave_length, get_slice_indices
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


wave_length_array, halogen_isolation = linearize_count_dips_halogen_lamp(23, False)


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


_, led_isolation = linearize_count_dips_led_lamp(23, False)


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


_, combined_isolation = linearize_count_dips_both_lamps(23, False)

normalization_factor = np.max(combined_isolation) - np.min(combined_isolation)

dip_positions = np.array([])

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
