# Imports

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from typing import Tuple
from scipy.optimize import curve_fit
from scipy.stats import norm


peak_locations = {
    'Hg': [(38, 60), (204, 224), (830, 850), (1020, 1032), (1037, 1050)],
    'Na': [(1100, 1105), (1105, 1115), (1820, 1828), (1892, 1903), (0, 2040)],
    'LED_absorption': [(200, 400), (615, 1200)]
}


def transform_data(file_name: str) -> Tuple[np.array, np.array]:
    dataframe = pd.read_csv(file_name, delimiter='\t', header=16, encoding='latin1')
    dataframe = dataframe.iloc[:-1, :]
    dataframe = dataframe.reset_index()
    dataframe.rename(columns={dataframe.columns[0]: 'channel', dataframe.columns[1]: 'counts'}, inplace=True)
    dataframe.channel = dataframe.channel.apply(lambda x: int(x.split(',')[0]))
    dataframe.counts = dataframe.counts.apply(lambda x: float(x.replace(',', '.')))

    return dataframe.channel.to_numpy(), dataframe.counts.to_numpy()


# channels_underground, counts_underground = transform_data('Hg_100ms_75avg_underground.txt')
# channels_lamp, counts_lamp = transform_data('Hg_100ms_75avg_lamp.txt')

channels_underground, counts_underground = transform_data('NA_100ms_75avg_underground.txt')
channels_lamp, counts_lamp = transform_data('NA_1ms_150avg_lamp.txt')


def channel_to_wave_length(channel_array):
    I = 395.164
    C1 = 0.189753
    C2 = -1.118880e-05
    C3 = -1.240474e-09

    return I + C1 * channel_array + C2 * channel_array ** 2 + C3 * channel_array ** 3


# channels = channels_underground
# channels = channel_to_wave_length(channels)
# counts = counts_lamp - counts_underground


channels, counts_underground = transform_data('LED_5ms_100avg_underground.txt')
_, counts_lamp = transform_data('LED_5ms_100avg_lamp.txt')
_, counts_absorption = transform_data('LED_5ms_500avg_absorption_65grad.txt')
counts = counts_absorption

peak_index = 1


# Example: Suppose the peak is around channel 1500 to 1700
peak_start = peak_locations.get('LED_absorption')[peak_index][0]
peak_end = peak_locations.get('LED_absorption')[peak_index][1]

# Select the data
channels_peak = channels[peak_start:peak_end]
counts_peak = counts[peak_start:peak_end]


def skewed_gaussian(x, a, xi, omega, alpha, c):
    # Compute t
    t = (x - xi) / omega
    # Compute the PDF part
    pdf = np.exp(-0.5 * t**2)
    # Compute the CDF part
    cdf = norm.cdf(alpha * t)
    # Combine parts
    return a * (pdf * cdf * (2 / omega)) + c


# Initial guesses
a_guess = np.max(counts_peak) - np.min(counts_peak)
xi_guess = channels_peak[np.argmax(counts_peak)]
omega_guess = (channels_peak[-1] - channels_peak[0]) / 6  # Approximate width
alpha_guess = 5  # Start with 0 (no skewness)
c_guess = np.min(counts_peak)

initial_guesses = [a_guess, xi_guess, omega_guess, alpha_guess, c_guess]

# Perform the fit
popt, pcov = curve_fit(skewed_gaussian, channels_peak, counts_peak, p0=initial_guesses, maxfev=10000)

# Extract optimal parameters
a_opt, xi_opt, omega_opt, alpha_opt, c_opt = popt


# Calculate the parameter errors
perr = np.sqrt(np.diag(pcov))
a_err, xi_err, omega_err, alpha_err, c_err = perr

print(f"Amplitude (a): {a_opt:.3f} ± {a_err:.3f}")
print(f"Mean (xi): {xi_opt:.3f} ± {xi_err:.3f}")
print(f"Width (omega): {omega_opt:.3f} ± {omega_err:.3f}")
print(f"Skewness (alpha): {alpha_opt:.3f} ± {alpha_err:.3f}")
print(f"Offset (c): {c_opt:.3f} ± {c_err:.3f}")


# Generate x-values for plotting the fitted curve
x_fit = np.linspace(channels_peak.min(), channels_peak.max(), 1000)
y_fit = skewed_gaussian(x_fit, a_opt, xi_opt, omega_opt, alpha_opt, c_opt)

plt.figure(figsize=(12, 5))
plt.plot(channels_peak, counts_peak, 'b.', label='Data')
plt.plot(x_fit, y_fit, 'r-', label='Skewed Gaussian Fit')
plt.xlabel('Channels')
plt.ylabel('Counts')
plt.title('Skewed Gaussian Fit on Peak')
plt.legend()
plt.show()


