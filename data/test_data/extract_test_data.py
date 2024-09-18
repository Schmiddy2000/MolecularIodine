# Imports

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from typing import Tuple
from scipy.stats import linregress
from scipy.optimize import curve_fit


def transform_data(file_name: str) -> Tuple[np.array, np.array]:
    dataframe = pd.read_csv(file_name, delimiter='\t', header=16, encoding='latin1')
    dataframe = dataframe.iloc[:-1, :]
    dataframe = dataframe.reset_index()
    dataframe.rename(columns={dataframe.columns[0]: 'channel', dataframe.columns[1]: 'counts'}, inplace=True)
    dataframe.channel = dataframe.channel.apply(lambda x: int(x.split(',')[0]))
    dataframe.counts = dataframe.counts.apply(lambda x: float(x.replace(',', '.')))
    
    return dataframe.channel.to_numpy(), dataframe.counts.to_numpy()


channels_underground, counts_underground = transform_data('LED_5ms_100avg_underground.txt')
_, counts_lamp = transform_data('LED_5ms_100avg_lamp.txt')
_, counts_absorption = transform_data('LED_5ms_500avg_absorption_65grad.txt')


def channel_to_wave_length(channel_array):
    I = 395.164
    C1 = 0.189753
    C2 = -1.118880e-05
    C3 = -1.240474e-09

    return I + C1 * channel_array + C2 * channel_array ** 2 + C3 * channel_array ** 3


LED_max = 61150
absorption_max = 51800

channels = channels_underground # channel_to_wave_length(channels_underground)
counts = counts_absorption # (absorption_max / LED_max) * counts_lamp - counts_absorption
# (absorption_max / LED_max) *

# channels, counts = transform_data('33.txt')
# channels = channel_to_wave_length(channels)
#
plt.plot(channels, counts)
plt.show()

# Example: Suppose the peak is around channel 1500 to 1700
peak_start = 0
peak_end = 2040

# Select the data
channels_peak = channels[peak_start:peak_end]
counts_peak = counts[peak_start:peak_end]


def gaussian(x, a, mu, sigma, c):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c


# Initial guesses
a_guess = np.max(counts_peak) - np.min(counts_peak)
mu_guess = channels_peak[np.argmax(counts_peak)]
sigma_guess = (channels_peak[-1] - channels_peak[0]) / 6  # Rough estimate
c_guess = np.min(counts_peak)

initial_guesses = [a_guess, mu_guess, sigma_guess, c_guess]

popt, pcov = curve_fit(gaussian, channels_peak, counts_peak, p0=initial_guesses)

# Extract the optimal parameters
a_opt, mu_opt, sigma_opt, c_opt = popt

perr = np.sqrt(np.diag(pcov))
a_err, mu_err, sigma_err, c_err = perr


print(f"Amplitude (a): {a_opt:.3f} ± {a_err:.3f}")
print(f"Mean (mu): {mu_opt:.3f} ± {mu_err:.3f}")
print(f"Sigma (σ): {sigma_opt:.3f} ± {sigma_err:.3f}")
print(f"Offset (c): {c_opt:.3f} ± {c_err:.3f}")


# Generate x-values for plotting the fitted curve
x_fit = np.linspace(channels_peak.min(), channels_peak.max(), 1000)
y_fit = gaussian(x_fit, a_opt, mu_opt, sigma_opt, c_opt)

plt.figure(figsize=(12, 5))

plt.plot(channels_peak, counts_peak, 'b.', label='Data')
plt.plot(x_fit, y_fit, 'r-', label='Gaussian Fit')

# params = linregress(channels[1200:2000], counts[1200:2000])
# print(params[0])
# print(params[1])
#
# plt.plot(channels, counts)
# plt.plot(channels_peak, counts_peak)
# # plt.plot(channels, [max(0, c - params[1]) for c in counts])
# plt.tight_layout()
# plt.show()


wave_lengths = np.array([253.7, 265, 404.7, 435.8, 546.1, 578])
channel_values = np.array([50, 217, 601, 841, 1028, 1041])


def channel_transformation(x, a, b, c):
    return
