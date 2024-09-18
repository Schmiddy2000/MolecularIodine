# Imports
import numpy as np
from matplotlib import pyplot as plt

from tools.reader import extract_data
from tools.transformers import channel_to_wave_length


channels, counts = extract_data('../data/channel_data/halo_7ms_250avg_absorption_50grad.txt')
wave_lengths = channel_to_wave_length(channels)


def averager(counts_array: np.array):
    averaged_counts_array = counts_array
    last_true_value = counts_array[0]

    for i in range(len(counts_array) - 2):
        new_last_true_value = averaged_counts_array[i + 1]
        averaged_counts_array[i + 1] = np.mean([last_true_value, averaged_counts_array[i + 1],
                                                averaged_counts_array[i + 2]])
        last_true_value = new_last_true_value

    return averaged_counts_array


plt.figure(figsize=(12, 5))
plt.plot(wave_lengths, counts)
plt.plot(wave_lengths, averager(counts))
plt.show()
