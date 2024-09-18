# Imports
import numpy as np
from matplotlib import pyplot as plt

from tools.reader import extract_data
from tools.transformers import channel_to_wave_length


underground_channels, underground_counts = extract_data('../data/channel_data/halo_8ms_250avg_noabsorption.txt')
absorption_50_channels, absorption_50_counts = extract_data('../data/channel_data/halo_7ms_250avg_absorption_50grad.txt')
absorption_65_channels, absorption_65_counts = extract_data('../data/channel_data/halo_7ms_250avg_absorption_65grad.txt')

underground_max = 44700
absorption_50_max = 60500

wave_lengths = channel_to_wave_length(underground_channels)
diff_counts_50 = (absorption_50_max / underground_max) * underground_counts - absorption_50_counts
diff_counts_65 = (absorption_50_max / underground_max) * underground_counts - absorption_50_counts

plt.figure(figsize=(12, 5))
plt.plot(wave_lengths, underground_counts, label='Without probe')
plt.plot(wave_lengths, absorption_50_counts, label='50°')
plt.plot(wave_lengths, absorption_65_counts, label='65°')
plt.plot(wave_lengths, diff_counts_50, label='scaled diff no probe and 50°')
plt.plot(wave_lengths, absorption_50_counts - absorption_65_counts, label='50° - 65°')
plt.legend()
plt.show()
