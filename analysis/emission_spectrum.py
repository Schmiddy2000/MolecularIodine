# Imports
import numpy as np
from matplotlib import pyplot as plt

from tools.reader import extract_data
from tools.transformers import channel_to_wave_length, averager

channels, initial_counts = extract_data('../data/emission_data/laser_38_100ms_500_avg_initial.txt')
averaged_counts = np.zeros(len(channels))
_, background_counts = extract_data('../data/emission_data/Laser_500ms_500avg_underground.txt')

for i in range(1, 7):
    file_path = f'../data/emission_data/laser_43_100ms_500avg_{i}.txt'
    averaged_counts += extract_data(file_path)[1]

counter_counts = (extract_data('../data/emission_data/laser_43_100ms_500avg_1.txt')[1] -
                  extract_data('../data/emission_data/laser_43_100ms_500avg_2.txt')[1])

wave_lengths = channel_to_wave_length(channels)
averaged_counts = averaged_counts / 6
normalized_counts = background_counts - initial_counts

plt.figure(figsize=(12, 5))
# plt.plot(wave_lengths, averaged_counts - counter_counts)
plt.plot(wave_lengths, normalized_counts)

plt.plot(wave_lengths, [val if val > 0 else 0 for val in averaged_counts - averager(averaged_counts)])
plt.show()

