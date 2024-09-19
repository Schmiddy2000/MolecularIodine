# Imports
import matplotlib.pyplot as plt
import numpy as np

from tools.reader import extract_data
from tools.transformers import channel_to_wave_length
from scipy.constants import h, c


channels, counts = extract_data('../data/emission_data/laser_254_1000ms_150avg_vertical_14.txt')
_, background_counts = extract_data('../data/emission_data/Laser_500ms_500avg_underground.txt')
wave_lengths = channel_to_wave_length(channels)

peak_positions = np.array([])

# plt.plot(wave_lengths, counts - background_counts)
# plt.show()


my_list = [i for i in range(1, 22)]

peak_position = 1e-9 * np.array([532.2, 538.5, 544.6, 550.95, 557.5, 563.95, 570.6, 577.4, 584.2, 591.3, 598, 605.7,
                                 612, 620.5,627.6, 635.8, 643.7, 652, 659.9, 668.46, 676.8, 685, 693])
# quantum_number= np.array([i for i in range(1,21)])

energy_differences = 6.242e18 * h * c * np.array([1 / abs(peak_position[i] - peak_position[i + 1]) for i in range(len(peak_position) - 2)])

plt.figure(figsize=(12, 5))
plt.xlabel(r'Quantum state $\nu$')
plt.scatter(my_list, energy_differences)
plt.show()
print(my_list)
print(energy_differences)
