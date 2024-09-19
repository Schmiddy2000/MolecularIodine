# Imports
import numpy as np
from matplotlib import pyplot as plt

from tools.reader import extract_data
from tools.transformers import channel_to_wave_length, averager


channels, counts = extract_data('../data/channel_data/halo_7ms_250avg_absorption_50grad.txt')
wave_lengths = channel_to_wave_length(channels)


plt.figure(figsize=(12, 5))
plt.plot(wave_lengths, counts)
plt.plot(wave_lengths, averager(counts))
plt.show()
