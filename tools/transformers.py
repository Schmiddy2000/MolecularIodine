# Imports
import numpy as np


def channel_to_wave_length(channel_array: np.array) -> np.array:
    I = 395.164
    C1 = 0.189753
    C2 = -1.118880e-05
    C3 = -1.240474e-09

    return I + C1 * channel_array + C2 * channel_array ** 2 + C3 * channel_array ** 3
