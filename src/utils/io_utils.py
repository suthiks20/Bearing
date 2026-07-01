import numpy as np


def load_nasa_file(path):
    """Load NASA IMS whitespace-separated vibration file (20480 samples, N channels).
    2nd_test: 4 channels (one accelerometer per bearing).
    """
    return np.loadtxt(path)
