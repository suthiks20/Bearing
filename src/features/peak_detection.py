import numpy as np
from scipy.signal import find_peaks, peak_widths


def envelope(signal, win_len=201):
    env = np.abs(signal)
    if win_len < 3:
        return env
    kernel = np.ones(win_len) / win_len
    return np.convolve(env, kernel, mode='same')


def adaptive_peak_windows(env, fs, height_factor=0.6, distance_s=0.01,
                          rel_height_for_width=0.5, min_win_samples=256,
                          max_win_samples=4096):
    """Return [(peak_idx, left_idx, right_idx), ...] for peaks on the envelope."""
    thr = np.mean(env) + height_factor * np.std(env)
    distance = max(1, int(distance_s * fs))
    peaks, _ = find_peaks(env, height=thr, distance=distance,
                          prominence=0.1 * np.std(env))
    if len(peaks) == 0:
        return []

    widths_results = peak_widths(env, peaks, rel_height=rel_height_for_width)
    widths   = widths_results[0]
    left_ips = widths_results[2].astype(int)
    right_ips = widths_results[3].astype(int)

    windows = []
    for i, p in enumerate(peaks):
        w = max(1, int(widths[i]))
        extra = max(int(1.0 * w), int(0.01 * fs))
        l = max(0, left_ips[i] - extra)
        r = min(len(env) - 1, right_ips[i] + extra)

        if (r - l) < min_win_samples:
            half = min_win_samples // 2
            l = max(0, p - half)
            r = min(len(env) - 1, p + half)
        if (r - l) > max_win_samples:
            half = max_win_samples // 2
            l = max(0, p - half)
            r = min(len(env) - 1, p + half)

        windows.append((int(p), int(l), int(r)))
    return windows
