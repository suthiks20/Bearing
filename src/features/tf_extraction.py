"""
Time-frequency extraction: CWT (fallback) and FrSST via ssqueezepy.

IMPORTANT — fallback policy:
  If ssqueezepy is absent, FRSST_AVAILABLE is False and a warning is printed
  at import time. process_folder() will also warn explicitly when it degrades
  to CWT. The fallback is NEVER silent.
"""
import warnings

import numpy as np
import pywt
from scipy.signal import find_peaks

# Explicit availability check — never hidden in a silent try/except
try:
    from ssqueezepy import ssq_cwt as _ssq_cwt
    FRSST_AVAILABLE = True
except ImportError:
    _ssq_cwt = None
    FRSST_AVAILABLE = False
    warnings.warn(
        "ssqueezepy is not installed — FrSST/SST is UNAVAILABLE. "
        "All TF transforms will use plain CWT instead. "
        "To restore FrSST: pip install ssqueezepy",
        ImportWarning,
        stacklevel=1,
    )


def compute_cwt_tf(window_signal, fs, wavelet='morl', scales=None):
    if scales is None:
        scales = np.arange(1, 256)
    coeffs, freqs = pywt.cwt(window_signal, scales, wavelet,
                              sampling_period=1.0 / fs)
    return np.abs(coeffs), freqs


def compute_sst_tf(window_signal, fs):
    """
    Synchrosqueezed CWT (FrSST) via ssqueezepy 0.6.x.

    ssqueezepy 0.6.6 returns a 4-tuple: (Tx, Wx, ssq_freqs, scales).
    Older versions returned 3-tuple (Tx, Wx, freqs) or 2-tuple (Tx, freqs).
    We index positionally so this works across versions without unpacking.

    Raises RuntimeError if ssqueezepy is not installed — never falls back
    silently; callers must check FRSST_AVAILABLE first.
    """
    if not FRSST_AVAILABLE:
        raise RuntimeError(
            "ssqueezepy not installed — cannot run FrSST. "
            "Check FRSST_AVAILABLE before calling compute_sst_tf, "
            "or select method='cwt'."
        )
    x = np.asarray(window_signal, dtype=float)
    try:
        out = _ssq_cwt(x, fs=fs)
    except TypeError:
        # Pre-0.6 API did not accept fs as a keyword argument
        out = _ssq_cwt(x)

    Tx    = out[0]                          # synchrosqueezed coefficients
    freqs = out[2] if len(out) >= 3 else out[1]   # ssq_freqs array
    return np.abs(Tx), freqs


def extract_top_freqs_from_tf(tf_mag, freqs, top_n=3, exclusion_hz=1.0):
    """Pick top_n dominant frequencies from a time-frequency magnitude map."""
    avg = np.mean(tf_mag, axis=1)
    valid = freqs > exclusion_hz
    if not np.any(valid):
        return []

    f = freqs[valid]
    a = avg[valid]
    if np.max(a) <= 0:
        return []

    a = a / (np.max(a) + 1e-12)
    peaks, _ = find_peaks(a, height=np.mean(a) + 0.2 * np.std(a), distance=1)

    if len(peaks) == 0:
        idx = np.argsort(a)[-top_n:]
    else:
        pick = np.argsort(a[peaks])[-top_n:]
        idx  = peaks[pick]

    order = np.argsort(f[idx])
    sel_f = f[idx][order]
    sel_a = a[idx][order]
    return list(zip(sel_f.tolist(), sel_a.tolist()))
