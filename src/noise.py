"""Measurement noise on the bus voltages (E9, R2-9).

White Gaussian noise at a given SNR in dB. The reference signal level is one
value per feeder, the median over buses of the per-bus voltage std, so every
bus gets the same noise whatever its own fluctuation. A per-bus level would
make the noise depend on the position in the feeder, which is the effect E9
measures.
"""

import numpy as np

# Second entry of the noise generator's seed sequence, so the noise never
# shares a stream with the training seeds
NOISE_STREAM = 9


def voltage_noise_reference(all_x):
    """Median over buses of the voltage std across all graphs and time steps (p.u.).

    all_x is (graphs, buses, time steps) and holds the phase-averaged voltage
    of the active phases (src.data.load_evcs_data), so the zero readings of
    missing phases never enter the std.
    """
    return float(np.median(all_x.std(axis=(0, 2), dtype=np.float64)))


def add_voltage_noise(all_x, snr_db, seed):
    """Return all_x plus white Gaussian noise at snr_db, and the values for the result rows.

    sigma = 10^(-snr_db / 20) * voltage_noise_reference(all_x). The standard
    normal draw depends only on seed, so all noise levels of a seed add the
    same pattern scaled by their sigma, and all methods and backbones of a seed
    see the same data. The generator is separate from the global numpy and
    torch generators, which stay untouched, and all_x is not modified.
    """
    ref = voltage_noise_reference(all_x)
    sigma = 10.0 ** (-snr_db / 20.0) * ref
    rng = np.random.default_rng([seed, NOISE_STREAM])
    noise = rng.standard_normal(all_x.shape, dtype=np.float32)
    noise *= np.float32(sigma)
    noisy = all_x + noise
    return noisy, {'V_noise_snr_db': float(snr_db), 'V_noise_ref': ref, 'V_noise_sigma': sigma}
