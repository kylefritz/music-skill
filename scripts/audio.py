"""audio.py - the measurements both scripts share: loudness, true peak, band energy, WAV in and out.

Everything works at 48 kHz stereo float arrays shaped (samples, 2).
"""
import subprocess
import wave

import numpy as np
from scipy import signal

SR = 48000
BAND_EDGES = [20, 80, 250, 800, 2500, 7000, 20000]
BAND_NAMES = ["sub", "bass", "low mid", "mid", "high", "air"]

# ITU-R BS.1770 K-weighting at 48 kHz: a high shelf for the head, then a high-pass.
_K_SHELF = ([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585])
_K_HIGHPASS = ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def k_weight(x):
    y = signal.lfilter(*_K_SHELF, x, axis=0)
    return signal.lfilter(*_K_HIGHPASS, y, axis=0)


def _block_power(x, seconds, hop_seconds):
    """Mean square of K-weighted audio over sliding blocks, channels summed."""
    k = k_weight(np.atleast_2d(x.T).T.astype(np.float64))
    n, hop = int(seconds * SR), int(hop_seconds * SR)
    if len(k) < n:
        return np.array([np.sum(np.mean(k ** 2, axis=0))])
    c = np.concatenate([np.zeros((1, k.shape[1])), np.cumsum(k ** 2, axis=0)])
    starts = np.arange(0, len(k) - n + 1, hop)
    return np.sum((c[starts + n] - c[starts]) / n, axis=1)


def lufs(x):
    """Integrated loudness, BS.1770-4: 400 ms blocks, -70 LUFS absolute gate, -10 LU relative gate."""
    p = _block_power(x, 0.4, 0.1)
    loud = lambda q: -0.691 + 10 * np.log10(max(np.mean(q), 1e-12))
    p = p[-0.691 + 10 * np.log10(np.maximum(p, 1e-12)) > -70]
    if not len(p):
        return -70.0
    gated = p[-0.691 + 10 * np.log10(p) > loud(p) - 10]
    return float(loud(gated if len(gated) else p))


def short_term(x, hop_seconds=1.0):
    """Short-term loudness (3 s windows) over time: (times of window centers, LUFS)."""
    p = _block_power(x, 3.0, hop_seconds)
    return np.arange(len(p)) * hop_seconds + 1.5, -0.691 + 10 * np.log10(np.maximum(p, 1e-12))


def loudness_range(x):
    """EBU Tech 3342 loudness range: spread of short-term loudness, 10th to 95th percentile."""
    _, st = short_term(x)
    st = st[st > -70]
    if len(st) < 2:
        return 0.0
    st = st[st > 10 * np.log10(np.mean(10 ** (st / 10))) - 20]
    return float(np.percentile(st, 95) - np.percentile(st, 10)) if len(st) > 1 else 0.0


def true_peak(x):
    """Peak after 4x oversampling, in dBTP: what a DAC or a lossy codec will actually reach."""
    return float(db(np.max(np.abs(signal.resample_poly(x, 4, 1, axis=0)))))


def band_frames(mono, n_fft=2048, hop=1024):
    """RMS amplitude in each of the six bands, frame by frame: array (frames, 6)."""
    f, _, z = signal.stft(mono, SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    power = np.abs(z) ** 2
    out = np.zeros((power.shape[1], 6))
    for b in range(6):
        sel = (f >= BAND_EDGES[b]) & (f < BAND_EDGES[b + 1])
        out[:, b] = np.sqrt(power[sel].sum(axis=0))
    return out


def band_profile(mono):
    """Where a sound puts its energy while it is sounding: six band amplitudes."""
    fr = band_frames(mono)
    total = fr.sum(axis=1)
    if not len(total) or total.max() <= 0:
        return [0.0] * 6
    active = total > total.max() * 10 ** (-40 / 20)
    return [float(v) for v in fr[active].mean(axis=0)]


def read_audio(path):
    """Decode anything ffmpeg can read to 48 kHz stereo float."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).astype(np.float64)


def write_wav(path, x, seed=0):
    """16-bit stereo WAV with triangular dither, so quiet tails don't turn to grit."""
    rng = np.random.default_rng(seed)
    dither = (rng.random(x.shape) - rng.random(x.shape)) / 32768
    pcm = np.clip(np.round((x + dither) * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
