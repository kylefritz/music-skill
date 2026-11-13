#!/usr/bin/env python3
"""listen.py - hear a piece of audio as numbers and one picture, for a reader that can't listen.

    uv run --no-project --with numpy --with scipy --with pillow python -I listen.py look  TRACK [--sheet out.png] [--json]
    ... listen.py check TRACK --lufs -14 --peak -1 [--seconds 30] [--bpm 124] [--key "G minor"]
    ... listen.py mix   STEMS_DIR | a.wav b.wav ...

look   tempo, key, loudness, true peak, where the energy sits, stereo width, clipping, and a
       timeline of loudness and band energy that shows the intro, drops and breakdowns. --sheet
       draws it: waveform, spectrogram, six band lanes, loudness curve.
check  pass or fail against targets, exit 1 on a miss. Run it before handing a file over.
mix    stems against each other: which pairs fill the same band at the same moment.

Reads anything ffmpeg can decode (wav, mp3, m4a, flac, the sound of a video).
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import ndimage, signal

import audio
from audio import BAND_EDGES, BAND_NAMES, SR

NOTE = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# Krumhansl & Kessler 1982: how well each of the 12 notes fits a major or a minor key.
KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def clock(s):
    return f"{int(s // 60)}:{s % 60:04.1f}"


def tempo(mono):
    """BPM from how often the spectrum jumps. Returns (bpm, strength); strength under ~2 is a guess."""
    _, _, z = signal.stft(mono, SR, nperseg=1024, noverlap=1024 - 480, boundary=None, padded=False)   # 100 frames/s
    flux = np.maximum(np.diff(np.log1p(100 * np.abs(z)), axis=1), 0).sum(axis=0)
    flux = np.maximum(flux - ndimage.uniform_filter1d(flux, 100), 0)
    n = len(flux)
    if n < 400 or not flux.any():
        return 0.0, 0.0
    ac = np.fft.irfft(np.abs(np.fft.rfft(flux, 2 * n)) ** 2)[:n]
    ac /= ac[0]
    bpms = np.arange(70, 190.05, .1)
    lag = 6000 / bpms
    score = np.zeros(len(bpms))
    for m in range(1, 17):                                   # a comb: the beat, two beats, ... four bars
        ok = m * lag < n - 1
        score[ok] += np.interp(m * lag[ok], np.arange(n), ac)
    score *= np.exp(-.5 * (np.log2(bpms / 120) / .6) ** 2)   # between half and double time, prefer near 120
    best = int(np.argmax(score))
    return float(round(bpms[best], 1)), float(score[best] / (np.median(score) + 1e-9))


def key(mono):
    """Home note and major/minor. The 12-note energy is matched to Krumhansl-Kessler profiles,
    and the bass gets a vote, because the lowest part sits on the home note most of the time.
    On 24 tracks from banger.py with known keys: profiles alone found the home note in 14,
    with the bass vote 19. Major vs minor is a weaker call; modes are not told apart."""
    if len(mono) < 65536:
        mono = np.pad(mono, (0, 65536 - len(mono)))
    f, _, z = signal.stft(mono, SR, nperseg=65536, noverlap=32768, boundary=None, padded=False)
    mag = np.abs(z).mean(axis=1)

    def chroma(lo, hi):
        sel = (f >= lo) & (f <= hi)
        pc = np.round(12 * np.log2(f[sel] / 440) + 69).astype(int) % 12
        return np.bincount(pc, weights=mag[sel], minlength=12)

    full, low, upper = chroma(40, 2000), chroma(35, 160), chroma(200, 4000)
    if not full.any():
        return []
    found = []
    for tonic in range(12):
        minor = upper[(tonic + 3) % 12] >= upper[(tonic + 4) % 12]       # which third is louder
        fit = float(np.corrcoef(full, np.roll(KK_MINOR if minor else KK_MAJOR, tonic))[0, 1])
        found.append((fit + float(low[tonic] / max(low.max(), 1e-12)), f"{NOTE[tonic]} {'minor' if minor else 'major'}"))
    return sorted(found, reverse=True)[:3]


def band_levels(x):
    fr = audio.band_frames(x.mean(axis=1))
    return 10 * np.log10(np.maximum((fr ** 2).mean(axis=0), 1e-20)), fr


def analyze(x):
    mono = x.mean(axis=1)
    peak = np.abs(x).max()
    quiet = np.abs(x).max(axis=1) < 10 ** (-60 / 20)
    head = int(np.argmax(~quiet)) if (~quiet).any() else len(x)
    tail = int(np.argmax(~quiet[::-1])) if (~quiet).any() else 0
    bpm, strength = tempo(mono)
    levels, frames = band_levels(x)
    share = 10 ** (levels / 10)
    mid, side = mono, (x[:, 0] - x[:, 1]) / 2
    low = audio.signal.sosfilt(signal.butter(4, 120, "low", fs=SR, output="sos"), x, axis=0)
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 1.0
    loud = audio.lufs(x)
    tp = audio.true_peak(x)

    # A timeline in equal windows: 8 bars when the tempo is clear, else about a twelfth of the track.
    dur = len(x) / SR
    win = 8 * 240 / bpm if strength >= 2 and bpm and dur > 16 * 240 / bpm else max(dur / 12, 3.0)
    fps = SR / 1024
    timeline = []
    t = head / SR
    while t < dur - tail / SR - win * .25:
        a, b = int(t * fps), max(int(min(t + win, dur) * fps), int(t * fps) + 1)
        seg = x[int(t * SR):int(min(t + win, dur) * SR)]
        lv = 10 * np.log10(np.maximum((frames[a:b] ** 2).mean(axis=0), 1e-20)) - levels
        timeline.append(dict(start=round(t, 2), end=round(min(t + win, dur), 2), lufs=round(audio.lufs(seg), 1),
                             bands=[round(float(v), 1) for v in lv]))
        t += win
    return dict(
        seconds=round(dur, 2), lufs=round(loud, 2), true_peak=round(tp, 2), loudness_range=round(audio.loudness_range(x), 1),
        peak_to_loudness=round(tp - loud, 1), bpm=bpm, bpm_strength=round(strength, 1),
        key=[dict(key=k, score=round(r, 2)) for r, k in key(mono)],
        balance_db={n: round(float(v - levels.max()), 1) for n, v in zip(BAND_NAMES, levels)},
        stereo_correlation=round(corr(x[:, 0], x[:, 1]), 2), low_end_correlation=round(corr(low[:, 0], low[:, 1]), 2),
        side_vs_mid_db=round(float(audio.db(side.std() + 1e-12) - audio.db(mid.std() + 1e-12)), 1),
        clipped_samples=int((np.abs(x) >= .999).sum()), sample_peak_db=round(float(audio.db(peak)), 2),
        silence_head=round(head / SR, 2), silence_tail=round(tail / SR, 2),
        dc_offset=round(float(np.abs(x.mean(axis=0)).max()), 5), window_seconds=round(win, 2), timeline=timeline)


def notes_on(r):
    """Plain warnings an agent should act on."""
    out = []
    if r["clipped_samples"] > 8:
        out.append(f"{r['clipped_samples']} samples at full scale: clipped. Lower the gain before the limiter.")
    if r["true_peak"] > -1:
        out.append(f"true peak {r['true_peak']} dBTP is over -1: lossy encoding can clip it.")
    if r["low_end_correlation"] < .6:
        out.append(f"low end is wide (correlation {r['low_end_correlation']} under 120 Hz): bass can vanish in mono.")
    if r["stereo_correlation"] < 0:
        out.append("left and right are out of phase: it will cancel on a phone speaker.")
    if r["bpm_strength"] < 2:
        out.append("no steady beat found: treat the BPM as a guess.")
    if r["key"] and len(r["key"]) > 1 and r["key"][0]["score"] - r["key"][1]["score"] < .1:
        out.append(f"key is a close call between {r['key'][0]['key']} and {r['key'][1]['key']}.")
    if r["silence_head"] > .3:
        out.append(f"{r['silence_head']} s of silence before the sound starts.")
    if r["peak_to_loudness"] < 7:
        out.append(f"peak-to-loudness is {r['peak_to_loudness']} dB: heavily limited, likely flat and tiring.")
    if r["dc_offset"] > .01:
        out.append(f"DC offset {r['dc_offset']}: high-pass at 20 Hz.")
    return out


def report(path, r):
    k = r["key"]
    lines = [
        f"{os.path.basename(path)}  {clock(r['seconds'])}  {r['bpm']:.1f} BPM (strength {r['bpm_strength']})  "
        + (f"{k[0]['key']} (score {k[0]['score']}; next {k[1]['key']} {k[1]['score']})" if k else "no pitch"),
        f"  loudness {r['lufs']} LUFS  true peak {r['true_peak']} dBTP  range {r['loudness_range']} LU  "
        f"peak-to-loudness {r['peak_to_loudness']} dB",
        "  balance, dB under the loudest band  " + "  ".join(f"{n} {v:.0f}" for n, v in r["balance_db"].items()),
        "  stereo  mono" if r["side_vs_mid_db"] < -60 else
        f"  stereo  correlation {r['stereo_correlation']}  under 120 Hz {r['low_end_correlation']}  "
        f"side is {r['side_vs_mid_db']} dB vs mid",
        "",
        f"  timeline, {r['window_seconds']:.1f} s windows" + (" (8 bars)" if r["bpm_strength"] >= 2 else "")
        + ": loudness, then each band in dB against its level over the whole track",
        "    start     LUFS   " + "".join(f"{n:>8}" for n in BAND_NAMES)]
    for w in r["timeline"]:
        lines.append(f"    {clock(w['start']):>7}  {w['lufs']:6.1f}   " + "".join(f"{v:+8.0f}" for v in w["bands"]))
    lines += ["  ! " + n for n in notes_on(r)]
    return "\n".join(lines)


# ---------- the sheet ----------

def heat(v):
    """0..1 -> a dark-to-bright color ramp."""
    stops = np.array([[8, 10, 30], [70, 20, 110], [190, 55, 90], [250, 150, 50], [255, 245, 190]], float)
    p = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(p.astype(int), len(stops) - 2)
    return (stops[i] + (stops[i + 1] - stops[i]) * (p - i)[..., None]).astype(np.uint8)


def sheet(path, x, r, out):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font, small = ImageFont.load_default(size=17), ImageFont.load_default(size=13)
    except TypeError:
        font = small = ImageFont.load_default()
    W, left, right = 1600, 84, 16
    pw = W - left - right
    rows = dict(wave=110, spec=300, bands=6 * 24, loud=110)
    H = 96 + sum(rows.values()) + 4 * 26 + 30
    im = Image.new("RGB", (W, H), (250, 250, 247))
    d = ImageDraw.Draw(im)
    ink, grey, dur, mono = (20, 28, 45), (120, 128, 140), len(x) / SR, x.mean(axis=1)
    k = r["key"]
    d.text((left, 12), f"{os.path.basename(path)}   {clock(dur)}   {r['bpm']:.1f} BPM   {k[0]['key'] if k else ''}", fill=ink, font=font)
    d.text((left, 38), f"{r['lufs']} LUFS   true peak {r['true_peak']} dBTP   range {r['loudness_range']} LU   "
           + "   ".join(f"{n} {v:.0f} dB" for n, v in r["balance_db"].items()), fill=ink, font=small)
    for i, n in enumerate(notes_on(r)[:2]):
        d.text((left, 58 + 16 * i), "! " + n, fill=(170, 40, 30), font=small)
    y = 96
    tx = lambda t: left + t / dur * pw

    def label(text, y0, h):
        d.text((8, y0 + h / 2 - 8), text, fill=grey, font=small)

    # waveform: the loudest and quietest sample under each pixel
    h = rows["wave"]
    cols = np.array_split(mono, pw)
    hi, lo = np.array([c.max() for c in cols]), np.array([c.min() for c in cols])
    d.rectangle([left, y, left + pw, y + h], fill=(255, 255, 255))
    for i in range(pw):
        d.line([left + i, y + h / 2 - hi[i] * h / 2, left + i, y + h / 2 - lo[i] * h / 2], fill=(60, 80, 160))
    label("wave", y, h)
    y += h + 26

    # spectrogram on a log frequency axis, 30 Hz to 16 kHz
    h = rows["spec"]
    f, _, z = signal.stft(mono, SR, nperseg=4096, noverlap=2048, boundary=None, padded=False)
    mag = audio.db(np.abs(z))
    freqs = np.geomspace(30, 16000, h)
    grid = np.empty((h, pw))
    src_t = np.linspace(0, mag.shape[1] - 1, pw)
    for row, hz in enumerate(freqs):
        i = np.searchsorted(f, hz)
        band = mag[max(i - 1, 0):i + max(2, int(i * .03))].max(axis=0)
        grid[h - 1 - row] = np.interp(src_t, np.arange(len(band)), band)
    top = np.percentile(grid, 99.5)
    im.paste(Image.fromarray(heat((grid - (top - 70)) / 70)), (left, y))
    for hz in (50, 100, 250, 1000, 4000, 10000):
        yy = y + h - 1 - np.searchsorted(freqs, hz)
        d.text((left - 52, yy - 7), f"{hz if hz < 1000 else str(hz // 1000) + 'k'} Hz", fill=grey, font=small)
        d.line([left - 5, yy, left, yy], fill=grey)
    y += h + 26

    # six band lanes: energy in each band over time, 50 dB of range per lane
    fr = audio.db(audio.band_frames(mono))
    lane = 24
    for b in range(6):
        v = np.interp(np.linspace(0, len(fr) - 1, pw), np.arange(len(fr)), fr[:, 5 - b])
        strip = heat((v - (fr[:, 5 - b].max() - 50)) / 50)
        im.paste(Image.fromarray(np.repeat(strip[None], lane - 3, axis=0)), (left, y + b * lane))
        d.text((8, y + b * lane + 3), BAND_NAMES[5 - b], fill=grey, font=small)
    y += rows["bands"] + 26

    # short-term loudness
    h = rows["loud"]
    t, st = audio.short_term(x, .5)
    d.rectangle([left, y, left + pw, y + h], fill=(255, 255, 255))
    lo_, hi_ = min(-40, r["lufs"] - 20), max(-5, st.max() + 2)
    ly = lambda v: y + h - (np.clip(v, lo_, hi_) - lo_) / (hi_ - lo_) * h
    for v in range(int(np.ceil(lo_ / 10) * 10), int(hi_) + 1, 10):
        d.line([left, ly(v), left + pw, ly(v)], fill=(225, 228, 232))
        d.text((left - 34, ly(v) - 7), str(v), fill=grey, font=small)
    d.line([left, ly(r["lufs"]), left + pw, ly(r["lufs"])], fill=(200, 60, 50))
    if len(t) > 1:
        d.line([(tx(a), ly(b)) for a, b in zip(t, st)], fill=ink, width=2)
    label("LUFS", y, h - 60)
    y += h

    # time axis, with a tick at each timeline window (8 bars when the tempo is known)
    step = next(s for s in (5, 10, 15, 30, 60, 120, 300) if dur / s <= 14)
    for s in np.arange(0, dur, step):
        d.line([tx(s), y, tx(s), y + 6], fill=grey)
        d.text((tx(s) + 3, y + 6), clock(s)[:-2], fill=grey, font=small)
    for w in r["timeline"]:
        d.line([tx(w["start"]), 96, tx(w["start"]), y], fill=(255, 255, 255) if r["bpm_strength"] >= 2 else (200, 200, 200))
    im.save(out)


# ---------- commands ----------

def cmd_look(a):
    x = audio.read_audio(a.file)
    r = analyze(x)
    print(json.dumps(r, indent=1) if a.json else report(a.file, r))
    if a.sheet:
        sheet(a.file, x, r, a.sheet)
        print(a.sheet)


def cmd_check(a):
    r = analyze(audio.read_audio(a.file))
    k = r["key"][0]["key"] if r["key"] else ""
    tests = [("loudness", abs(r["lufs"] - a.lufs) <= a.tol, f"{r['lufs']} LUFS, want {a.lufs} ±{a.tol}"),
             ("true peak", r["true_peak"] <= a.peak + .05, f"{r['true_peak']} dBTP, want {a.peak} or under"),
             ("clipping", r["clipped_samples"] <= 8, f"{r['clipped_samples']} samples at full scale"),
             ("start", r["silence_head"] <= .3, f"{r['silence_head']} s of silence at the start")]
    if a.seconds:
        tests.append(("length", abs(r["seconds"] - a.seconds) <= a.seconds_tol, f"{r['seconds']} s, want {a.seconds} ±{a.seconds_tol}"))
    if a.bpm:
        tests.append(("tempo", abs(r["bpm"] - a.bpm) <= 1, f"{r['bpm']} BPM, want {a.bpm} ±1"))
    if a.key:
        tests.append(("home note", k.split(" ")[0].lower() == a.key.split(" ")[0].lower(), f"{k}, want {a.key}"))
    for name, ok, detail in tests:
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    sys.exit(0 if all(t[1] for t in tests) else 1)


def cmd_mix(a):
    files = sorted(glob.glob(os.path.join(a.stems[0], "*.wav"))) if os.path.isdir(a.stems[0]) else a.stems
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    frames = [audio.band_frames(audio.read_audio(f).mean(axis=1)) for f in files]
    n = min(len(f) for f in frames)
    frames = [f[:n] for f in frames]
    loud = [audio.lufs(audio.read_audio(f)) for f in files]
    print(f"{'stem':<10}{'LUFS':>7}  " + "".join(f"{b:>9}" for b in BAND_NAMES) + "   (share of the stem's energy)")
    for name, lv, fr in zip(names, loud, frames):
        e = (fr ** 2).sum(axis=0)
        print(f"{name:<10}{lv:7.1f}  " + "".join(f"{v / max(e.sum(), 1e-20) * 100:8.0f}%" for v in e))
    pairs = []
    for i, a_ in enumerate(frames):
        for j, b_ in enumerate(frames):
            if i == j or not a_.sum():
                continue
            per_band = np.minimum(a_, b_).sum(axis=0) / a_.sum()            # share of A covered by B, same moment, same band
            pairs.append((float(per_band.sum()), names[i], names[j], BAND_NAMES[int(np.argmax(per_band))]))
    print("\nmost covered: share of the first stem's sound that the second fills, same band, same moment")
    for share, x, y, band in sorted(pairs, reverse=True)[:8]:
        print(f"  {x:<8} by {y:<8} {share * 100:4.0f}%   mostly in {band}" + ("   <- move one of them" if share > .6 else ""))
    quiet = [f"{n} ({max(loud) - lv:.0f} dB under the loudest)" for n, lv in zip(names, loud) if max(loud) - lv > 35]
    if quiet:
        print("\nvery quiet next to the loudest stem: " + ", ".join(quiet))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("look")
    p.add_argument("file")
    p.add_argument("--sheet", help="write a PNG: waveform, spectrogram, band lanes, loudness")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_look)
    p = sub.add_parser("check")
    p.add_argument("file")
    p.add_argument("--lufs", type=float, default=-14.0)
    p.add_argument("--tol", type=float, default=1.5)
    p.add_argument("--peak", type=float, default=-1.0)
    p.add_argument("--seconds", type=float)
    p.add_argument("--seconds-tol", type=float, default=.5)
    p.add_argument("--bpm", type=float)
    p.add_argument("--key")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("mix")
    p.add_argument("stems", nargs="+")
    p.set_defaults(fn=cmd_mix)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
