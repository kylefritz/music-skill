#!/usr/bin/env python3
"""banger.py - make a house or techno track from a seed: search, listen, arrange, render, master.

    uv run --no-project --with numpy --with scipy python banger.py --seed 4242 --out /abs/dir
    ... --style deep-house --bpm 122 --key A --bars 24
    ... --seconds 30 --lufs -22 --mute lead        # a bed that ends with a 30 s film
    ... --stems                                     # also write one WAV per instrument

Writes track.wav (48 kHz stereo), track.mid, score.txt and score.json. The same seed and options
always give the same file. groove.py finds the notes; this file is the synth, the mix and the
listening step (it renders each instrument alone, measures where its energy sits, and hands that
to the second search pass).
"""
import argparse
import json
import os
import random
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import ndimage, signal

import audio
import groove
from audio import SR

TAIL = 2.5          # seconds of reverb ring-out after the last bar
DRUMS = ("kick", "clap", "hat", "ohat", "perc", "ride")
BUS = {**{d: "drums" for d in DRUMS}, "bass": "bass", "chords": "music", "lead": "music", "fx": "fx"}
TAU = 2 * np.pi


# ---------- building blocks ----------

def decay(t, seconds):
    """Exponential fall to -80 dB over `seconds`."""
    return np.exp(-9.21 * np.maximum(t, 0) / seconds)


def lowpass(x, hz):
    return signal.sosfilt(signal.butter(2, hz, "low", fs=SR, output="sos"), x, axis=0)


def highpass(x, hz):
    return signal.sosfilt(signal.butter(2, hz, "high", fs=SR, output="sos"), x, axis=0)


def bandpass(x, hz, q):
    return signal.lfilter(*signal.iirpeak(hz, q, fs=SR), x, axis=0)


def triangle(phase):
    return 2 / np.pi * np.arcsin(np.sin(phase))


def pan(mono, p):
    a = (p + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], axis=1)


def shelf(x, hz, gain_db, high):
    """RBJ shelving EQ, slope 1."""
    a_ = 10 ** (gain_db / 40)
    w = TAU * hz / SR
    c, al = np.cos(w), np.sin(w) / 2 * np.sqrt(2)
    s = -1 if high else 1
    k = 2 * np.sqrt(a_) * al
    b = [a_ * ((a_ + 1) - s * (a_ - 1) * c + k), s * 2 * a_ * ((a_ - 1) - s * (a_ + 1) * c),
         a_ * ((a_ + 1) - s * (a_ - 1) * c - k)]
    a = [(a_ + 1) + s * (a_ - 1) * c + k, -s * 2 * ((a_ - 1) + s * (a_ + 1) * c), (a_ + 1) + s * (a_ - 1) * c - k]
    return signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=0)


def additive(hz, t, wave, cutoff, q_db):
    """A saw, square or triangle built from its harmonics, each one scaled by a resonant low-pass
    whose cutoff can move over time. No aliasing, and no per-sample filter loop."""
    q = 10 ** (q_db / 20)
    top = int(min(18000, 6 * np.max(cutoff)) / hz)
    out = np.zeros(len(t))
    for k in range(1, top + 1):
        if wave == "saw":
            amp = 2 / np.pi / k * (1 if k % 2 else -1)
        elif k % 2 == 0:
            continue
        elif wave == "square":
            amp = 4 / np.pi / k
        else:
            amp = 8 / np.pi ** 2 / k ** 2 * (1 if k % 4 == 1 else -1)
        r = k * hz / cutoff
        out += amp * np.sin(TAU * k * hz * t) / np.sqrt((1 - r * r) ** 2 + (r / q) ** 2)
    return out


def sweep_filter(x, kind, f0, f1, blocks, state=None):
    """A filter whose cutoff glides from f0 to f1: short blocks, state carried across them."""
    out = np.empty_like(x)
    edges = np.linspace(0, len(x), blocks + 1).astype(int)
    for i in range(blocks):
        hz = f0 * (f1 / f0) ** ((i + .5) / blocks)
        sos = signal.butter(2, min(hz, SR * .45), kind, fs=SR, output="sos")
        if state is None:
            state = np.zeros((sos.shape[0], 2, x.shape[1]))
        out[edges[i]:edges[i + 1]], state = signal.sosfilt(sos, x[edges[i]:edges[i + 1]], axis=0, zi=state)
    return out, state


# ---------- the instruments ----------

class Kit:
    def __init__(self, g, seed):
        self.g = g
        self.rng = np.random.default_rng(seed)
        self.cache = {}
        self.noise = self.rng.uniform(-1, 1, SR * 4)
        t = np.arange(SR * 2) / SR
        # Six square waves at clashing ratios: the classic drum-machine cymbal.
        metal = sum(np.sign(np.sin(TAU * 40 * r * t)) for r in (2, 3, 4.16, 5.43, 6.79, 8.21)) * .17
        nz = self.noise[:len(t)]
        self.hat_src = highpass(bandpass(metal * 1.3, 10000, .7) + .4 * nz, g.hat_hp)
        self.ohat_src = highpass(bandpass(metal * 1.3, 8500, .7) + .4 * nz, g.hat_hp * .8)
        self.ride_src = bandpass(metal * 1.2 + .15 * nz, 6500, .6)

    def nz(self, n):
        if n > len(self.noise) - 1:
            return self.rng.uniform(-1, 1, n)
        o = int(self.rng.integers(0, len(self.noise) - n))
        return self.noise[o:o + n]

    def cached(self, key, make):
        if key not in self.cache:
            self.cache[key] = make()
        return self.cache[key]

    def kick(self, v):
        def make():
            g = self.g
            t = np.arange(int((g.kick_decay + .05) * SR)) / SR
            f = g.kick_hz * g.kick_sweep ** (1 - np.minimum(t / .07, 1))       # pitch drops to the home note
            body = np.sin(TAU * np.cumsum(f) / SR) * np.minimum(t / .003, 1) * decay(t, g.kick_decay) * v
            fp = g.kick_hz * 8 * .25 ** np.minimum(t / .028, 1)
            punch = triangle(TAU * np.cumsum(fp) / SR) * .45 * v * np.minimum(t / .002, 1) * decay(t, .07)
            click = highpass(self.noise[:len(t)], 3000) * .2 * v * decay(t, .012)
            return np.tanh(2.2 * (body + punch)) / np.tanh(2.2) + click
        return self.cached(("kick", round(v, 2)), make)

    def clap(self, v):
        g = self.g
        t = np.arange(int((.033 + g.clap_decay + .05) * SR)) / SR
        env = np.where(t >= .033, v * decay(t - .033, g.clap_decay), 0.0)
        for i in range(3):                                   # three quick hands before the body
            a = i * .011
            seg = (t >= a) & (t < a + .011)
            env[seg] = v * .2 ** np.minimum((t[seg] - a) / .009, 1)
        body = bandpass(self.nz(len(t)), g.clap_hz, 1.1) * env
        air = highpass(self.nz(len(t)), 4200) * np.where(t >= .02, .35 * v * decay(t - .02, .09), 0.0)
        return body, air

    def hat(self, v, is_open):
        d = self.g.ohat_decay if is_open else self.g.hat_decay
        n = int((d + .02) * SR)
        src = self.ohat_src if is_open else self.hat_src
        o = int(self.rng.integers(0, len(src) - n))
        t = np.arange(n) / SR
        return src[o:o + n] * np.minimum(t / .001, 1) * decay(t, d) * v

    def ride(self, v):
        n = int(.26 * SR)
        o = int(self.rng.integers(0, len(self.ride_src) - n))
        t = np.arange(n) / SR
        return self.ride_src[o:o + n] * np.minimum(t / .002, 1) * decay(t, .24) * v

    def perc(self, v):
        g = self.g
        if g.perc_type == "shaker":
            t = np.arange(int(.12 * SR)) / SR
            return bandpass(self.nz(len(t)), 5500, 2) * .3 * v * np.minimum(t / .012, 1) * decay(t - .012, .063)
        conga = g.perc_type == "conga"
        f0, d = groove.mtof(g.low + (31 if conga else 48)), (.13 if conga else .04)
        t = np.arange(int((d + .03) * SR)) / SR
        f = f0 * (1.25 if conga else 1) ** (1 - np.minimum(t / .03, 1))
        osc = np.sin(TAU * np.cumsum(f) / SR) if conga else triangle(TAU * np.cumsum(f) / SR)
        return osc * (.36 if conga else .26) * v * np.minimum(t / .002, 1) * decay(t, d)

    def bass(self, midi, dur, accent):
        def make():
            g = self.g
            hz = groove.mtof(midi)
            t = np.arange(int((dur + .15) * SR)) / SR
            open_ = 4.2 if accent else 3
            cut = g.bass_cut * open_ ** (1 - np.minimum(t / .14, 1))           # the filter closes on each note
            osc = additive(hz, t, g.bass_wave, cut, g.bass_q) + .45 * additive(hz * 2 ** (9 / 1200), t, g.bass_wave, cut, g.bass_q)
            env = .3 * np.minimum(t / .006, 1) * np.where(t < dur, 1, np.exp(-(t - dur) / .02))
            return np.tanh(1.6 * (osc + .6 * np.sin(TAU * hz * t)) * env) / np.tanh(1.6)
        return self.cached(("bass", midi, round(dur, 4), accent), make)

    def chord(self, notes, dur):
        def make():
            kind = self.g.chord_type
            v = .5 / np.sqrt(len(notes) * 3)
            if kind == "pad":
                t = np.arange(int((dur + .6) * SR)) / SR
                cut = 600 + 1300 * np.minimum(t / (dur * .6), 1)
                env = v * np.minimum(t / .08, 1) * np.where(t < dur * .96, 1, np.exp(-(t - dur * .96) / .1))
            else:
                d, start = (.22, 3400) if kind == "pluck" else (.34, 2600)
                t = np.arange(int((d + .05) * SR)) / SR
                cut = start * (480 / start) ** np.minimum(t / (d * .8), 1)
                env = 1.2 * v * np.minimum(t / .004, 1) * decay(t, d)
            out = np.zeros((len(t), 2))
            for cents, p in ((-10, -.55), (0, 0), (10, .55)):               # three detuned copies, spread wide
                out += pan(sum(additive(groove.mtof(m) * 2 ** (cents / 1200), t, "saw", cut, 1.2) for m in notes), p)
            return out * env[:, None]
        return self.cached(("chord", tuple(notes), round(dur, 4)), make)

    def lead(self, midi):
        def make():
            t = np.arange(int(.32 * SR)) / SR
            cut = 5200 * (900 / 5200) ** np.minimum(t / .13, 1)
            out = sum(pan(additive(groove.mtof(midi) * 2 ** (c / 1200), t, self.g.lead_wave, cut, 3), p)
                      for c, p in ((-7, -.3), (7, .3)))
            return out * (np.minimum(t / .004, 1) * decay(t, .28))[:, None]
        return self.cached(("lead", midi), make)

    def swept_noise(self, n, f0, f1, band):
        """Noise through a filter that glides: a bank of fixed bands with moving gains."""
        nz = self.rng.uniform(-1, 1, n)
        centers = np.geomspace(150, 14000, 14)
        f = f0 * (f1 / f0) ** (np.arange(n) / n)
        out = np.zeros(n)
        for c in centers:
            part = signal.sosfilt(signal.butter(2, [c / 1.18, c * 1.18], "band", fs=SR, output="sos"), nz)
            w = np.exp(-.5 * (np.log2(c / f) / .45) ** 2) if band else 1 / (1 + (c / f) ** 4)
            out += part * w
        return out

    def crash(self, v):
        t = np.arange(int(1.7 * SR)) / SR
        return highpass(self.nz(len(t)), 4500) * v * decay(t, 1.5)

    def riser(self, dur, v):
        g = self.g
        n = int((dur + .05) * SR)
        t = np.arange(n) / SR
        env = np.where(t < dur, t / dur, np.maximum(0, 1 - (t - dur) / .05))
        f = groove.mtof(g.low + 36) * 4 ** np.minimum(t / dur, 1)             # a tone that climbs two octaves
        return self.swept_noise(n, 300, 9000, True) * v * 1.6 * env, np.sin(TAU * np.cumsum(f) / SR) * v * .3 * env

    def downlift(self, dur):
        n = int(dur * SR)
        return self.swept_noise(n, 9000, 200, False) * .16 * decay(np.arange(n) / SR, dur) * .35

    def impact(self):
        t = np.arange(int(1.4 * SR)) / SR
        f = self.g.kick_hz * 3 * .25 ** np.minimum(t / .5, 1)
        return np.sin(TAU * np.cumsum(f) / SR) * .5 * np.minimum(t / .005, 1) * decay(t, 1.3)


# ---------- placing notes on a timeline ----------

class Tape:
    def __init__(self, seconds, stems):
        self.n = int(seconds * SR)
        self.bus = {k: np.zeros((self.n, 2), np.float32) for k in ("drums", "bass", "music", "fx", "rev", "dly")}
        self.stems = {} if stems else None

    def _add(self, buf, t, sig):
        i = int(round(t * SR))
        if i < 0:
            sig, i = sig[-i:], 0
        m = min(len(sig), self.n - i)
        if m > 0:
            buf[i:i + m] += sig[:m]

    def add(self, layer, t, sig, p=0.0, rev=0.0, dly=0.0):
        if sig.ndim == 1:
            sig = pan(sig, p)
        self._add(self.bus[BUS[layer]], t, sig)
        if self.stems is not None:
            if layer not in self.stems:
                self.stems[layer] = np.zeros((self.n, 2), np.float32)
            self._add(self.stems[layer], t, sig)
        if rev:
            self._add(self.bus["rev"], t, sig * rev)
        if dly:
            self._add(self.bus["dly"], t, sig * dly)


def place(g, pat, plan, kit, stems=False, tail=TAIL):
    """Turn bars into sound on the buses. plan entries of None mean the bare loop."""
    step = 60 / g.bpm / 4
    bar = 16 * step
    tape = Tape(len(plan) * bar + tail, stems)
    kicks = []
    chords = pat.get("chords") or dict(roots=[0] * 4, inv=[0] * 4)
    for k, pb in enumerate(plan):
        t0 = k * bar
        ev = groove.bar_events(pat, g, pb, k)
        if pb:
            if pb["first"] and pb["drop"]:
                tape.add("fx", t0, kit.impact())
                tape.add("fx", t0, kit.crash(.2), rev=.3)
            elif pb["first"] and pb["section"] == "Break":
                tape.add("fx", t0, kit.downlift(bar * 2), rev=.3)
                tape.add("fx", t0, kit.crash(.12), rev=.3)
            elif pb["first"] and pb["section"] in ("Groove", "Outro"):
                tape.add("fx", t0, kit.crash(.12), rev=.3)
            if pb["build"] and pb["i"] == pb["len"] - min(4, pb["len"]):
                noise, tone = kit.riser(bar * min(4, pb["len"]) - step * 4, .2)
                tape.add("fx", t0, noise)
                tape.add("fx", t0, tone, rev=.4)
            if pb["pre"]:
                noise, tone = kit.riser(bar, .12)
                tape.add("fx", t0, noise)
                tape.add("fx", t0, tone, rev=.4)
        v = ev["vel"]
        for s in range(ev["gap_from"]):
            t = t0 + s * step + (g.swing * step if s % 2 else 0)
            if v.get("kick") and v["kick"][s]:
                tape.add("kick", t, kit.kick(v["kick"][s]))
                kicks.append(t)
            if v.get("clap") and v["clap"][s]:
                body, air = kit.clap(v["clap"][s] * .58)
                tape.add("clap", t + g.feel["clap"], body, rev=.7 if ev["wet_clap"] and s >= 12 else .24)
                tape.add("clap", t + g.feel["clap"], air, p=.3)
            if v.get("ohat") and v["ohat"][s]:
                tape.add("ohat", t + g.feel["hat"], kit.hat(.2, True), p=.2)
            if v.get("hat") and v["hat"][s]:
                tape.add("hat", t + g.feel["hat"], kit.hat(.26 * v["hat"][s] * kit.rng.uniform(.92, 1.08), False),
                         p=.22 if s % 4 == 2 else -.14)
            if v.get("perc") and v["perc"][s]:
                tape.add("perc", t + g.feel["perc"], kit.perc(v["perc"][s] * kit.rng.uniform(.9, 1)), p=g.perc_pan, rev=.16)
            if ev["ride"] and ev["ride"][s]:
                tape.add("ride", t + g.feel["hat"], kit.ride(.1 * ev["ride"][s]), p=.3)
            root, inv = chords["roots"][ev["chord"]], chords["inv"][ev["chord"]]
            if ev["bass"] and ev["bass"][s] >= 0:
                n = 1
                while n < 4 and ev["bass"][(s + n) % 16] < 0:
                    n += 1
                dur = min(n, g.bass_gate) * step * .92
                tape.add("bass", t + g.feel["bass"], kit.bass(groove.bass_midi(root, ev["bass"][s], g), dur, ev["bass"][s] == 2))
            if ev["chord_hit"] and ev["chord_hit"][s]:
                n = 1
                while s + n < ev["gap_from"] and not ev["chord_hit"][s + n]:
                    n += 1
                tape.add("chords", t, kit.chord(groove.voicing(root, inv, g), n * step), rev=.3,
                         dly=0 if g.chord_type == "pad" else .22)
            if ev["lead"] and ev["lead"][s] >= 0:
                m = g.lead_base + groove.semi(ev["lead"][s], g)
                tape.add("lead", t, kit.lead(m) * .25, rev=.22, dly=.38)
                if ev["octave"]:
                    tape.add("lead", t, kit.lead(m + 12) * .085, rev=.22, dly=.38)
    return tape, kicks


def measure(g, pat, seed):
    """The listening step: play four bars of each instrument alone and measure its spectrum."""
    tape, _ = place(g, pat, [None] * 4, Kit(g, seed), stems=True, tail=.5)
    return {k: audio.band_profile(s.mean(axis=1)) for k, s in tape.stems.items() if k in groove.LAYERS}


# ---------- mixing and mastering ----------

def section_sweeps(tape, g, plan):
    """Low-pass the chords and lead through the breakdown and build; high-pass the drums in the build."""
    bar = int(16 * 60 / g.bpm / 4 * SR)
    for bus, key, kind in (("music", "lp", "low"), ("drums", "hp", "high")):
        state = None
        for k, pb in enumerate(plan):
            if not pb or not pb[key]:
                state = None
                continue
            seg = tape.bus[bus][k * bar:(k + 1) * bar]
            seg[:], state = sweep_filter(seg.astype(np.float64), kind, pb[key][0], pb[key][1], 16, state)


def duck_curve(n, kicks, depth):
    """Sidechain: each kick pushes the bass, chords and effects down, then lets them back up."""
    t = np.arange(int(.5 * SR)) / SR
    shape = np.where(t < .06, depth + (1 - depth) * np.exp(-t / .004), 1 - (1 - depth) * np.exp(-(t - .06) / .07))
    env = np.ones(n)
    for k in kicks:
        i = int(k * SR)
        m = min(len(shape), n - i)
        env[i:i + m] = np.minimum(env[i:i + m], shape[:m])
    return env[:, None]


def reverb(send, seed):
    rng = np.random.default_rng(seed + 1)
    n = int(2.8 * SR)
    ir = rng.uniform(-1, 1, (n, 2)) * ((1 - np.arange(n) / n) ** 2.4)[:, None]
    ir *= 10 ** (-58 / 20) / np.sqrt(np.mean(ir ** 2)) * 44100 / SR
    wet = highpass(send.astype(np.float64), 250)
    wet = np.concatenate([np.zeros((int(.02 * SR), 2)), wet])[:len(send)]       # 20 ms pre-delay
    return np.stack([signal.fftconvolve(wet[:, c], ir[:, c])[:len(send)] for c in (0, 1)], axis=1) * .5


def ping_pong(send, step):
    """Echoes every three 16ths, bouncing left and right, each one darker and quieter."""
    echo = send.astype(np.float64).mean(axis=1)
    out = np.zeros((len(send), 2))
    d = int(3 * step * SR)
    for i in range(1, 7):
        echo = lowpass(echo, 3400)
        if i * d >= len(echo):
            break
        out[i * d:] += pan(echo[:len(echo) - i * d], -.75 if i % 2 else .75) * .4 ** (i - 1)
    return out * .5


def compress(x, threshold=-16, ratio=3, knee=10, attack=.012, release=.2):
    """Glue compressor. The level detector runs on every 32nd sample to keep the Python loop short."""
    hop = 32
    peak = np.abs(x).max(axis=1)
    lv = np.pad(peak, (0, -len(peak) % hop)).reshape(-1, hop).max(axis=1)
    ca, cr = np.exp(-hop / (attack * SR)), np.exp(-hop / (release * SR))
    env, e = np.empty_like(lv), 0.0
    for i, v in enumerate(lv):
        c = ca if v > e else cr
        e = c * e + (1 - c) * v
        env[i] = e
    over = audio.db(env) - threshold
    cut = np.where(over <= -knee / 2, 0, np.where(over >= knee / 2, over, (over + knee / 2) ** 2 / (2 * knee))) * (1 - 1 / ratio)
    gain = np.interp(np.arange(len(x)), np.arange(len(lv)) * hop + hop / 2, 10 ** (-cut / 20))
    return x * gain[:, None]


def limit(x, ceiling_db):
    """Look-ahead peak limiter: the gain is down before each peak arrives and never overshoots."""
    w = int(.02 * SR)
    need = np.minimum(1, 10 ** (ceiling_db / 20) / np.maximum(np.abs(x).max(axis=1), 1e-9))
    gain = ndimage.uniform_filter1d(ndimage.minimum_filter1d(need, 2 * w + 1), w)
    return x * gain[:, None], float(audio.db(gain.min()))


def master(mix, target_lufs, ceiling_db):
    x = highpass(mix * .8, 28)
    x = shelf(shelf(x, 90, 1.5, False), 9000, 2, True)
    x = compress(x)
    x = np.tanh(1.3 * x) / np.tanh(1.3)                                      # soft clip
    limited = 0.0
    for _ in range(3):                                                       # limiting costs loudness; go again
        gain = target_lufs - audio.lufs(x)
        y, limited = limit(x * 10 ** (gain / 20), ceiling_db - .3)
        if abs(audio.lufs(y) - target_lufs) < .1:
            break
        x = y
    over = audio.true_peak(y) - ceiling_db
    if over > 0:
        y *= 10 ** (-over / 20)
    return y, limited


def mixdown(tape, kicks, g, plan, seed, target_lufs, ceiling_db):
    section_sweeps(tape, g, plan)
    duck = duck_curve(tape.n, kicks, g.duck)
    b = tape.bus
    wet = reverb(b["rev"], seed) + ping_pong(b["dly"], 60 / g.bpm / 4)
    mix = b["drums"].astype(np.float64) + b["fx"] + (b["bass"] + b["music"] + wet) * duck
    return master(mix, target_lufs, ceiling_db)


# ---------- MIDI ----------

def write_midi(path, g, pat, plan):
    """The arrangement as a type-1 MIDI file, one track per instrument, for a DAW."""
    ppq, st = 480, 120
    drum_note = dict(kick=36, clap=39, hat=42, ohat=46, perc=dict(conga=63, shaker=70, rim=37)[g.perc_type], ride=51)
    tracks = {k: [] for k in groove.LAYERS + ["ride"]}

    def note(track, ch, tick, dur, pitch, vel):
        tracks[track] += [(tick, 1, bytes([0x90 | ch, pitch, max(1, min(127, int(vel * 127)))])),
                          (tick + dur, 0, bytes([0x80 | ch, pitch, 0]))]

    chords = pat["chords"]
    for k, pb in enumerate(plan):
        ev = groove.bar_events(pat, g, pb, k)
        root, inv = chords["roots"][ev["chord"]], chords["inv"][ev["chord"]]
        for s in range(ev["gap_from"]):
            tick = (k * 16 + s) * st + (int(g.swing * st) if s % 2 else 0)
            for d in DRUMS:
                vel = ev["ride"][s] if d == "ride" and ev["ride"] else ev["vel"].get(d, [0] * 16)[s] if d != "ride" else 0
                if vel:
                    note(d, 9, tick, st // 2, drum_note[d], vel)
            if ev["bass"] and ev["bass"][s] >= 0:
                note("bass", 1, tick, st, groove.bass_midi(root, ev["bass"][s], g), .8)
            if ev["chord_hit"] and ev["chord_hit"][s]:
                n = 1
                while s + n < ev["gap_from"] and not ev["chord_hit"][s + n]:
                    n += 1
                for m in groove.voicing(root, inv, g):
                    note("chords", 2, tick, n * st - 10, m, .7)
            if ev["lead"] and ev["lead"][s] >= 0:
                note("lead", 3, tick, st, g.lead_base + groove.semi(ev["lead"][s], g) + (12 if ev["octave"] else 0), .8)

    def vlq(n):
        out = [n & 0x7F]
        while n > 0x7F:
            n >>= 7
            out.insert(0, (n & 0x7F) | 0x80)
        return bytes(out)

    def chunk(name, events):
        data, last = b"\x00\xff\x03" + vlq(len(name)) + name.encode(), 0
        for tick, _, msg in sorted(events):
            data += vlq(tick - last) + msg
            last = tick
        data += b"\x00\xff\x2f\x00"
        return b"MTrk" + struct.pack(">I", len(data)) + data

    tempo = [(0, 0, b"\xff\x51\x03" + struct.pack(">I", round(60e6 / g.bpm))[1:])]
    body = chunk(f"{g.style} {g.key_name}", tempo) + b"".join(chunk(k, v) for k, v in tracks.items() if v)
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 1, 1 + sum(1 for v in tracks.values() if v), ppq) + body)


# ---------- command line ----------

def fit_to_seconds(g, seconds, bpm_fixed):
    """The longest arrangement that fits, nudging the tempo (within the style) so the last bar
    ends TAIL seconds before the target and the reverb rings out to it."""
    lo, hi = g.bpm_range[0] - 4, g.bpm_range[1] + 4
    play = max(seconds - TAIL, 1)
    for bars in range(int(play * hi / 240) // 2 * 2, 6, -2):
        exact = bars * 240 / play
        if not bpm_fixed and lo <= exact <= hi:
            return bars, exact
        if bars * 240 / g.bpm <= play:
            return bars, g.bpm
    return 8, g.bpm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, help="same seed, same track (default: random)")
    ap.add_argument("--style", choices=sorted(groove.STYLES))
    ap.add_argument("--bpm", type=float)
    ap.add_argument("--key", choices=groove.NOTE, help="home note")
    ap.add_argument("--mode", choices=sorted(groove.MODES))
    ap.add_argument("--bars", type=int, help="any even number from 8: 8 is a loop, 16 intro-drop-outro, 72 the full form (default)")
    ap.add_argument("--seconds", type=float, help="fit a total length: picks the bars and nudges the tempo")
    ap.add_argument("--lufs", type=float, default=-14.0, help="integrated loudness (default -14; a bed under a voice: -22)")
    ap.add_argument("--peak", type=float, default=-1.0, help="true-peak ceiling in dBTP (default -1)")
    ap.add_argument("--mute", default="", help="layers to leave out of the arrangement, e.g. lead,clap")
    ap.add_argument("--stems", action="store_true", help="also write stems/<layer>.wav")
    ap.add_argument("--out", help="output folder (default ./banger-<seed>)")
    a = ap.parse_args()

    began = time.time()
    seed = a.seed if a.seed is not None else random.randrange(1000, 10000)
    key = groove.NOTE.index(a.key) if a.key else None
    g = groove.make_genome(seed, a.style, a.bpm, key, a.mode)
    bars = a.bars or 72
    if a.seconds:
        bars, bpm = fit_to_seconds(g, a.seconds, bool(a.bpm))
        bars = min(bars, a.bars) if a.bars else bars
        if not a.bpm:
            g = groove.make_genome(seed, a.style, bpm, key, a.mode)

    g, pat, report = groove.compose(seed, a.style, g.bpm, key, a.mode, measure=lambda g, p: measure(g, p, seed))
    plan = groove.make_plan(g, bars)
    muted = {m for m in a.mute.split(",") if m}
    for pb in plan:
        pb["on"] = [x for x in pb["on"] if x not in muted]

    tape, kicks = place(g, pat, plan, Kit(g, seed), stems=a.stems)
    track, limited = mixdown(tape, kicks, g, plan, seed, a.lufs, a.peak)

    out = os.path.abspath(a.out or f"banger-{seed}")
    os.makedirs(out, exist_ok=True)
    audio.write_wav(os.path.join(out, "track.wav"), track, seed)
    if a.stems:
        os.makedirs(os.path.join(out, "stems"), exist_ok=True)
        for k, s in tape.stems.items():
            audio.write_wav(os.path.join(out, "stems", f"{k}.wav"), s.astype(np.float64), seed)
    write_midi(os.path.join(out, "track.mid"), g, pat, plan)

    sections, last = [], None
    for i, pb in enumerate(plan):
        if pb["section"] != last:
            sections.append(f"{pb['section']} {i * 240 / g.bpm:.0f}s")
            last = pb["section"]
    text = "\n".join([
        groove.describe(g, pat, report), "",
        f"arrangement: {bars} bars  " + "  ".join(sections),
        f"track.wav  {len(track) / SR:.1f}s  {audio.lufs(track):.1f} LUFS  true peak {audio.true_peak(track):.1f} dBTP  "
        f"limiter took off up to {abs(limited):.1f} dB" + (f"  muted: {', '.join(sorted(muted))}" if muted else ""),
        f"made in {time.time() - began:.0f}s: banger.py " + " ".join(sys.argv[1:])])
    with open(os.path.join(out, "score.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out, "score.json"), "w") as f:
        json.dump(groove.to_json(g, pat, report, plan), f, indent=1)
    print(text)
    print(out)


if __name__ == "__main__":
    main()
