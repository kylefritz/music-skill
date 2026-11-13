#!/usr/bin/env python3
"""groove.py - write a house or techno track as data, with no audio and no dependencies.

A seed rolls a style, tempo, key and a set of targets. Then each instrument is found by search:
start from a random 16-step pattern, try every one-step change, score each candidate against a
short list of rules, keep the best, stop when nothing scores higher. Later instruments are scored
against the ones already found. A second pass re-tunes five layers against the measured spectrum
of every instrument, so they stay out of each other's way. The patterns are then laid out as a
track with an intro, drops, a breakdown and a build.

The method and the choice of rules follow Banger Lab (https://claude.ai/artifact/N6bSgS4ZH5ifFQfg8sm4dv).
Every rule cites the study it comes from in SOURCES. The weights are a simplification, not fitted
to data.

    python3 groove.py --seed 4242            # print the score as text
    python3 groove.py --seed 4242 --json     # the same as JSON
"""
import argparse
import json
import random
from types import SimpleNamespace

LAYERS = ["kick", "clap", "hat", "ohat", "perc", "bass", "chords", "lead"]
SECOND_PASS = ["hat", "perc", "bass", "chords", "lead"]
# How strong each 16th-note step of a 4/4 bar feels (Lerdahl & Jackendoff's metrical grid).
GRID = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]
LEVEL = [0.0, 0.35, 0.7, 1.0]  # hit level 0..3 -> loudness
BANDS = [("sub", 20, 80), ("bass", 80, 250), ("low mid", 250, 800),
         ("mid", 800, 2500), ("high", 2500, 7000), ("air", 7000, 20000)]
# Rough band profile per instrument, used only until the real audio has been measured.
PRIOR_PROFILE = {
    "kick": [1, .5, .15, .05, .03, .01], "clap": [0, .01, .15, .5, .3, .1],
    "hat": [0, 0, 0, .01, .06, .2], "ohat": [0, 0, 0, .01, .08, .2],
    "perc": [0, .03, .25, .15, .06, .02], "bass": [.5, .6, .3, .1, .02, 0],
    "chords": [0, .08, .35, .3, .12, .02], "lead": [0, .02, .15, .3, .15, .03],
}
NOTE = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MODES = {"minor": [0, 2, 3, 5, 7, 8, 10], "dorian": [0, 2, 3, 5, 7, 9, 10],
         "phrygian": [0, 1, 3, 5, 7, 8, 10]}
DIMINISHED = {"minor": 1, "dorian": 5, "phrygian": 4}  # scale degree whose triad is diminished
BASS_STEPS = [0, 4, 7, 2]  # what a bass note can be, in scale steps: root, fifth, octave, third
BASS_CHAR = "R5O3"

SOURCES = {
    "butler": "Butler 2006, Unlocking the Groove: how house and techno layer loops and build sections",
    "witek": "Witek et al. 2014, PLOS ONE 9(4): medium syncopation gives the strongest wish to move",
    "toussaint": "Toussaint 2005, The Euclidean Algorithm Generates Traditional Musical Rhythms",
    "krumhansl": "Krumhansl & Kessler 1982, Psychological Review 89(4): the home note and chord are heard as most stable",
    "tymoczko": "Tymoczko 2011, A Geometry of Music: good chord changes move each voice a short distance",
    "huron": "Huron 2001, Music Perception 19(1): small pitch steps keep a melody heard as one line",
    "plomp": "Plomp & Levelt 1965, JASA 38: sounds sharing a narrow frequency band interfere",
    "caplin": "Caplin 1998, Classical Form: the question-and-answer phrase",
    "davies": "Davies et al. 2013, Music Perception 30(5): larger timing shifts lowered groove ratings",
    "solberg": "Solberg & Dibben 2019, Music Perception: breakdown, build-up and drop",
    "lerdahl": "Lerdahl & Jackendoff 1983, A Generative Theory of Tonal Music: the strong/weak beat grid",
    "lhl": "Longuet-Higgins & Lee 1984, Music Perception 1(4): the syncopation count",
}

STYLES = {
    "deep-house": dict(name="Deep house", bpm=(120, 124), swing=(.08, .16), modes=["dorian", "minor"],
        ext=(4, 5), variety=(2, 4), kick_extra=(0, 0), clap_extra=(0, 1), hats=(6, 10), open_hats=(2, 4),
        percs=(3, 5), perc_sync=(3, 6), bass_notes=(4, 6), bass_sync=(5, 8), chord=["stab", "pad"],
        chord_hits=(2, 4), chord_sync=(4, 8), lead_notes=(4, 7), lead_sync=(3, 7),
        bass_wave=["square", "triangle"], lead_wave=["triangle", "square"], kick_decay=(.30, .38),
        perc=["conga", "shaker", "rim"], bass_cut=(500, 900), bass_q=(1, 3), accent=["off"]),
    "tech-house": dict(name="Tech house", bpm=(124, 128), swing=(.05, .12), modes=["minor", "dorian", "phrygian"],
        ext=(3, 4), variety=(1, 2), kick_extra=(0, 1), clap_extra=(0, 2), hats=(8, 12), open_hats=(2, 4),
        percs=(5, 7), perc_sync=(4, 8), bass_notes=(6, 9), bass_sync=(5, 9), chord=["stab"],
        chord_hits=(2, 3), chord_sync=(4, 8), lead_notes=(3, 6), lead_sync=(4, 8),
        bass_wave=["saw", "square"], lead_wave=["square", "saw"], kick_decay=(.26, .34),
        perc=["conga", "rim", "shaker"], bass_cut=(600, 1200), bass_q=(2, 6), accent=["off", "on"]),
    "peak-techno": dict(name="Peak-time techno", bpm=(130, 136), swing=(0, .03), modes=["phrygian", "minor"],
        ext=(3, 3), variety=(1, 2), kick_extra=(0, 2), clap_extra=(0, 1), hats=(4, 8), open_hats=(4, 4),
        percs=(3, 6), perc_sync=(3, 7), bass_notes=(10, 12), bass_sync=(9, 11), chord=["stab", "pad"],
        chord_hits=(1, 3), chord_sync=(3, 7), lead_notes=(5, 9), lead_sync=(4, 8),
        bass_wave=["saw"], lead_wave=["saw", "square"], kick_decay=(.24, .32),
        perc=["rim", "shaker", "conga"], bass_cut=(350, 700), bass_q=(1, 4), accent=["off", "on"]),
    "melodic-techno": dict(name="Melodic techno", bpm=(122, 126), swing=(0, .06), modes=["minor", "dorian"],
        ext=(4, 4), variety=(3, 4), kick_extra=(0, 1), clap_extra=(0, 1), hats=(6, 10), open_hats=(2, 4),
        percs=(3, 5), perc_sync=(3, 6), bass_notes=(6, 10), bass_sync=(6, 10), chord=["pad", "pluck"],
        chord_hits=(1, 3), chord_sync=(2, 6), lead_notes=(8, 12), lead_sync=(3, 7),
        bass_wave=["saw", "square"], lead_wave=["saw", "triangle"], kick_decay=(.28, .36),
        perc=["shaker", "rim", "conga"], bass_cut=(500, 1000), bass_q=(1, 4), accent=["off"]),
}


# ---------- small helpers ----------

def clamp(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def near(x, target, width):
    """1 at the target, 0.5 at half the width away, and never flat, so the search always has a slope."""
    d = (x - target) / (width / 2)
    return 1 / (1 + d * d)


def count(bits):
    return sum(1 for b in bits if b)


def bits_of(notes):
    """A pitched line (-1 = rest) as hits."""
    return [1 if v >= 0 else 0 for v in notes]


def syncopation(bits):
    """A simple Longuet-Higgins & Lee count: a note on a weak step, held over a stronger one."""
    total = 0
    for i in range(16):
        if not bits[i]:
            continue
        j, strongest = (i + 1) % 16, 0
        while not bits[j] and j != i:
            strongest = max(strongest, GRID[j])
            j = (j + 1) % 16
        if strongest > GRID[i]:
            total += strongest - GRID[i]
    return total


def evenness(bits):
    """How close the gaps between hits are to equal (Toussaint's Euclidean rhythms score 1)."""
    pos = [i for i in range(16) if bits[i]]
    k = len(pos)
    if k < 2:
        return float(k)
    ideal = 16 / k
    off = sum(abs(((pos[(i + 1) % k] - pos[i]) % 16 or 16) - ideal) for i in range(k))
    return clamp(1 - off / k / ideal)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# ---------- the run: style, tempo, key, targets ----------

def make_genome(seed, style=None, bpm=None, key=None, mode=None):
    r = random.Random(f"genome/{seed}")
    st = STYLES[style or r.choice(list(STYLES))]
    R = lambda k: r.randint(*st[k])
    F = lambda k: r.uniform(*st[k])
    g = SimpleNamespace(seed=seed, style=st["name"], bpm_range=st["bpm"])
    g.mode = r.choice(st["modes"])
    g.pc = r.randrange(12)
    g.bpm = R("bpm")
    g.swing = F("swing")
    g.ext = R("ext")                    # 3 = triads, 4 = sevenths, 5 = ninths
    g.variety = R("variety")            # different chords in the 4-bar loop
    g.kick_extra, g.clap_extra = R("kick_extra"), R("clap_extra")
    g.hats, g.open_hats = R("hats"), R("open_hats")
    g.percs, g.perc_sync = R("percs"), R("perc_sync")
    g.bass_notes, g.bass_sync = R("bass_notes"), R("bass_sync")
    g.chord_type = r.choice(st["chord"])
    g.chord_hits, g.chord_sync = R("chord_hits"), R("chord_sync")
    g.lead_notes, g.lead_sync = R("lead_notes"), R("lead_sync")
    g.bass_root_share = r.uniform(.6, .9)
    g.taste = [r.uniform(.4, 1) for _ in range(7)]          # how much this run likes each chord
    g.move = [w * r.uniform(.7, 1) for w in (.4, .8, .7, 1, .85, .85, .9)]  # root motion by interval
    g.bass_wave, g.lead_wave = r.choice(st["bass_wave"]), r.choice(st["lead_wave"])
    g.kick_decay, g.kick_sweep = F("kick_decay"), r.uniform(3, 4.5)
    g.perc_type = r.choice(st["perc"])
    g.bass_cut, g.bass_q = F("bass_cut"), F("bass_q")
    g.hat_hp, g.hat_decay, g.ohat_decay = r.uniform(6500, 9500), r.uniform(.03, .055), r.uniform(.16, .3)
    g.clap_hz, g.clap_decay = r.uniform(1100, 1900), r.uniform(.11, .2)
    g.duck = r.uniform(.25, .5)         # how far the kick pushes everything else down
    g.hat_accent = r.choice(st["accent"])
    g.perc_ghost = r.uniform(.25, .5)
    g.crowd = 4
    g.late_lead = r.random() < .5
    g.fill_kind = r.randint(0, 1)
    g.perc_pan = r.choice([-1, 1]) * r.uniform(.2, .45)
    # Microtiming in seconds. All under 12 ms: larger shifts lowered groove ratings (Davies 2013).
    g.feel = dict(hat=r.uniform(.004, .012), clap=r.uniform(-.008, -.002),
                  perc=r.uniform(0, .01), bass=r.uniform(0, .006))
    gate = r.uniform(1, 2)

    if mode:
        g.mode = mode
    if key is not None:
        g.pc = key
    if bpm:
        g.bpm = bpm
    g.scale = MODES[g.mode]
    g.dim = DIMINISHED[g.mode]
    if g.mode == "phrygian":
        g.ext = min(4, g.ext)
    if g.chord_type == "pad":
        g.chord_hits, g.chord_sync = 1, 0
    g.taste[0], g.taste[g.dim] = 1, 0
    g.allowed = [d for d in range(7) if d != g.dim]
    g.low = 28 + (g.pc - 4) % 12        # the home note between E1 and Eb2
    g.kick_hz = mtof(g.low)             # the kick is tuned to the home note
    g.chord_tonic = 48 + g.pc
    g.lead_base = 60 + g.pc if g.pc <= 5 else 48 + g.pc
    g.bass_gate = .9 if g.bass_notes >= 10 else gate
    g.key_name = f"{NOTE[g.pc]} {g.mode}"
    return g


# ---------- harmony ----------

def semi(deg, g):
    return g.scale[deg % 7] + 12 * (deg // 7)


def chord_degrees(root, g):
    d = [root, root + 2, root + 4]
    if g.ext >= 4:
        d.append(root + 6)
    if g.ext >= 5 and semi(root + 8, g) - semi(root, g) == 14:
        d.append(root + 8)
    return d


def voicing(root, inv, g):
    """MIDI notes for a chord, inverted `inv` times and moved by octaves to sit around D4."""
    n = [g.chord_tonic + semi(d, g) for d in chord_degrees(root, g)]
    for k in range(inv):
        n[k % len(n)] += 12
    shift = round((62 - sum(n) / len(n)) / 12) * 12
    return sorted(x + shift for x in n)


def voice_distance(a, b):
    """Mean distance in semitones from each note to the nearest note of the other chord."""
    s = sum(min(abs(x - y) for y in b) for x in a) + sum(min(abs(x - y) for x in a) for y in b)
    return s / (len(a) + len(b))


def bass_midi(root, idx, g):
    m = g.low + g.scale[root]
    if m >= 40:
        m -= 12
    return m + semi(root + BASS_STEPS[idx], g) - semi(root, g)


def chord_name(root, g):
    third = semi(root + 2, g) - semi(root, g)
    fifth = semi(root + 4, g) - semi(root, g)
    seventh = semi(root + 6, g) - semi(root, g)
    minor, n = third == 3, len(chord_degrees(root, g))
    numeral = ["I", "II", "III", "IV", "V", "VI", "VII"][root]
    roman = ("b" if g.scale[root] < MAJOR[root] else "") + (numeral.lower() if minor else numeral)
    roman += "o" if fifth == 6 else ""
    name = NOTE[(g.pc + g.scale[root]) % 12] + ("m" if minor else "")
    if n == 4:
        name += "maj7" if seventh == 11 else "7"
        roman += "7"
    elif n >= 5:
        name += "maj9" if seventh == 11 else "9"
        roman += "9"
    return roman, name


# ---------- pattern kinds: how to start one and what its one-step changes are ----------

def random_hits(r, n, value=lambda r: 1):
    p = [0] * 16
    for i in r.sample(range(16), n):
        p[i] = value(r)
    return p


def step_changes(p, values, empty):
    """Every pattern one edit away: set one step to another value, or slide a hit one step."""
    out = []
    for i in range(len(p)):
        for v in values:
            if v != p[i]:
                q = p[:]
                q[i] = v
                out.append(q)
        if p[i] != empty:
            for d in (1, len(p) - 1):
                j = (i + d) % len(p)
                if p[j] == empty:
                    q = p[:]
                    q[j], q[i] = p[i], empty
                    out.append(q)
    return out


def binary_init(lo, hi):
    return lambda r, ctx: random_hits(r, r.randint(lo, hi))


def level_init(lo, hi):
    return lambda r, ctx: random_hits(r, r.randint(lo, hi), lambda r: r.randint(1, 3))


def binary_moves(p, ctx):
    return step_changes(p, (0, 1), 0)


def level_moves(p, ctx):
    return step_changes(p, (0, 1, 2, 3), 0)


def bass_init(r, ctx):
    p = [-1] * 16
    for i in r.sample(range(16), r.randint(3, 8)):
        p[i] = 0 if r.random() < .6 else r.randint(1, 3)
    return p


def bass_moves(p, ctx):
    return step_changes(p, (-1, 0, 1, 2, 3), -1)


def chords_init(r, ctx):
    return dict(roots=[r.choice(ctx.g.allowed) for _ in range(4)],
                inv=[r.randint(0, 2) for _ in range(4)],
                hit=random_hits(r, r.randint(1, 4)))


def chords_moves(p, ctx):
    roots, inv, hit = p["roots"], p["inv"], p["hit"]
    out = [dict(roots=roots, inv=inv, hit=h) for h in binary_moves(hit, ctx)]
    for i in range(4):
        for d in ctx.g.allowed:
            if d != roots[i]:
                out.append(dict(roots=roots[:i] + [d] + roots[i + 1:], inv=inv, hit=hit))
        for v in range(3):
            if v != inv[i]:
                out.append(dict(roots=roots, inv=inv[:i] + [v] + inv[i + 1:], hit=hit))
        if i and roots[i] != roots[i - 1]:      # hold the previous bar's chord
            out.append(dict(roots=roots[:i] + [roots[i - 1]] + roots[i + 1:],
                            inv=inv[:i] + [inv[i - 1]] + inv[i + 1:], hit=hit))
    for x in set(roots):                        # swap one chord for another everywhere
        for y in ctx.g.allowed:
            if y != x:
                out.append(dict(roots=[y if v == x else v for v in roots], inv=inv, hit=hit))
    for k in range(1, 4):                       # rotate the progression
        out.append(dict(roots=roots[k:] + roots[:k], inv=inv[k:] + inv[:k], hit=hit))
    return out


def lead_init(r, ctx):
    p = [-1] * 32
    d = r.choice([0, 2, 4, 7])
    for _ in range(r.randint(6, 14)):
        d = max(0, min(9, d + r.randint(-2, 2)))
        p[r.randrange(32)] = d
    return p


def lead_moves(p, ctx):
    out = []
    for i in range(32):
        for v in range(-1, 10):
            if v != p[i]:
                q = p[:]
                q[i] = v
                out.append(q)
    for j in range(16):                         # make the two bars agree at one step
        if p[j] != p[j + 16]:
            a, b = p[:], p[:]
            a[j + 16], b[j] = p[j], p[j + 16]
            out += [a, b]
    return out


# ---------- rules ----------
# A rule is (name, weight, source, fn). fn(pattern, ctx) returns 0..1. ctx.g is the genome and
# ctx.pat holds the layers found so far.

def _hit(p, i):
    return 1 if p[i] > 0 else 0


def _clear_of(p, ctx, others, present=lambda v: v > 0):
    n = k = 0
    for i in range(16):
        if present(p[i]):
            n += 1
            if any(ctx.pat[o][i] for o in others):
                k += 1
    return 1 - k / n if n else 0


def _kick_pickups(p, ctx):
    extra = [i for i in range(16) if p[i] and i % 4]
    return sum(1 for i in extra if i % 4 >= 2) / len(extra) if extra else 1


def _kick_room(p, ctx):
    n = count(p)
    return 1 - sum(1 for i in range(16) if p[i] and p[i - 1]) / n if n else 0


def _clap_ghosts_soft(p, ctx):
    extra = [i for i in range(16) if p[i] and i not in (4, 12)]
    if not extra:
        return 1
    return sum((.5 if i % 2 else 0) + (.5 if p[i] == 1 else 0) for i in extra) / len(extra)


def _hat_dynamics(p, ctx):
    on_beat = ctx.g.hat_accent == "on"
    n = s = 0
    for i in range(16):
        if p[i]:
            n += 1
            ideal = 1 if i % 2 else (3 if (i % 4 == 0) == on_beat else 2)
            s += 1 - abs(p[i] - ideal) / 2
    return s / n if n else 0


def _perc_ghosts(p, ctx):
    n = count(p)
    return near(sum(1 for v in p if v == 1) / n, ctx.g.perc_ghost, .6) if n else 0


def _bass_root(p, ctx):
    notes = [v for v in p if v >= 0]
    if not notes:
        return 0
    share = sum(1 for v in notes if v in (0, 2)) / len(notes)
    return near(share, ctx.g.bass_root_share, .4) * (1 if notes[0] in (0, 2) else .7)


def _sync_rule(target_key, width):
    def fn(p, ctx):
        b = bits_of(p) if min(p) < 0 else [1 if v else 0 for v in p]
        return near(syncopation(b), getattr(ctx.g, target_key), width) if count(b) else 0
    return fn


def _root_moves(p, ctx):
    s = n = 0
    for i in range(4):
        a, b = p["roots"][i], p["roots"][(i + 1) % 4]
        if a != b:
            s += ctx.g.move[(b - a) % 7]
            n += 1
    return s / n if n else 1


def _voice_leading(p, ctx):
    v = [voicing(x, p["inv"][i], ctx.g) for i, x in enumerate(p["roots"])]
    s = n = 0
    for i in range(4):
        j = (i + 1) % 4
        if p["roots"][i] != p["roots"][j] or p["inv"][i] != p["inv"][j]:
            s += voice_distance(v[i], v[j])
            n += 1
    return clamp(1 - (s / n - 1) / 3) if n else 1


def _chord_rhythm(p, ctx):
    n = count(p["hit"])
    if not n:
        return 0
    return .5 * near(n, ctx.g.chord_hits, 3) + .5 * near(syncopation(p["hit"]), ctx.g.chord_sync, 5)


def _lead_fits(p, ctx):
    g = ctx.g
    roots = ctx.pat["chords"]["roots"] if "chords" in ctx.pat else [0] * 4
    s = w = 0
    for bar in range(4):
        tones = [semi(d, g) % 12 for d in chord_degrees(roots[bar], g)]
        o = (bar % 2) * 16
        for i in range(16):
            v = p[o + i]
            if v < 0:
                continue
            n = g.scale[v % 7]
            fit = 1 if n in tones else .1 if any((n - x) % 12 == 1 for x in tones) else .6
            s += fit * GRID[i]
            w += GRID[i]
    return s / w if w else 0


def _lead_steps(p, ctx):
    v = [x for x in p if x >= 0]
    if len(v) < 2:
        return 0
    return near(sum(abs(v[(i + 1) % len(v)] - v[i]) for i in range(len(v))) / len(v), 1.5, 2.5)


def _lead_question_answer(p, ctx):
    same = sum(1 for i in range(16) if (p[i] >= 0) == (p[i + 16] >= 0))
    q = next((v for v in reversed(p[:16]) if v >= 0), -1)
    a = next((v for v in reversed(p[16:]) if v >= 0), -1)
    open_end = 0 if q < 0 else .3 if q % 7 == 0 else 1
    closed_end = 0 if a < 0 else 1 if a % 7 == 0 else .6 if a % 7 in (2, 4) else .2
    return .4 * near(same / 16, .8, .6) + .3 * open_end + .3 * closed_end


def _lead_sync(p, ctx):
    s = 0
    for h in (0, 16):
        b = bits_of(p[h:h + 16])
        s += near(syncopation(b), ctx.g.lead_sync, 5) if count(b) else 0
    return s / 2


# Rules that look across layers. They only run in the second pass, once every part exists.

def velocity_at(layer, p, s, g):
    i = s % 16
    if layer in ("kick", "ohat"):
        return 1.0 if p[i] else 0.0
    if layer in ("clap", "hat", "perc"):
        return LEVEL[p[i]]
    if layer == "bass":
        return 1.0 if p[i] >= 0 else 0.0
    if layer == "chords":
        return 1.0 if p["hit"][i] else (.5 if g.chord_type == "pad" else 0.0)
    return ((p[i] >= 0) + (p[i + 16] >= 0)) / 2


def _steps(layer):
    return 32 if layer == "lead" else 16


def _own_velocity(layer, p, s, g):
    return (1.0 if p[s] >= 0 else 0.0) if layer == "lead" else velocity_at(layer, p, s, g)


def clash(layer, p, ctx):
    """Share of this layer's energy that lands in a band another layer fills on the same step."""
    me = ctx.profiles[layer]
    total = sum(me)
    if not total:
        return 0
    covered = weight = 0
    for s in range(_steps(layer)):
        v = _own_velocity(layer, p, s, ctx.g)
        if v <= 0:
            continue
        weight += v
        for b in range(len(BANDS)):
            others = sum(velocity_at(k, ctx.pat[k], s, ctx.g) * ctx.profiles[k][b]
                         for k in LAYERS if k != layer and k in ctx.pat)
            covered += min(v * me[b], others)
    return covered / (weight * total) if weight else 0


def crowding(layer, p, ctx):
    hits = bad = 0
    for s in range(_steps(layer)):
        if _own_velocity(layer, p, s, ctx.g) <= 0 and not (layer == "chords" and p["hit"][s]):
            continue
        if layer == "chords" and not p["hit"][s]:
            continue
        hits += 1
        others = sum(1 for k in LAYERS if k != layer and k in ctx.pat and
                     (ctx.pat[k]["hit"][s % 16] if k == "chords" else velocity_at(k, ctx.pat[k], s, ctx.g) > 0))
        if others >= ctx.g.crowd:
            bad += 1
    return bad / hits if hits else 0


def _answers_chords(p, ctx):
    notes = [s for s in range(32) if p[s] >= 0]
    if not notes:
        return 0
    return 1 - sum(1 for s in notes if ctx.pat["chords"]["hit"][s % 16]) / len(notes)


def _mix(layer):
    return ("Clear mix", 2, "plomp", lambda p, ctx: 1 - clash(layer, p, ctx))


def _crowd(layer):
    return ("Not too crowded", 2, None, lambda p, ctx: 1 - crowding(layer, p, ctx))


SPECS = {
    "kick": dict(extra=6, init=binary_init(3, 6), moves=binary_moves, rules=[
        ("Four on the floor", 4, "butler", lambda p, c: (p[0] + p[4] + p[8] + p[12]) / 4),
        ("Few extra kicks", 2, None, lambda p, c: near(count(p) - (p[0] + p[4] + p[8] + p[12]), c.g.kick_extra, 2)),
        ("Room to ring", 1.5, None, _kick_room),
        ("Extras lead into a beat", 1, None, _kick_pickups)]),
    "clap": dict(extra=6, init=level_init(2, 4), moves=level_moves, rules=[
        ("Backbeat", 4, "butler", lambda p, c: (p[4] + p[12]) / 6),
        ("Off the downbeats", 2, None, lambda p, c: 1 - (_hit(p, 0) + _hit(p, 8)) / count(p) if count(p) else 0),
        ("Ghost claps", 1.5, None, lambda p, c: near(count(p) - _hit(p, 4) - _hit(p, 12), c.g.clap_extra, 2)),
        ("Ghosts stay soft", 1, "witek", _clap_ghosts_soft)]),
    "hat": dict(extra=8, init=level_init(4, 9), moves=level_moves, rules=[
        ("Fills the gaps", 3, "butler", lambda p, c: sum(_hit(p, i) for i in (2, 6, 10, 14)) / 4),
        ("Right amount", 2, None, lambda p, c: near(count(p), c.g.hats, 5)),
        ("Evenly spread", 2, "toussaint", lambda p, c: evenness(p)),
        ("Repeats each half bar", 1, "butler", lambda p, c: sum(1 for i in range(8) if _hit(p, i) == _hit(p, i + 8)) / 8),
        ("Loud and soft", 2, None, _hat_dynamics)],
        second=lambda: [_mix("hat"), _crowd("hat")]),
    "ohat": dict(extra=6, init=binary_init(1, 4), moves=binary_moves, rules=[
        ("On the off-beat", 3, "butler", lambda p, c: sum(p[i] for i in (2, 6, 10, 14)) / count(p) if count(p) else 0),
        ("Right amount", 2, None, lambda p, c: near(count(p), c.g.open_hats, 2)),
        ("Clear of kick and clap", 2, None, lambda p, c: _clear_of(p, c, ("kick", "clap")))]),
    "perc": dict(extra=8, init=level_init(2, 7), moves=level_moves, rules=[
        ("Evenly spread", 2, "toussaint", lambda p, c: evenness(p)),
        ("Medium syncopation", 2, "witek", _sync_rule("perc_sync", 4)),
        ("Fills empty space", 2, "butler", lambda p, c: _clear_of(p, c, ("kick", "clap"))),
        ("Right amount", 1.5, None, lambda p, c: near(count(p), c.g.percs, 2)),
        ("Loud and soft", 1.5, None, _perc_ghosts)],
        second=lambda: [_mix("perc"), _crowd("perc")]),
    "bass": dict(extra=8, init=bass_init, moves=bass_moves, rules=[
        ("Leaves room for the kick", 3, "plomp", lambda p, c: _clear_of(p, c, ("kick",), lambda v: v >= 0)),
        ("Root first", 2.5, "krumhansl", _bass_root),
        ("Right amount", 2, None, lambda p, c: near(count(bits_of(p)), c.g.bass_notes, 4)),
        ("Push and pull", 2, "witek", _sync_rule("bass_sync", 5))],
        second=lambda: [_mix("bass")]),
    "chords": dict(extra=8, init=chords_init, moves=chords_moves, rules=[
        ("Starts at home", 3, "krumhansl", lambda p, c: 1 if p["roots"][0] == 0 else 0),
        ("Strong root moves", 2.5, "tymoczko", _root_moves),
        ("Smooth voice leading", 2.5, "tymoczko", _voice_leading),
        ("Right amount of change", 2, None, lambda p, c: near(len(set(p["roots"])), c.g.variety, 2)),
        ("This run's taste", 1.5, None, lambda p, c: sum(c.g.taste[x] for x in p["roots"][1:]) / 3),
        ("Chord rhythm", 2, "witek", _chord_rhythm)],
        second=lambda: [_mix("chords"), _crowd("chords")]),
    "lead": dict(extra=8, init=lead_init, moves=lead_moves, rules=[
        ("Fits the chords", 3, "plomp", _lead_fits),
        ("Small steps", 2, "huron", _lead_steps),
        ("Question and answer", 2.5, "caplin", _lead_question_answer),
        ("Right amount", 1.5, None, lambda p, c: near(count(bits_of(p)) / 2, c.g.lead_notes, 4)),
        ("Push and pull", 1.5, "witek", _lead_sync)],
        second=lambda: [_mix("lead"), _crowd("lead"), ("Answers the chords", 1.5, "butler", _answers_chords)]),
}


# ---------- search ----------

def rules_for(layer, ctx):
    spec = SPECS[layer]
    return spec["rules"] + (spec["second"]() if ctx.second and "second" in spec else [])


def evaluate(layer, p, ctx):
    rules = rules_for(layer, ctx)
    parts = [clamp(fn(p, ctx)) for _, _, _, fn in rules]
    return sum(r[1] * s for r, s in zip(rules, parts)) / sum(r[1] for r in rules), parts


def climb(layer, p, ctx, rng, max_rounds=80):
    """Hill-climb: each round scores every one-step change plus a few random two-step jumps."""
    spec = SPECS[layer]
    score, parts = evaluate(layer, p, ctx)
    best_by_round, tried = [score], 1
    for _ in range(max_rounds):
        cands = spec["moves"](p, ctx)
        for _ in range(spec["extra"]):
            a = rng.choice(spec["moves"](p, ctx))
            cands.append(rng.choice(spec["moves"](a, ctx)))
        best, best_score, ties = None, -1, 0
        for c in cands:
            s, sp = evaluate(layer, c, ctx)
            tried += 1
            if s > best_score + 1e-9:
                best, best_score, ties = (c, sp), s, 1
            elif s > best_score - 1e-9:
                ties += 1
                if rng.random() < 1 / ties:     # many patterns tie; pick among them at random
                    best = (c, sp)
        if best_score <= score + 1e-9:
            break
        p, parts, score = best[0], best[1], best_score
        best_by_round.append(score)
    rules = rules_for(layer, ctx)
    return p, dict(fit=score, tried=tried, rounds=len(best_by_round) - 1, history=best_by_round,
                   rules=[dict(name=r[0], weight=r[1], source=r[2], score=s) for r, s in zip(rules, parts)])


def first_pass(g, rng, starts=4):
    """Find each layer in order, each one scored against the layers before it. A climb stops at
    the first pattern no single change improves, so each layer gets a few random starts and
    keeps the best: with one start, 11 of 100 seeds ended with a percussion part far from its
    target; with four, 1 of 100."""
    ctx = SimpleNamespace(g=g, pat={}, profiles=PRIOR_PROFILE, second=False)
    report = {}
    for layer in LAYERS:
        tried = 0
        for _ in range(starts):
            p, r = climb(layer, SPECS[layer]["init"](rng, ctx), ctx, rng)
            tried += r["tried"]
            if layer not in report or r["fit"] > report[layer]["fit"]:
                ctx.pat[layer], report[layer] = p, r
        report[layer]["tried"] = tried
    return ctx.pat, report


def second_pass(g, pat, rng, profiles=None):
    """Re-tune five layers with every part in place. profiles: band energy per layer, measured
    from the rendered sound ({layer: [six numbers]}); the rough prior is used for any missing."""
    prof = dict(PRIOR_PROFILE)
    prof.update(profiles or {})
    ctx = SimpleNamespace(g=g, pat=dict(pat), profiles=prof, second=True)
    report = {}
    for layer in SECOND_PASS:
        ctx.pat[layer], report[layer] = climb(layer, ctx.pat[layer], ctx, rng)
    return ctx.pat, report


# ---------- arrangement ----------

ALL = list(LAYERS)


def sections(bars):
    """The shape of a track for any even bar count from 8 up. 72 bars is the full form: intro,
    groove, drop, breakdown, build-up, second drop, outro (Butler 2006; Solberg & Dibben 2019).
    Shorter tracks keep the parts in that order and lose the groove, then the breakdown."""
    if bars < 8 or bars % 2:
        raise ValueError("bars must be an even number, 8 or more")
    if bars < 12:
        return [("Loop", bars, ALL, {})]
    if bars < 24:
        return [("Intro", 4, ["kick", "hat", "perc"], dict(add={2: ["bass"]})),
                ("Drop", bars - 8, ALL, dict(drop=True, big=True)),
                ("Outro", 4, ["kick", "hat", "perc", "bass"], dict(cut={2: ["bass"]}))]
    n = 8 if bars >= 48 else 4                      # length of the intro, breakdown, build and outro
    drops = bars - (5 * n if n == 8 else 4 * n)
    a = max(4, drops // 2 // 4 * 4)
    out = [("Intro", n, ["kick", "hat"], dict(add={n // 2: ["perc"]}))]
    if n == 8:
        out.append(("Groove", n, ["kick", "hat", "perc", "clap", "ohat", "bass"], {}))
    return out + [
        ("Drop A", a, ALL, dict(drop=True, late_lead=a // 2 if a >= 8 else None)),
        ("Break", n, ["chords", "lead"], dict(add={n // 2: ["perc"]}, lp=(900, 5000))),
        ("Build", n, ["chords", "lead", "perc", "hat"], dict(build=True, lp=(5000, 18000), hp=(30, 450))),
        ("Drop B", drops - a, ALL, dict(drop=True, big=True)),
        ("Outro", n, ["kick", "hat", "perc", "clap", "bass"], dict(cut={n // 2: ["clap", "bass"]}))]


def make_plan(g, bars=72):
    """One entry per bar: which layers play, fills, filter sweeps and section flags."""
    plan = []
    for si, (name, length, layers, opt) in enumerate(sections(bars)):
        on = set(layers)
        late = opt.get("late_lead") if g.late_lead else None
        if late:
            on.discard("lead")
        sweep = lambda a, k: a[0] * (a[1] / a[0]) ** (k / length)
        for i in range(length):
            on |= set(opt.get("add", {}).get(i, []))
            on -= set(opt.get("cut", {}).get(i, []))
            if late and i == late:
                on.add("lead")
            lp, hp = opt.get("lp"), opt.get("hp")
            plan.append(dict(
                section=name, si=si, i=i, len=length, on=sorted(on), first=i == 0,
                drop=bool(opt.get("drop")), big=bool(opt.get("big")), build=bool(opt.get("build")),
                fill=(2 if (i + 1) % 8 == 0 else 1 if (i + 1) % 4 == 0 else 0) if "kick" in on else 0,
                lp=(sweep(lp, i), sweep(lp, i + 1)) if lp else None,
                hp=(sweep(hp, i), sweep(hp, i + 1)) if hp else None))
    for k, b in enumerate(plan):
        nxt = plan[k + 1] if k + 1 < len(plan) else None
        b["pre"] = bool(nxt and nxt["first"] and nxt["drop"] and not b["build"])
    return plan


def bar_events(pat, g, pb, bar):
    """What sounds in one bar: the patterns plus fills, rolls and section changes.
    pb is the plan entry (None = the bare loop, every layer on)."""
    on = lambda layer: layer in pat and (pb is None or layer in pb["on"])
    fill = pb["fill"] if pb else 0
    big = bool(pb and pb["big"])
    ev = dict(chord=bar % 4, vel={}, bass=None, lead=None, chord_hit=None, ride=None,
              octave=False, gap_from=16, wet_clap=fill == 2)
    v = ev["vel"]
    if on("kick"):
        v["kick"] = [(1.0 if i % 4 == 0 else .78) if x else 0.0 for i, x in enumerate(pat["kick"])]
        if fill == 2:
            if g.fill_kind == 0:
                v["kick"][12:16] = [0.0] * 4            # drop the kick out before the turn
            else:
                v["kick"][14], v["kick"][15] = .8, .9   # or double it
    if on("clap"):
        v["clap"] = [LEVEL[x] for x in pat["clap"]]
        if fill == 1 and not v["clap"][15]:
            v["clap"][15] = .35
        if fill == 2:
            for k in range(4):
                v["clap"][12 + k] = max(v["clap"][12 + k], .4 + .18 * k)
    if pb and pb["build"]:                               # the build-up roll: quarters, eighths, sixteenths
        left = pb["len"] - pb["i"]
        roll = [0.0] * 16
        for s in range(16):
            if left in (3, 4) and s % 4 == 0:
                roll[s] = .45
            elif left == 2 and s % 2 == 0:
                roll[s] = .4 + .02 * s
            elif left == 1:
                roll[s] = .35 + .05 * s
        v["clap"] = roll
        if left == 1:
            ev["gap_from"] = 12                          # one beat of silence before the drop
    if on("hat"):
        v["hat"] = [LEVEL[x] for x in pat["hat"]]
        if big:
            for s in range(1, 16, 2):
                v["hat"][s] = v["hat"][s] or .3
        if fill == 1:
            v["hat"][13], v["hat"][15] = max(v["hat"][13], .5), max(v["hat"][15], .6)
    kick_at = lambda s: bool(v.get("kick") and v["kick"][s])
    if on("ohat"):
        v["ohat"] = [1.0 if x else 0.0 for x in pat["ohat"]]
        if fill and not kick_at(14):
            v["ohat"][14] = 1.0
    if "hat" in v and "ohat" in v:                       # an open hat chokes the closed one
        v["hat"] = [0.0 if o else h for h, o in zip(v["hat"], v["ohat"])]
    if on("perc"):
        v["perc"] = [LEVEL[x] for x in pat["perc"]]
        if fill == 2 and g.fill_kind == 1:
            for k in range(4):
                v["perc"][12 + k] = max(v["perc"][12 + k], .5 + .15 * k)
    if on("bass"):
        ev["bass"] = pat["bass"][:]
        if fill:
            if not kick_at(13):
                ev["bass"][13] = 2
            if not kick_at(15):
                ev["bass"][15] = 1
    if on("chords"):
        ev["chord_hit"] = pat["chords"]["hit"][:]
    if on("lead"):
        o = (bar % 2) * 16
        ev["lead"] = pat["lead"][o:o + 16]
        ev["octave"] = big
    if big:
        ev["ride"] = [0.0 if i % 2 else (.9 if i % 4 == 2 else .6) for i in range(16)]
    return ev


# ---------- the whole thing, and how to read it ----------

def compose(seed, style=None, bpm=None, key=None, mode=None, measure=None):
    """Run both passes. measure(g, patterns) -> {layer: band profile} is the listening step;
    without it the second pass uses the rough prior."""
    g = make_genome(seed, style, bpm, key, mode)
    rng = random.Random(f"search/{seed}")
    pat, first = first_pass(g, rng)
    profiles = measure(g, pat) if measure else None
    pat, second = second_pass(g, pat, rng, profiles)
    return g, pat, dict(first=first, second=second, profiles=profiles)


def lanes(g, pat):
    """The patterns as text an agent can read."""
    out = []
    for layer in LAYERS:
        p = pat[layer]
        if layer in ("kick", "ohat"):
            s = "".join("X" if x else "." for x in p)
        elif layer in ("clap", "hat", "perc"):
            s = "".join(".oxX"[x] for x in p)
        elif layer == "bass":
            s = "".join("." if x < 0 else BASS_CHAR[x] for x in p)
        elif layer == "chords":
            s = "".join("X" if x else "." for x in p["hit"]) + "  " + " | ".join(
                "{} {}".format(*chord_name(r, g)) for r in p["roots"])
        else:
            s = "".join("." if x < 0 else str(x) for x in p[:16]) + " " + "".join(
                "." if x < 0 else str(x) for x in p[16:])
        out.append(f"{layer:<7}{s}")
    return out


def describe(g, pat, report):
    lines = [f"seed {g.seed}  {g.style}  {g.bpm:.5g} BPM  {g.key_name}  swing {g.swing * 100:.0f}%  "
             f"chords: {g.chord_type}",
             "timing ms: " + "  ".join(f"{k} {v * 1000:+.0f}" for k, v in g.feel.items()), "",
             "       1...2...3...4...   (. rest, o soft, x medium, X loud; bass R root 5 fifth O octave 3 third;",
             "                           lead = scale degree 0-9 over two bars)"]
    lines += lanes(g, pat)
    lines.append("")
    for layer in LAYERS:
        r = report["second"].get(layer) or report["first"][layer]
        tried = report["first"][layer]["tried"] + (report["second"][layer]["tried"] if layer in report["second"] else 0)
        weak = [f"{x['name']} {x['score'] * 100:.0f}%" for x in r["rules"] if x["score"] < .7]
        lines.append(f"{layer:<7}fit {r['fit'] * 100:3.0f}%  {tried:5d} patterns tried"
                     + ("   weakest: " + ", ".join(weak) if weak else ""))
    return "\n".join(lines)


def to_json(g, pat, report, plan=None):
    return dict(genome={k: v for k, v in vars(g).items()}, patterns=pat, report=report,
                chords=[chord_name(r, g) for r in pat["chords"]["roots"]],
                plan=[dict(bar=i + 1, section=b["section"], on=b["on"], fill=b["fill"]) for i, b in enumerate(plan or [])],
                sources=SOURCES)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--style", choices=sorted(STYLES))
    ap.add_argument("--bpm", type=int)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else random.randrange(1000, 10000)
    g, pat, report = compose(seed, a.style, a.bpm)
    if a.json:
        print(json.dumps(to_json(g, pat, report, make_plan(g)), indent=1))
    else:
        print(describe(g, pat, report))


if __name__ == "__main__":
    main()
