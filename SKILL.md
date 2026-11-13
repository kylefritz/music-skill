---
name: music-skill
description: Make, check and understand music from a terminal, for an agent that can't hear it. Use when someone wants music made - a house or techno track, a beat, a loop, a bed under a video or a voice, music at an exact length, tempo or key, stems or MIDI - or wants to know what's in a piece of audio ("what's the BPM", "what key is this", "where's the drop", "why is this mix muddy"), or needs audio checked before it goes out (loudness in LUFS, true peak, clipping, mono compatibility). Also for why a groove works - syncopation, swing, microtiming, voice leading, breakdown and drop - and for checking music a generative model made.
---

# music-skill

Scripts live in `scripts/` next to this file; write it as `$S` below, but spell the absolute path
out in each command, because shell variables don't carry between Bash calls.

You can't hear, so every step turns music into something you can read: **patterns as text, numbers
you check, and one picture you look at.** Making a track is seed → read the score → render → `look`
→ fix. Understanding audio is `look` → read the timeline and the sheet → judge. Delivering is
`check`, so "it's the right loudness and won't clip" is a fact.

Needs [uv](https://docs.astral.sh/uv/) and ffmpeg. Every command runs as
`uv run --no-project --with numpy --with scipy --with pillow python -I <script> …`, written `RUN` below.

## Making a track: banger.py

```bash
RUN $S/banger.py --seed 4242 --out <scratch>/b4242
```

A 72-bar house or techno track takes about 10 seconds: `track.wav` (48 kHz stereo, -14 LUFS),
`track.mid`, `score.txt` and `score.json`. The same seed and options always give the same file.

| Want | Option |
| --- | --- |
| A style | `--style deep-house` (120-124 BPM), `tech-house` (124-128), `peak-techno` (130-136), `melodic-techno` (122-126) |
| A tempo, a key | `--bpm 124`, `--key A`, `--mode minor` (or `dorian`, `phrygian`) |
| A length | `--bars 24` (any even number from 8), or `--seconds 30`: it picks the bars and nudges the tempo so the last bar ends 2.5 s early and the reverb rings out on time |
| A bed under a voice | `--lufs -22 --mute lead` (also try muting `clap`) |
| Parts for a DAW or a mix check | `--stems` writes `stems/<layer>.wav`; `track.mid` is always written |

Bars decide the shape: 8-10 is a loop; 12-22 is intro, drop, outro; 24-46 adds a breakdown and a
build-up between two drops; 48 and up is the full form with a groove section and 8-bar parts.

**Read the score before you render more.** It prints the patterns as text:

```text
seed 4242  Melodic techno  122 BPM  G minor  swing 1%  chords: pluck
       1...2...3...4...
kick   X...X...X.X.X...
clap   .o..X.......X...
hat    x.X...X.x.X...X.
bass   .....ORO.O.O3RRR
chords ...X............  i7 Gm7 | bVII7 F7 | bVI7 Ebmaj7 | iv7 Cm7
lead   47244.42.22..442 .3315.553.5.3557
perc   fit  74%   1293 patterns tried   weakest: Evenly spread 52%, Clear mix 20%
```

`.` rest, `o` soft, `x` medium, `X` loud. Bass: `R` root, `5` fifth, `O` octave, `3` third. Lead:
scale degree 0-9 over two bars. `fit` is how well the part meets its rules; `weakest` names the
rules it missed.

To pick a seed, audition as text first: `python3 $S/groove.py --seed N` runs only the search
(0.5 s, no audio, no dependencies). Try a handful, read the chords and the lead, render the one
you'd defend. You can't judge "does it sound good": say so, and give the person the file.

## How a track gets written

The method is from [Banger Lab](https://claude.ai/artifact/N6bSgS4ZH5ifFQfg8sm4dv). It is search,
not a model, and every rule comes from a study of rhythm or pitch perception.

1. **Roll.** The seed picks a style, tempo, key, scale, and targets: hats per bar, how much
   syncopation, how many different chords.
2. **Guess.** Each instrument starts as a random 16-step pattern.
3. **Test.** Every one-step change (add, remove, slide, louder, softer) is scored against that
   instrument's rules, plus a few random two-step jumps. The best one is kept.
4. **Lock.** When no change scores higher, the part is done and the next instrument is written on
   top of it. Order: kick, clap, hats, open hat, percussion, bass, chords, lead. Each layer gets
   four random starts and keeps the best.
5. **Listen.** `banger.py` renders each instrument alone, measures where its energy sits in six
   bands, and a second pass re-tunes hats, percussion, bass, chords and lead so parts don't fill
   the same band on the same step.
6. **Arrange.** The parts become sections, with fills at the end of every 4 and 8 bars, a
   low-pass that opens through the breakdown, a snare roll and riser in the build, one beat of
   silence, then the drop.

## What makes it groove: the rules

Use these when you write or edit music by any means: a pattern by hand, MIDI, or a prompt for a
music model. `score.json` has every rule's score and source for a track.

| Part | Rule | From |
| --- | --- | --- |
| Kick | on every beat; at most 0-2 extras, placed just before a beat; never two in a row | Butler 2006 |
| Clap | loud on beats 2 and 4, off beats 1 and 3; extras are quiet and on weak 16ths | Butler 2006; Witek 2014 |
| Hats | on the off-beats, between the kicks; evenly spread; the same in both halves of the bar; loud and soft alternate | Butler 2006; Toussaint 2005 |
| Open hat | halfway between two kicks, never on a kick or clap | Butler 2006 |
| Percussion | evenly spread, where kick and clap are silent, with 25-50% quiet ghost notes | Toussaint 2005; Witek 2014 |
| Bass | plays where the kick doesn't; root (or its octave) 60-90% of the time, fifth or third for color | Plomp & Levelt 1965; Krumhansl & Kessler 1982 |
| Chords | bar 1 is the home chord; roots move by step, third or fourth; each voice moves the shortest way to the next chord | Krumhansl & Kessler 1982; Tymoczko 2011 |
| Lead | chord notes on strong beats; never a semitone above a chord note; small steps; bar 1 asks (ends off the home note), bar 2 answers with a similar rhythm (ends on it); plays in the gaps between chord hits | Plomp & Levelt 1965; Huron 2001; Caplin 1998 |
| Everything | **medium syncopation**: some notes on weak steps that skip a strong one. Too little is stiff, too much loses the beat | Witek et al. 2014 |
| Timing | swing and push/pull stay **under 12 ms** per instrument (hats late, clap early). Larger shifts lowered groove ratings | Davies et al. 2013 |
| Mix | two parts in the same band on the same step blur each other; no more than 4 parts hitting one step | Plomp & Levelt 1965 |
| Form | breakdown, build-up and drop: take the kick and bass out, raise tension, bring them back | Solberg & Dibben 2019; Butler 2006 |

Every note comes from one scale and the kick is tuned to the home note, which is why any seed
stays in key. The weights between rules are a simplification, not fitted to listening data, so
`fit` ranks patterns within a track; it is not a quality score across tracks.

## Listening: look

```bash
RUN $S/listen.py look TRACK --sheet <scratch>/sheet.png
```

Works on anything ffmpeg decodes, including a video's sound and music from a generative model.
It prints tempo, key, loudness, true peak, balance, stereo width, warnings, and a timeline:

```text
track.wav  2:24.1  122.0 BPM (strength 7.9)  G minor (score 1.62; next Eb major 1.61)
  loudness -14.0 LUFS  true peak -1.3 dBTP  range 9.3 LU  peak-to-loudness 12.7 dB
  balance, dB under the loudest band  sub -2  bass 0  low mid -8  mid -17  high -24  air -27
  timeline, 15.7 s windows (8 bars): loudness, then each band in dB against its level over the whole track
    start     LUFS        sub    bass low mid     mid    high     air
     0:47.2   -12.6         +2      +2      +1      +2      +1      -2
     1:03.0   -21.1        -29     -12      +1      -7      -7      -6
     1:18.7   -20.5        -33     -12      +1      -1      +3      +6
     1:34.4   -12.6         +2      +2      +1      +2      +1      -0
```

Read structure off the timeline. Sub and bass 20-30 dB down is a breakdown (kick and bass are
out). High and air climbing while sub stays out is a build. Everything back at once is the drop.
Then **read the sheet** (waveform, spectrogram, six band lanes, loudness curve): a riser is a
diagonal line, a kick is a column of bright dots at the bottom, a flat block of color with no
gaps is a mix with no dynamics. Give times from the sheet's axis when you describe a moment.

How far to trust it, measured on 24 tracks rendered with known answers:

- **Tempo**: within 0.2 BPM on 24 of 24. It prefers 70-190 BPM, so half or double time is
  possible on other material. Strength under 2 means no steady beat: say the BPM is a guess.
- **Key**: the home note was right on 19 of 24, and it's less reliable on short or sparse audio.
  Major vs minor is a weaker call, and it can't tell modes apart. When the top two scores are
  close it says so; report both.
- **Loudness and true peak**: the same numbers as `ffmpeg -af ebur128=peak=true` to 0.1.

## Stems against each other: mix

```bash
RUN $S/listen.py mix <scratch>/b4242/stems
```

Per stem: loudness and where its energy sits. Then the pairs that cover each other most (same
band, same moment), and stems that are very quiet next to the loudest. A pair over 60% is two
parts fighting for one space: move one in time, pitch or level.

## Numbers to hit, and check

| Target | Number |
| --- | --- |
| A track on its own (streaming normalizes near here) | **-14 LUFS** integrated |
| The finished mix of a web video or podcast | -16 LUFS |
| A bed under a voice | about -22 LUFS |
| True peak, always | **-1 dBTP or under**: lossy encoding overshoots |
| Peak-to-loudness under 7 dB | over-limited: flat and tiring |
| Stereo correlation under 120 Hz below 0.6 | bass may vanish in mono (phones, clubs) |

```bash
RUN $S/listen.py check TRACK --lufs -14 --peak -1 --seconds 30 --bpm 124
```

`check` must pass before you hand a file over; it exits 1 on a miss. It also fails clipping and
silence at the start. Music from a model needs this most: one generated bed measured +3.2 dBTP
with 2,121 clipped samples. Fix with ffmpeg (`loudnorm=I=-14:TP=-1.5:LRA=11`), then check again.

## Delivering

- Small copy: `ffmpeg -i track.wav -c:a aac -b:a 160k track.m4a` (or `-c:a libmp3lame -q:a 2`).
  Run `check` on the copy: encoding moves the true peak.
- Under a film: render with `--seconds <film length>` and `--lufs -22`, and hand it to
  [video-skill](https://github.com/kylefritz/video-skill), which checks the final mix.
- A loop: `--bars 8`, then cut the 2.5 s tail (`-t <8 bars in seconds>`) so it repeats cleanly.
- Keep `score.txt` with the file: its last line is the exact command that made it.

## When this is the wrong tool

`banger.py` makes four-on-the-floor electronic music from oscillators and noise. It has no
samples, no vocals and no other genres, and its mix is darker and narrower than a mastered
release (the demo's mids sit 17 dB under its bass). For acoustic instruments, a song, or a named
mood, use a music model or a library track, then run `look` and `check` on what comes back. What
this gives you instead: exact tempo, key and length, the same result from the same seed, stems,
MIDI, nothing to license, and a reason for every note.

## Common mistakes

| Mistake | Fix |
| --- | --- |
| Saying a track "sounds great" | You haven't heard it. Report the numbers and what the sheet shows; the person judges the sound |
| Rendering 20 seeds to choose one | `groove.py --seed N` prints the score in 0.5 s; render the pick |
| Comparing `fit` across tracks | It ranks patterns inside one track only |
| Trusting the key on a 10-second clip | Give both top candidates |
| Handing over the WAV without `check` | `check` it, and check the encoded copy too |
| Music from a model used as-is | `look` first: clipping and +dBTP peaks are common |
| A bed fighting the voice | `--lufs -22 --mute lead`, and keep 1-4 kHz clear |
