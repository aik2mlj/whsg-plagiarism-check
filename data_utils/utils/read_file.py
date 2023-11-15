import numpy as np
import mir_eval
import os
from .song_analyzer import Pop909Song


"""
Functions to read melody, chord and phrase label file from POP909 melody dataset.
* Phrase label: human_label1.txt or human_label2.txt
"""


def _cum_time_to_time_dur(d):
    d_cumsum = np.cumsum(d)
    starts = np.insert(d_cumsum, 0, 0)[0: -1]
    return starts, d


def _parse_phrase_label(phrase_string):
    phrase_start = [i for i, s in enumerate(phrase_string) if s.isalpha()] + [len(phrase_string)]
    phrase_names = [phrase_string[phrase_start[i]: phrase_start[i + 1]] for i in range(len(phrase_start) - 1)]
    phrase_type = [pn[0] for pn in phrase_names]
    phrase_lgth = np.array([int(pn[1:]) for pn in phrase_names])
    return phrase_names, phrase_type, phrase_lgth


def read_phrase_label(label_fn):
    """
    label_fn: human_label1.txt
    """

    with open(label_fn) as f:
        phrase_label = f.readlines()[0].strip()
    phrase_names, phrase_type, phrase_lgth = _parse_phrase_label(phrase_label)

    phrase_starts, _ = _cum_time_to_time_dur(phrase_lgth)

    phrases = [{'name': pn, 'type': pt, 'lgth': pl, 'start': ps}
               for pn, pt, pl, ps in zip(phrase_names, phrase_type,
                                         phrase_lgth, phrase_starts)]
    return phrases


"""
* Melody: melody.txt
"""


def _melody_to_nmat(m):
    starts, durs = _cum_time_to_time_dur(m[:, 1])
    pitches = m[:, 0]

    is_pitch = m[:, 0] != 0
    nmat = np.stack([starts, pitches, durs], -1)[is_pitch]
    return nmat


def read_melody(melody_fn):
    """melody_fn: melody.txt"""
    # convert txt file to numpy array of (pitch, duration)
    with open(melody_fn) as f:
        melody = f.readlines()
    melody = [m.strip().split(' ') for m in melody]
    melody = np.array([[int(m[0]), int(m[1])] for m in melody])

    # convert melody file to numpy array of (onset, duration)
    melody = _melody_to_nmat(melody)

    return melody


"""
* Chord: finalized_chord.txt
"""


def read_chord_string(c):
    """
    chord symbols are dirty. We must replace "\x01" with " ".
    There are 'sus4(b7' chords that forgets the ')'.
    Check with if c_name[-7:] == 'sus4(b7'.
    There are other similar problems.

    E.g.,
    if c_name[-7:] == 'sus4(b7':
        c_name = c_name.replace('sus4(b7', 'sus4(b7)')
    """
    c = c.strip().split(' ')
    c_name = c[0]
    c_dur = int(c[-1])

    c_name = c_name.replace('\x01', '')

    root, chroma, bass = mir_eval.chord.encode(c_name)
    chroma = np.roll(chroma, shift=root)
    return np.concatenate([np.array([root]), chroma, np.array([bass]), np.array([c_dur])])


def _chord_to_nmat(chords):
    starts, durs = _cum_time_to_time_dur(chords[:, -1])

    chords = np.concatenate([starts[:, np.newaxis], chords], -1)

    return chords


def read_chord(chord_fn):
    """chord_fn: finalized_chord.txt"""
    with open(chord_fn) as f:
        chord = f.readlines()
    chord = np.stack([read_chord_string(c) for c in chord], 0)

    # convert chord to output nmat
    chord = _chord_to_nmat(chord)
    return chord


def read_data(data_fn, acc_fn, num_beat_per_measure=4, num_step_per_beat=4,
              clean_chord_unit=None, song_name=None, label=1):
    if label == 1:
        label_fn = os.path.join(data_fn, 'human_label1.txt')
    elif label == 2:
        label_fn = os.path.join(data_fn, 'human_label2.txt')
    else:
        raise NotImplementedError

    label = read_phrase_label(label_fn)

    melody_fn = os.path.join(data_fn, 'melody.txt')
    melody = read_melody(melody_fn)

    chord_fn = os.path.join(data_fn, 'finalized_chord.txt')
    chord = read_chord(chord_fn)

    clean_chord_unit = num_beat_per_measure if clean_chord_unit is None else clean_chord_unit

    acc_mats = np.load(os.path.join(acc_fn, 'acc_mat.npz'))
    bridge_track, piano_track = acc_mats['bridge'], acc_mats['piano']
    acc = np.concatenate([bridge_track, piano_track], 0)
    acc = acc[acc[:, 0].argsort()]

    song = Pop909Song(melody, chord, acc, label, num_beat_per_measure,
                      num_step_per_beat, song_name, clean_chord_unit)

    return song

