import muspy
import pretty_midi as pm
import numpy as np
import torch
import matplotlib.pyplot as plt
import music21

BIN = 4

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_music(midi_path):
    music = muspy.read_midi(midi_path)
    music.adjust_resolution(BIN)
    if len(music.time_signatures) == 0:
        music.time_signatures.append(muspy.TimeSignature(0, 4, 4))
    return music


def get_note_matrix(music, tracks=[0]):
    """
    get note matrix: same format as pop909-4-bin
        for piano, this function simply extracts notes of given tracks
    """
    notes = []
    for inst_idx in tracks:
        inst = music.tracks[inst_idx]
        for note in inst.notes:
            onset = int(note.time)
            duration = int(note.duration)
            if onset >= 0:
                # this is compulsory because there may be notes
                # with zero duration after adjusting resolution
                if duration == 0:
                    duration = 1
                notes.append([
                    onset,
                    note.pitch,
                    duration,
                ])
    # sort according to (start, duration)
    # notes.sort(key=lambda x: (x[0] * BIN + x[1], x[2]))
    notes.sort(key=lambda x: (x[0], x[1], x[2]))
    return notes


def nmat_to_prmat(nmat, num_2bar, n_step=32):
    prmat = torch.zeros((num_2bar, n_step, 128))
    for o, p, d in nmat:
        n_2bar = o // n_step
        if n_2bar >= num_2bar:
            break
        o_2bar = o % n_step
        prmat[n_2bar, o_2bar, p] = d
    return prmat


def nmat_to_melprmat(nmat, num_bar, n_step=16):  # only 1 bar!
    prmat_ec2vae = torch.zeros((num_bar * n_step, 130))
    t = 0
    for o, p, d in nmat:
        if o >= num_bar * n_step:
            break
        if o > t:
            for i in range(t, o):
                prmat_ec2vae[i, 129] = 1  # rest
        prmat_ec2vae[o, p] = 1
        for i in range(o + 1, min(o + d, num_bar * n_step)):
            prmat_ec2vae[i, 128] = 1
        t = o + d
    if t < num_bar * n_step:
        prmat_ec2vae[t : num_bar * n_step, 129] = 1
    prmat_ec2vae = prmat_ec2vae.reshape((num_bar, n_step, 130))
    return prmat_ec2vae.to(device)


def nmat_to_chd_ec2vae(chd_nmat, num_bar, n_step=16):  # only 1 bar
    """
    track indicates the chord track in the midi file
    """
    chd = torch.zeros((num_bar * n_step, 12))
    idx = 0
    chd_nmat = np.array(chd_nmat)
    if chd_nmat.shape[1] == 16:
        # chd_nmat from training
        # chd_nmat : [:, 16]
        #     0: start time (in beats, not in steps)
        #     1: root
        #     2-13: absolute chroma
        #     14: bass
        #     15: dur (in beats, not in steps)
        while idx < len(chd_nmat):
            t = chd_nmat[idx][0] * 4
            d = chd_nmat[idx][15] * 4
            chd[t : min(num_bar * n_step, t + d), :] = torch.from_numpy(
                chd_nmat[idx][2 : 14]
            )
            idx += 1
    elif chd_nmat.shape[1] == 3:
        while idx < len(chd_nmat):
            t = chd_nmat[idx][0]
            d = chd_nmat[idx][2]
            ps = []
            while idx < len(chd_nmat) and chd_nmat[idx][0] == t:
                ps.append(chd_nmat[idx][1])
                idx += 1
            if len(ps) == 1:
                continue  # only bass changed, ignore

            tmp_chd = [p % 12 for p in ps[1 :]]
            # tmp_chd = reduce_chd_quality(tmp_chd)

            for p in tmp_chd:
                chd[t : min(num_bar * n_step, t + d), p] = 1
    else:
        raise RuntimeError
    chd = chd.reshape((num_bar, n_step, 12))
    return chd


def nmat_to_melchroma(nmat, num_bar, n_step=16):  # only 1 bar!
    """pitch % 12, without sustain & rest"""
    prmat = torch.zeros((num_bar * n_step, 12))
    for o, p, d in nmat:
        if o >= num_bar * n_step:
            break
        prmat[o, p % 12] = 1
    prmat = prmat.reshape((num_bar, n_step, 12))
    return prmat.to(device)


def nmat_to_midi_file(nmat, fpath):
    """
    nmat: (#, 3)
        onset, pitch, duration
    """
    if "Tensor" in str(type(nmat)):
        nmat = nmat.cpu().detach().numpy()
    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)
    for note in nmat:
        if note[2] > 0:
            piano.notes.append(
                pm.Note(
                    velocity=80,
                    pitch=note[1],
                    start=note[0] * 1 / 8,
                    end=(note[0] + note[2]) * 1 / 8,
                )
            )
    midi.instruments.append(piano)
    midi.write(fpath)


def melprmat_to_midi_file(sample_roll, output='sample.mid'):
    # melprmat: (B, 32, 130)
    music = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program('Acoustic Grand Piano')
    piano = pm.Instrument(program=piano_program)
    t = 0
    if "Tensor" in str(type(sample_roll)):
        sample_roll = sample_roll.cpu().detach().numpy()
    for bar_ind, bars in enumerate(sample_roll):
        for step_ind, step in enumerate(bars):
            pitch = int(np.argmax(step))
            if pitch < 128:
                note = pm.Note(velocity=100, pitch=pitch, start=t, end=t + 1 / 8)
                t += 1 / 8
                piano.notes.append(note)
            elif pitch == 128:
                if len(piano.notes) > 0:
                    note = piano.notes.pop()
                else:
                    # p = np.random.randint(60, 72)
                    # note = pm.Note(velocity=100, pitch=int(p), start=0, end=t)
                    raise RuntimeError
                note = pm.Note(
                    velocity=100,
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end + 1 / 8
                )
                piano.notes.append(note)
                t += 1 / 8
            elif pitch == 129:
                t += 1 / 8
    music.instruments.append(piano)
    music.write(output)


def prmat_to_midi_file(prmat, fpath, labels=None):
    # prmat: (B, 32, 128)
    if "Tensor" in str(type(prmat)):
        prmat = prmat.cpu().detach().numpy()
    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)
    t = 0
    n_step = prmat.shape[1]
    t_bar = int(n_step / 8)
    for bar_ind, bars in enumerate(prmat):
        for step_ind, step in enumerate(bars):
            for key, dur in enumerate(step):
                dur = int(dur)
                if dur > 0:
                    note = pm.Note(
                        velocity=80,
                        pitch=key,
                        start=t + step_ind * 1 / 8,
                        end=min(t + (step_ind + int(dur)) * 1 / 8, t + t_bar),
                    )
                    piano.notes.append(note)
        t += t_bar
    midi.instruments.append(piano)
    if labels is not None:
        midi.lyrics.clear()
        t = 0
        for label in labels:
            midi.lyrics.append(pm.Lyric(label, t))
            t += t_bar
    midi.write(fpath)


def reduce_chd_quality(chd):  # 12
    chord = music21.chord.Chord(chd)
    chd_red = []
    root = None
    if chord.root() is not None:
        root = chord.root().pitchClass
        chd_red.append(root)
    if chord.third is not None:
        third = chord.third.pitchClass
        chd_red.append(third)
    if chord.fifth is not None:
        fifth = chord.fifth.pitchClass
        chd_red.append(fifth)
    if chord.semitonesFromChordStep(2) is not None:
        if chord.third is None:
            second = (chord.semitonesFromChordStep(2) + root) % 12
            chd_red.append(second)
    if chord.semitonesFromChordStep(4) is not None:
        if chord.third is None:
            fourth = (chord.semitonesFromChordStep(4) + root) % 12
            chd_red.append(fourth)
    return chd_red, root


def get_chord(music, num_2bar, tracks=[1]):
    """
    track indicates the chord track in the midi file
    """
    chd_nmat = get_note_matrix(music, tracks)

    chd = torch.zeros((num_2bar * 8, 36))
    idx = 0
    while idx < len(chd_nmat):
        t = chd_nmat[idx][0]
        d = chd_nmat[idx][2]
        ps = []
        while idx < len(chd_nmat) and chd_nmat[idx][0] == t:
            ps.append(chd_nmat[idx][1])
            idx += 1
        if len(ps) == 1:
            continue  # only bass changed, ignore

        tmp_chd = [p % 12 for p in ps[1 :]]
        tmp_chd, root = reduce_chd_quality(tmp_chd)

        if root is None:
            root = ps[0] % 12
        bass = root

        bar = t // 4
        d = min(d // 4, num_2bar * 8 - bar)
        chd[bar : bar + d, root] = 1
        chd[bar : bar + d, 24 + bass] = 1

        for p in tmp_chd:
            chd[bar : bar + d, 12 + p] = 1
    chd = chd.reshape((num_2bar, 8, 36))
    return chd


# def get_chord_ec2vae(music, num_2bar, tracks=[1]):
#     """
#     track indicates the chord track in the midi file
#     """
#     chd_nmat = get_note_matrix(music, tracks)
#
#     chd = torch.zeros((num_2bar * 32, 12))
#     idx = 0
#     while idx < len(chd_nmat):
#         t = chd_nmat[idx][0]
#         d = chd_nmat[idx][2]
#         ps = []
#         while idx < len(chd_nmat) and chd_nmat[idx][0] == t:
#             ps.append(chd_nmat[idx][1])
#             idx += 1
#         if len(ps) == 1:
#             continue  # only bass changed, ignore
#
#         tmp_chd = [p % 12 for p in ps[1 :]]
#         tmp_chd = reduce_chd_quality(tmp_chd)
#
#         for p in tmp_chd:
#             chd[t : min(num_2bar * 32, t + d), p] = 1
#     chd = chd.reshape((num_2bar, 32, 12))
#     return chd
#


def chd_to_midi_file(chords, output_fpath, one_beat=0.5):
    """
    retrieve midi from chords
    """
    if "Tensor" in str(type(chords)):
        chords = chords.cpu().detach().numpy()
    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)
    t = 0.0
    for seg in chords:
        for beat, chord in enumerate(seg):
            if chord.shape[0] == 14:
                root = int(chord[0])
                chroma = chord[1 : 13].astype(int)
                bass = int(chord[13])
            elif chord.shape[0] == 36:
                root = int(chord[0 : 12].argmax())
                chroma = chord[12 : 24].astype(int)
                bass = int(chord[24 :].argmax())
            else:
                raise RuntimeError

            chroma = np.roll(chroma, -bass)
            c3 = 48
            for i, n in enumerate(chroma):
                if n == 1:
                    note = pm.Note(
                        velocity=80,
                        pitch=c3 + i + bass,
                        start=t * one_beat,
                        end=(t + 1) * one_beat,
                    )
                    piano.notes.append(note)
            t += 1

    midi.instruments.append(piano)
    midi.write(output_fpath)


def chd_ec2vae_to_midi_file(chords, output_fpath, one_step=1 / 8):
    """
    retrieve midi from chords
    """
    if "Tensor" in str(type(chords)):
        chords = chords.cpu().detach().numpy()
    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)
    t = 0.0
    for seg in chords:
        for beat, chord in enumerate(seg):
            if chord.shape[0] == 12:
                chroma = chord.astype(int)
            else:
                raise RuntimeError

            c3 = 48
            for i, n in enumerate(chroma):
                if n == 1:
                    note = pm.Note(
                        velocity=80,
                        pitch=c3 + i,
                        start=t * one_step,
                        end=(t + 1) * one_step,
                    )
                    piano.notes.append(note)
            t += 1

    midi.instruments.append(piano)
    midi.write(output_fpath)


def phrase_config_from_string(phrase_annotation):
    index = 0
    phrase_configuration = []
    while index < len(phrase_annotation):
        label = phrase_annotation[index]
        index += 1
        n_bars = ""
        while index < len(phrase_annotation) and phrase_annotation[index].isdigit():
            n_bars += phrase_annotation[index]
            index += 1
        phrase_configuration.append((label, int(n_bars)))
    return phrase_configuration


def show_matrix(mat):
    matrix_np = mat.detach().numpy()

    # Plot the matrix
    plt.imshow(matrix_np, cmap="viridis", interpolation="nearest")
    plt.colorbar()  # Add a color bar for reference
    plt.title("Matrix Visualization")
    plt.show()
