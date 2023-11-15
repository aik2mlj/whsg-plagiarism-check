import warnings
import numpy as np


def remove_offset(
    note_mat, chord_mat, start_measure, measure_to_step_fn, measure_to_beat_fn
):

    start_step = measure_to_step_fn(start_measure)
    start_beat = measure_to_beat_fn(start_measure)

    note_mat_ = note_mat.copy()
    note_mat_[:, 0] -= start_step

    chord_mat_ = chord_mat.copy()
    chord_mat_[:, 0] -= start_beat

    return note_mat_, chord_mat_, start_measure


def chord_id_analysis(note_mat, chord_mat, step_to_beat_fn):
    """compute note to chord pointer"""
    chord_starts = chord_mat[:, 0]
    chord_ends = chord_mat[:, -1] + chord_starts

    note_onsets = note_mat[:, 0]
    onset_beats = step_to_beat_fn(note_onsets)

    chord_ids = np.where(
        np.logical_and(
            chord_starts <= onset_beats[:, np.newaxis], chord_ends
            > onset_beats[:, np.newaxis]
        )
    )  # (3, 4, 6, 7), (0, 1, 2, 3)
    assert (chord_ids[0] == np.arange(0, len(note_onsets))).all()

    return chord_ids[1]


def chord_tone_analysis(note_mat, chord_mat, chord_ids):
    # output col 1: normal chord tone, col 2: anticipation
    n_note = note_mat.shape[0]

    chords = chord_mat[chord_ids]
    pitches = note_mat[:, 1].astype(np.int64)

    is_reg_chord_tone = (chords[np.arange(0, n_note),
                                2 + pitches % 12] == 1).astype(np.int64)
    # todo: current wrong
    # if c_id + 1 not exceeding, i is last or i + 1 is next chord,
    # do anticipation
    next_c_exist_rows = chord_ids < chord_mat.shape[0] - 1

    last_note_in_chord = chord_ids[0 :-1] < chord_ids[1 :]
    last_note_in_chord = np.append(last_note_in_chord, True)

    anticipation_condition = np.logical_and(next_c_exist_rows, last_note_in_chord)
    anticipation_condition_inds = np.where(anticipation_condition)[0]
    # print(anticipation_condition_inds)
    # print(chord_ids)
    # print(len(chord_mat))
    # print(chord_ids)

    is_anticiptation = np.zeros(n_note, dtype=np.int64)
    # print(chord_ids[anticipation_condition_inds] + 1)
    # print(len(chords))
    is_anticiptation[anticipation_condition] = \
        chord_mat[chord_ids[anticipation_condition_inds] + 1,
                  2 + pitches[anticipation_condition] % 12] == 1
    # print(is_anticiptation)
    # print(chords[chord_ids[anticipation_condition_inds] + 1,
    #            2 + pitches[anticipation_condition] % 12])

    # is_anticipation_ = \
    #     np.logical_and(chord_ids[1:] == chord_ids[0: -1] + 1,
    #                    chords[np.arange(1, n_note),
    #                           2 + pitches[0: -1] % 12] == 1)
    # is_anticiptation[0: -1] = is_anticipation_.astype(np.int64)
    #
    is_anticiptation = \
        np.logical_and(np.logical_not(is_reg_chord_tone), is_anticiptation)

    is_chord_tone = np.logical_or(is_reg_chord_tone, is_anticiptation)

    tonal_chord_ids = chord_ids.copy()
    tonal_chord_ids[is_anticiptation] += 1
    return is_chord_tone, is_anticiptation, tonal_chord_ids


def bar_id_analysis(
    note_mat, chord_mat, chord_ids, is_anticipation, step_to_measure_fn,
    beat_to_measure_fn
):
    bar_id = step_to_measure_fn(note_mat[:, 0])
    tonal_bar_id = bar_id.copy()

    tonal_bar_id[is_anticipation] = \
        beat_to_measure_fn(chord_mat[chord_ids[is_anticipation] + 1, 0])
    return bar_id, tonal_bar_id


def preprocess_data(
    note_mat, chord_mat, start_measure, measure_to_step_fn, measure_to_beat_fn,
    step_to_beat_fn, step_to_measure_fn, beat_to_measure_fn
):

    note_mat, chord_mat, start_measure = \
        remove_offset(note_mat, chord_mat, start_measure,
                      measure_to_step_fn, measure_to_beat_fn)

    chord_ids = chord_id_analysis(note_mat, chord_mat, step_to_beat_fn)
    is_chord_tone, is_anticipation, tonal_chord_ids = \
        chord_tone_analysis(note_mat, chord_mat, chord_ids)
    bar_id, tonal_bar_id = \
        bar_id_analysis(note_mat, chord_mat, chord_ids,
                        is_anticipation, step_to_measure_fn, beat_to_measure_fn)

    onsets, pitches, durations = note_mat.T
    output_note_mat = \
        np.stack([onsets, pitches, durations, bar_id, tonal_bar_id,
                  chord_ids, tonal_chord_ids, is_chord_tone], -1)
    # print(output_note_mat)
    # print(chord_mat)  # for the last beat there is still a bug.
    # It could be the last note but there can be a following chord
    # print('------------')
    return output_note_mat, chord_mat, start_measure
