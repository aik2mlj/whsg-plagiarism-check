import numpy as np
import pretty_midi as pm
from ..tonal_reduction_algo.main import TrAlgo
from .format_converter import *
from .key_analysis import *
from .phrase_analysis import *


KEY_TEMPLATE = np.array([1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0., 1.])
KEY_TEMPLATES = \
    np.stack([np.roll(KEY_TEMPLATE, step) for step in range(12)], 0)


class Pop909Song:

    def __init__(self, melody, chord, acc, phrase_label, num_beat_per_measure=4,
                 num_step_per_beat=4, song_name=None, clean_chord_unit=4):
        self.num_beat_per_measure = num_beat_per_measure
        self.num_step_per_beat = num_step_per_beat
        self.phrase_names = \
            np.array([pl['name'] for pl in phrase_label])
        self.phrase_types = \
            np.array([pl['type'] for pl in phrase_label])
        self.phrase_starts = \
            np.array([pl['start'] for pl in phrase_label])
        self.phrase_lengths = \
            np.array([pl['lgth'] for pl in phrase_label])
        self.num_phrases = len(phrase_label)

        self.song_name = song_name
        self.melody = melody
        self.chord = chord
        self.acc = acc
        self.clean_chord_unit = clean_chord_unit

        self.clean_chord()

        self.total_measure = self.compute_total_measure()
        self.total_beat = self.total_measure * self.num_beat_per_measure
        self.total_step = self.total_beat * self.num_step_per_beat

        self.regularize_chord()
        self.regularize_phrases()
        self.rough_chord = self.get_rough_chord()
        self.song_dict = self.create_song_level_dict()
        self.fill_phrase_level_slices()
        # self.keys = self.key_estimation()
        # self.melody_reductions = self.get_melody_reduction(num_reduction=1)

    def compute_total_measure(self):
        # propose candidates from phrase, chord and melody
        if self.acc is None:
            last_step = (self.melody[:, 0] + self.melody[:, 2]).max()
        else:
            last_step = max((self.melody[:, 0] + self.melody[:, 2]).max(),
                            (self.acc[:, 0] + self.acc[:, 2]).max())
        num_measure0 = \
            int(np.ceil(last_step / self.num_step_per_beat /
                        self.num_beat_per_measure))

        last_beat = (self.chord[:, 0] + self.chord[:, -1]).max()
        num_measure1 = int(np.ceil(last_beat / self.num_beat_per_measure))

        num_measure2 = sum(self.phrase_lengths)
        return max(num_measure0, num_measure1, num_measure2)

    def regularize_chord(self):
        chord = self.chord
        end_time = (self.chord[:, 0] + self.chord[:, -1]).max()
        fill_n_beat = self.total_beat - end_time
        if fill_n_beat == 0:
            return

        pad_durs = [self.clean_chord_unit] * (fill_n_beat // self.clean_chord_unit)
        if fill_n_beat - sum(pad_durs) > 0:
            pad_durs = [fill_n_beat - sum(pad_durs)] + pad_durs
        for d in pad_durs:
            stack_chord = chord[-1].copy()
            stack_chord[0] = chord[-1, 0] + chord[-1, -1]
            stack_chord[-1] = d

            chord = np.concatenate([chord, stack_chord[np.newaxis, :]], 0)
        self.chord = chord

    def regularize_phrases(self):
        original_phrase_length = sum(self.phrase_lengths)
        if self.total_measure == original_phrase_length:
            return

        extra_phrase_length = self.total_measure - original_phrase_length
        extra_phrase_name = 'z' + str(extra_phrase_length)

        self.phrase_names = np.append(self.phrase_names, extra_phrase_name)
        self.phrase_types = np.append(self.phrase_types, 'z')
        self.phrase_lengths = np.append(self.phrase_lengths,
                                        extra_phrase_length)
        self.phrase_starts = np.append(self.phrase_starts,
                                       original_phrase_length)

    def clean_chord(self):
        chord = self.chord
        unit = self.clean_chord_unit

        new_chords = []
        n_chord = len(chord)
        for i in range(n_chord):
            chord_start = chord[i, 0]
            chord_dur = chord[i, -1]

            cum_dur = 0
            s = chord_start
            while cum_dur < chord_dur:
                d = min(unit - s % unit, chord_dur - cum_dur)
                c = chord[i].copy()
                c[0] = s
                c[-1] = d
                new_chords.append(c)

                s = s + d
                cum_dur += d

        new_chords = np.stack(new_chords, 0)
        self.chord = new_chords

    def step_to_beat(self, step):
        return step // self.num_step_per_beat

    def beat_to_step(self, beat):
        start_step = beat * self.num_step_per_beat
        end_step = start_step + self.num_step_per_beat
        return start_step, end_step

    def beat_to_measure(self, beat):
        return beat // self.num_beat_per_measure

    def measure_to_beat(self, measure):
        start_beat = measure * self.num_beat_per_measure
        end_beat = start_beat + self.num_beat_per_measure
        return start_beat, end_beat

    def step_to_measure(self, step):
        beat = self.step_to_beat(step)
        measure = self.beat_to_measure(beat)
        return measure

    def measure_to_step(self, measure):
        start_beat, end_beat = self.measure_to_beat(measure)
        start_step, _ = self.beat_to_step(start_beat)
        end_step, _ = self.beat_to_step(end_beat)
        return start_step, end_step

    def measure_to_phrase(self, measure):
        phrase = np.where(measure >= self.phrase_starts)[0][0]
        return phrase

    def phrase_to_measure(self, phrase):
        start_measure = self.phrase_starts[phrase]
        end_measure = self.phrase_lengths[phrase] + start_measure
        return start_measure, end_measure

    def beat_to_phrase(self, beat):
        measure = self.beat_to_measure(beat)
        phrase = self.measure_to_phrase(measure)
        return phrase

    def phrase_to_beat(self, phrase):
        start_measure, end_measure = self.phrase_to_measure(phrase)
        start_beat, _ = self.measure_to_beat(start_measure)
        end_beat, _ = self.measure_to_beat(end_measure)
        return start_beat, end_beat

    def step_to_phrase(self, step):
        beat = self.step_to_beat(step)
        phrase = self.beat_to_phrase(beat)
        return phrase

    def phrase_to_step(self, phrase):
        start_beat, end_beat = self.phrase_to_beat(phrase)
        start_step, _ = self.beat_to_step(start_beat)
        end_step, _ = self.beat_to_step(end_beat)
        return start_step, end_step

    def create_a_measure_level_dict(self, measure_no):
        start_beat, end_beat = self.measure_to_beat(measure_no)
        measure_dict = {
            'measure_no': measure_no,
            'start_beat': start_beat,
            'end_beat': end_beat,
            'measure_length': end_beat - start_beat,
            'mel_natural_slice': None,
            'mel_tonal_slice': None,
            'chd_slice': None,
            'rough_chd_slice': None,
            'cwords': []
        }
        return measure_dict

    def create_a_phrase_level_dict(self, phrase_id):
        start_measure = self.phrase_starts[phrase_id]
        phrase_length = self.phrase_lengths[phrase_id]
        end_measure = start_measure + phrase_length
        phrase_dict = {
            'phrase_name': self.phrase_names[phrase_id],
            'phrase_type': self.phrase_types[phrase_id],
            'phrase_length': self.phrase_lengths[phrase_id],
            'start_measure': start_measure,
            'end_measure': end_measure,
            'length': phrase_length,
            'mel_natural_slice': None,
            'mel_tonal_slice': None,
            'chd_slice': None,
            'rough_chd_slice': None,
            'measures': [self.create_a_measure_level_dict(measure_no)
                         for measure_no in range(start_measure, end_measure)]
        }
        return phrase_dict

    def create_song_level_dict(self):
        song_dict = {
            'song_name': self.song_name,
            'total_phrase': self.num_phrases,
            'total_measure': self.total_measure,
            'total_beat': self.total_beat,
            'total_step': self.total_step,
            'phrases': [self.create_a_phrase_level_dict(phrase_id)
                        for phrase_id in range(self.num_phrases)]
        }
        return song_dict

    def _fill_phrase_level_mel_slices(self):
        n_note = self.melody.shape[0]
        onset_beats = self.step_to_beat(self.melody[:, 0])

        current_ind = 0
        for phrase_id, phrase in enumerate(self.song_dict['phrases']):
            start_beat, end_beat = self.phrase_to_beat(phrase_id)
            for i in range(current_ind, n_note):
                if onset_beats[i] >= end_beat:
                    phrase[f'mel_slice'] = slice(current_ind, i)
                    current_ind = i
                    break
            else:
                phrase[f'mel_slice'] = slice(current_ind, n_note)
                current_ind = n_note

    def _fill_phrase_level_chd_slices(self, rough_chord=False):
        chord = self.rough_chord if rough_chord else self.chord
        slice_name = 'rough_chd_slice' if rough_chord else 'chd_slice'
        n_chord = chord.shape[0]
        current_ind = 0
        for phrase_id, phrase in enumerate(self.song_dict['phrases']):
            start_beat, end_beat = self.phrase_to_beat(phrase_id)
            for i in range(current_ind, n_chord):
                if chord[i, 0] >= end_beat:
                    phrase[slice_name] = slice(current_ind, i)
                    current_ind = i
                    break
            else:
                phrase[slice_name] = slice(current_ind, n_chord)
                current_ind = n_chord

    def fill_phrase_level_slices(self):
        self._fill_phrase_level_mel_slices()
        self._fill_phrase_level_chd_slices(rough_chord=False)
        self._fill_phrase_level_chd_slices(rough_chord=True)

    def get_rough_chord(self):
        def chroma1_in_chroma_2(chroma1, chroma2):
            return (chroma2[chroma1 != 0] == 1).all()

        def share_chroma(chroma1, chroma2):
            return np.count_nonzero(np.logical_and(chroma1, chroma2)) >= 3

        """
        Within one measure, consider merging chords if two chords share the same root and:
        1) one chord includes the other
        2) two chords share more than three notes
        """
        rough_chord = []

        i = 0
        while i < len(self.chord):
            c_i = self.chord[i]
            onset_i, root_i, chroma_i, bass_i, duration_i = c_i[0], c_i[1], c_i[2: 14], c_i[14], c_i[15]

            root, chroma, bass, duration = root_i, chroma_i, bass_i, duration_i
            j = i + 1

            while j < len(self.chord) and duration < self.clean_chord_unit:
                c_j = self.chord[j]
                onset_j, root_j, chroma_j, bass_j, duration_j = c_j[0], c_j[1], c_j[2: 14], c_j[14], c_j[15]
                if onset_j // self.clean_chord_unit != onset_i // self.clean_chord_unit:
                    break
                if root_i == root_j or bass_i == bass_j:
                    if chroma1_in_chroma_2(chroma_i, chroma_j):
                        # chroma i in chroma j, use chord_j
                        root, chroma, bass = root_j, chroma_j, bass_j
                        duration += duration_j
                        j += 1

                    elif chroma1_in_chroma_2(chroma_j, chroma_i):
                        # chroma j in chroma i, use chord_i
                        duration += duration_j
                        j += 1
                    elif share_chroma(chroma_i, chroma_j):
                        # share more than three chord tone
                        duration += duration_j
                        j += 1
                    else:
                        break
                else:
                    break

            rough_chord.append(
                np.concatenate([np.array([onset_i, root]), chroma, np.array([bass, duration])]))
            i = j

        rough_chord = np.stack(rough_chord, 0).astype(np.int64)
        return rough_chord

    def get_melody_reduction(self, num_reduction=1):
        tr_algo = TrAlgo()

        nbpm = self.num_beat_per_measure
        nspb = self.num_step_per_beat

        reductions = [[] for _ in range(num_reduction)]

        for phrase in self.song_dict['phrases']:
            mel_slice = phrase['mel_slice']
            chd_slice = phrase['rough_chd_slice']

            note_mat = self.melody[mel_slice].copy()
            chord_mat = self.rough_chord[chd_slice].copy()

            start_measure = phrase['start_measure']

            _, _, reduction_mats = \
                tr_algo.run(note_mat, chord_mat, start_measure, self.num_beat_per_measure, self.num_step_per_beat,
                            require_convert=True, num_path=1)
            for i in range(num_reduction):
                reductions[i].append(reduction_mats[i])

        reductions = [np.concatenate(reductions[i], 0) for i in range(num_reduction)]

        return reductions

    def get_pitch_rhythm_contours(self):
        pr_array = mono_note_matrix_to_pr_array(self.melody, self.total_step)
        pr_contour = pr_array_to_pr_contour(pr_array)

        rough_p_contour = extract_pitch_contour(pr_contour, self.num_step_per_beat,
                                                stride=self.num_beat_per_measure / 2.)
        detailed_p_contour = extract_pitch_contour(pr_contour, self.num_beat_per_measure,
                                                   stride=1.)

        rhythm = pr_array_to_rhythm(pr_array)
        rough_r_contour = extract_rhythm_intensity(rhythm, self.num_step_per_beat,
                                                   stride=self.num_beat_per_measure / 2., quantization_bin=4)
        detailed_r_contour = extract_rhythm_intensity(rhythm, self.num_step_per_beat,
                                                      stride=1., quantization_bin=3)

        return rough_p_contour, detailed_p_contour, rough_r_contour, detailed_r_contour

    def analyze(self):
        pr_array = mono_note_matrix_to_pr_array(self.melody, self.total_step)
        pr_mel = note_matrix_to_piano_roll(self.melody, self.total_step)

        pr_chord = chord_mat_to_chord_roll(self.chord, self.total_beat)

        pr_rough_chord = chord_mat_to_chord_roll(self.rough_chord, self.total_beat)

        key_roll = get_key_roll(pr_mel, pr_chord, self.phrase_starts, self.phrase_lengths, self.total_measure,
                                self.num_beat_per_measure, self.num_step_per_beat)

        # rough_p_contour, detailed_p_contour, rough_r_contour, detailed_r_contour = self.get_pitch_rhythm_contours()

        phrase_roll = phrase_to_phrase_roll(self.phrase_starts, self.phrase_lengths,
                                            self.phrase_types, self.total_measure)

        reduction = self.get_melody_reduction(num_reduction=1)[0]

        reduction_roll = note_matrix_to_piano_roll(reduction, self.total_beat)

        min_pitch, max_pitch = self.melody[:, 1].min(), self.melody[:, 1].max()

        acc_roll = note_matrix_to_piano_roll(self.acc, self.total_step)

        analysis = {
            # meta-level
            "name": self.song_name, "nbpm": self.num_beat_per_measure, "nspb": self.num_step_per_beat,
            "min_pitch": min_pitch, "max_pitch": max_pitch,
            # measure-level
            "phrase_roll": phrase_roll, "key_roll": key_roll,
            # beat-level
            "mel_reduction": reduction_roll, "chord_roll": pr_chord, "rough_chord_roll": pr_rough_chord,
            # step-level
            "mel_roll": pr_mel,
            # # rough style
            # "rough_p_contour": rough_p_contour, "rough_r_contour": rough_r_contour,
            # # detailed style
            # "detailed_p_contour": detailed_p_contour, "detailed_r_contour": detailed_r_contour,
            # acc-level
            "acc_roll": acc_roll
        }

        return analysis









