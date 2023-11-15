from .shortest_path_algo import find_tonal_shortest_paths
from .preprocess import preprocess_data
from .postprocess import check_path, assign_duration
"""
This file aims to provide api calls for tonal reduction
"""


class TrAlgo:
    def __init__(
        self,
        distance_factor=1.6,
        onset_factor=1.0,
        chord_factor=1.0,
        pitch_factor=1.0,
        duration_factor=1.0
    ):
        self.distance_factor = distance_factor
        self.onset_factor = onset_factor
        self.chord_factor = chord_factor
        self.pitch_factor = pitch_factor
        self.duration_factor = duration_factor

        self._note_mat = None
        self._chord_mat = None
        self._num_beat_per_measure = None
        self._num_step_per_beat = None

        self._start_measure = None

        self._reduction_mats = None
        self._report = None

    def preprocess_data(
        self, note_mat, chord_mat, start_measure, num_beat_per_measure,
        num_step_per_beat, require_convert
    ):
        def measure_to_step_fn(measure):
            return measure * num_beat_per_measure * num_step_per_beat

        def measure_to_beat_fn(measure):
            return measure * num_beat_per_measure

        def step_to_beat_fn(step):
            return step // num_step_per_beat

        def step_to_measure_fn(step):
            return step // (num_step_per_beat * num_beat_per_measure)

        def beat_to_measure_fn(beat):
            return beat // num_beat_per_measure

        if require_convert:
            note_mat, chord_mat, start_measure = preprocess_data(
                note_mat, chord_mat, start_measure, measure_to_step_fn,
                measure_to_beat_fn, step_to_beat_fn, step_to_measure_fn,
                beat_to_measure_fn
            )

        self.fill_data(
            note_mat, chord_mat, start_measure, num_beat_per_measure, num_step_per_beat
        )

    def fill_data(
        self, note_mat, chord_mat, start_measure, num_beat_per_measure,
        num_step_per_beat
    ):
        self._note_mat = note_mat
        self._chord_mat = chord_mat
        self._start_measure = start_measure
        self._num_beat_per_measure = num_beat_per_measure
        self._num_step_per_beat = num_step_per_beat

    def clear_data(self):
        self._note_mat = None
        self._chord_mat = None
        self._num_beat_per_measure = None
        self._num_step_per_beat = None

    def find_shortest_paths(self, num_path=1):
        return find_tonal_shortest_paths(
            self._note_mat, self._num_beat_per_measure, self._num_step_per_beat,
            num_path, self.distance_factor, self.onset_factor, self.chord_factor,
            self.pitch_factor, self.duration_factor
        )

    def path_to_chord_bins(self, path):
        return check_path(path, self._note_mat, self._chord_mat)

    def chord_bins_to_reduction_mat(self, chord_bin):
        return assign_duration(self._chord_mat, chord_bin, self._num_step_per_beat)

    def algo(self, num_path=1):
        # find top-k shortest paths and compute distance
        paths = self.find_shortest_paths(num_path)

        # postprocessing: put nodes into bins and delete nodes to avoid
        # exceeding note density of one node per quarter note

        chord_bins, reduction_report = \
            zip(*[self.path_to_chord_bins(path) for path in paths])
        reduction_mats = [self.chord_bins_to_reduction_mat(cb) for cb in chord_bins]
        self._reduction_mats = reduction_mats
        self._report = reduction_report

    def output(self, start_measure=None):

        start_measure = start_measure if start_measure is not None else \
            self._start_measure
        start_beat = start_measure * self._num_beat_per_measure
        start_step = start_beat * self._num_step_per_beat

        note_mat = self._note_mat.copy()
        note_mat[:, 0] += start_step

        chord_mat = self._chord_mat.copy()
        chord_mat[:, 0] += start_beat

        reduction_mats = self._reduction_mats.copy()
        for red_mat in reduction_mats:
            red_mat[:, 0] += start_step
            red_mat[:, 0] = red_mat[:, 0] // self._num_step_per_beat
            red_mat[:, 2] = red_mat[:, 2] // self._num_step_per_beat

        return note_mat, chord_mat, reduction_mats

    def run(
        self,
        note_mat,
        chord_mat,
        start_measure,
        num_beat_per_measure,
        num_step_per_beat,
        require_convert=True,
        num_path=1
    ):
        # print(note_mat)
        # print(chord_mat)
        # print(start_measure)
        self.preprocess_data(
            note_mat, chord_mat, start_measure, num_beat_per_measure, num_step_per_beat,
            require_convert
        )
        self.algo(num_path)
        note_mat, chord_mat, reduction_mats = self.output()
        return note_mat, chord_mat, reduction_mats
