from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from .utils.read_file import read_data
from .train_valid_split import load_split_file
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 30})

TRIPLE_METER_SONG = [
    34, 62, 102, 107, 152, 173, 176, 203, 215, 231, 254, 280, 307, 328, 369, 584, 592,
    653, 654, 662, 744, 749, 756, 770, 799, 843, 869, 872, 887
]

PROJECT_PATH = os.path.join(os.path.dirname(__file__), '..')

DATASET_PATH = os.path.join(PROJECT_PATH, 'data', 'original_shuqi_data')
ACC_DATASET_PATH = os.path.join(PROJECT_PATH, 'data', 'matched_pop909_acc')

LABEL_SOURCE = np.load(
    os.path.join(PROJECT_PATH, 'data', 'original_shuqi_data', 'label_source.npy')
)

SPLIT_FILE_PATH = os.path.join(PROJECT_PATH, 'data', 'pop909_mel', 'split.npz')


def expand_roll(roll, unit=4, contain_onset=False):
    # roll: (Channel, T, H) -> (Channel, T * unit, H)
    n_channel, length, height = roll.shape

    expanded_roll = roll.repeat(unit, axis=1)
    if contain_onset:
        expanded_roll = expanded_roll.reshape((n_channel, length, unit, height))
        expanded_roll[1 :: 2, :, 1 :] = np.maximum(
            expanded_roll[:: 2, :, 1 :], expanded_roll[1 :: 2, :, 1 :]
        )

        expanded_roll[:: 2, :, 1 :] = 0
        expanded_roll = expanded_roll.reshape((n_channel, length * unit, height))
    return expanded_roll


def analysis_to_key(analysis, unit, nbpm=None, nspb=None):
    assert unit in ['measure', 'beat', 'step']
    key_roll = analysis['key_roll']

    if unit == 'measure':
        pass
    elif unit == 'beat':
        key_roll = expand_roll(key_roll, unit=nbpm)
    else:
        key_roll = expand_roll(key_roll, unit=nbpm * nspb)
    return key_roll


def analysis_to_phrase(analysis, unit, nbpm=None, nspb=None):
    assert unit in ['measure', 'beat', 'step']
    phrase_roll = analysis['phrase_roll']

    if unit == 'measure':
        pass
    elif unit == 'beat':
        phrase_roll = expand_roll(phrase_roll, unit=nbpm)
    else:
        phrase_roll = expand_roll(phrase_roll, unit=nbpm * nspb)
    return phrase_roll


def analysis_to_reduction(analysis, unit, nspb=None):
    assert unit in ['beat', 'step']
    reduction = analysis['mel_reduction']
    if unit == 'beat':
        pass
    else:
        reduction = expand_roll(reduction, unit=nspb, contain_onset=True)
    return reduction


def analysis_to_chord(analysis, unit, nspb=None):
    assert unit in ['beat', 'step']
    reduction = analysis['chord_roll']
    if unit == 'beat':
        pass
    else:
        reduction = expand_roll(reduction, unit=nspb, contain_onset=True)
    return reduction


class HierarchicalDatasetBase(Dataset):
    def __init__(self, analyses, shift_high=0, shift_low=0, max_l=128, max_h=128):
        super(HierarchicalDatasetBase, self).__init__()
        self.shift_high = shift_high
        self.shift_low = shift_low
        self.max_l = max_l
        self.max_h = max_h

        self.min_pitches = [analysis['min_pitch'] for analysis in analyses]
        self.max_pitches = [analysis['max_pitch'] for analysis in analyses]
        self.nbpms = [analysis['nbpm'] for analysis in analyses]
        self.nspbs = [analysis['nspb'] for analysis in analyses]

        self.lengths = None
        self.start_ids_per_song = None
        self.indices = None

    def _song_id_to_indices(self):
        if self.start_ids_per_song is None:
            raise Exception("Value not filled.")
        return np.concatenate(
            [
                np.stack(
                    [np.ones(len(start_ids), dtype=np.int64) * song_id, start_ids], -1
                ) for song_id, start_ids in enumerate(self.start_ids_per_song)
            ], 0
        )

    def __len__(self):
        return len(self.indices) * (self.shift_high - self.shift_low + 1)

    def __getitem__(self, item):
        no = item // (self.shift_high - self.shift_low + 1)
        shift = item % (self.shift_high - self.shift_low + 1) + self.shift_low

        song_id, start_id = self.indices[no]
        return self.get_data_sample(song_id, start_id, shift)

    def get_data_sample(self, song_id, start_id, shift):
        raise NotImplementedError


class N2KeyDataset(HierarchicalDatasetBase):
    def __init__(self, analyses, shift_high=0, shift_low=0, max_l=256, max_h=12):
        super(N2KeyDataset,
              self).__init__(analyses, shift_high, shift_low, max_l, max_h)
        assert max_l >= 210
        self.key_rolls = [analysis['key_roll'] for analysis in analyses]
        self.phrase_rolls = [
            analysis['phrase_roll'][:, :, np.newaxis] for analysis in analyses
        ]

        self.lengths = np.array([roll.shape[1] for roll in self.key_rolls])

        self.start_ids_per_song = [
            np.arange(0, 1, dtype=np.int64) for _ in range(len(self.lengths))
        ]

        self.indices = self._song_id_to_indices()

    def get_data_sample(self, song_id, start_id, shift):
        # segment target data
        key_roll = self.key_rolls[song_id]  # (2, 128, 12)
        phrase_roll = self.phrase_rolls[song_id]

        # pitch augmentation
        key_roll = np.roll(key_roll, shift=shift, axis=-1)
        actual_l = key_roll.shape[1]

        img = np.zeros((8, self.max_l, self.max_h), dtype=np.float32)
        img[0 : 2, 0 : actual_l, 0 : 12] = key_roll
        img[2 : 8, 0 : actual_l] = phrase_roll

        return img[:, :, 0 : self.max_h]

    def show(self, item, show_img=True):
        sample = self[item]
        print(f"data_sample[{item}].shape = {sample.shape})")
        titles = ['key', 'phrase0-1', 'phrase2-3', 'phrase4-5']

        if show_img:
            fig, axs = plt.subplots(4, 1, figsize=(10, 30))
            for i in range(4):
                img = sample[2 * i : 2 * i + 2]
                img = np.pad(img, pad_width=((0, 1), (0, 0), (0, 0)), mode='constant')
                img = img.transpose((2, 1, 0))
                axs[i].imshow(img, origin='lower', aspect='auto')
                axs[i].title.set_text(titles[i])
            plt.show()


class Key2RedDataset(HierarchicalDatasetBase):
    def __init__(self, analyses, shift_high=0, shift_low=0, max_l=128, max_h=128):
        super(Key2RedDataset,
              self).__init__(analyses, shift_high, shift_low, max_l, max_h)
        self.key_rolls = [analysis['key_roll'] for analysis in analyses]
        self.expand_key_rolls()

        self.phrase_rolls = [
            analysis['phrase_roll'][:, :, np.newaxis] for analysis in analyses
        ]
        self.expand_phrase_rolls()

        self.chord_rolls = [analysis['rough_chord_roll'] for analysis in analyses]
        self.reductions = [analysis['mel_reduction'] for analysis in analyses]

        self.lengths = np.array([red.shape[1] for red in self.reductions])

        self.start_ids_per_song = [
            np.arange(0, lgth - self.max_l // 2, nbpm, dtype=np.int64)
            for lgth, nbpm in zip(self.lengths, self.nbpms)
        ]

        self.indices = self._song_id_to_indices()

    def expand_key_rolls(self):
        self.key_rolls = [
            expand_roll(roll, nbpm) for roll, nbpm in zip(self.key_rolls, self.nbpms)
        ]

    def expand_phrase_rolls(self):
        self.phrase_rolls = [
            expand_roll(roll, nbpm)
            for roll, nbpm in zip(self.phrase_rolls, self.nbpms)
        ]

    def get_data_sample(self, song_id, start_id, shift):
        # segment target data
        min_pitch, max_pitch = self.min_pitches[song_id] + shift, self.max_pitches[
            song_id] + shift
        if min_pitch < 48:
            shift = shift + 12 * int(np.ceil((48 - min_pitch) / 12))
        elif max_pitch > 108:
            shift = shift - 12 * int(np.ceil((max_pitch - 108) / 12))

        key_roll = self.key_rolls[song_id][:, start_id : start_id + self.max_l
                                          ]  # (2, 128, 12)
        phrase_roll = self.phrase_rolls[song_id][:, start_id : start_id + self.max_l
                                                ]  # (6, 128, 1)
        red_roll = self.reductions[song_id][:, start_id : start_id + self.max_l
                                           ]  # (2, 128, 128)
        chord_roll = self.chord_rolls[song_id][:, start_id : start_id + self.max_l
                                              ]  # (6, 128, 12)

        # pitch augmentation
        key_roll = np.roll(key_roll, shift=shift, axis=-1)
        red_roll = np.roll(red_roll, shift=shift, axis=-1)
        chord_roll = np.roll(chord_roll, shift=shift, axis=-1)
        actual_l = red_roll.shape[1]

        img = np.zeros((10, self.max_l, 132), dtype=np.float32)
        img[0 : 2, 0 : actual_l, 0 : 128] = red_roll
        img[0 : 2, 0 : actual_l, 36 : 48] = chord_roll[2 : 4]
        img[0 : 2, 0 : actual_l, 24 : 36] = chord_roll[4 : 6]

        img[4 : 10, 0 : actual_l] = phrase_roll

        img = img.reshape((10, self.max_l, 11, 12))
        img[2 : 4, 0 : actual_l] = key_roll[:, :, np.newaxis]
        img = img.reshape((10, self.max_l, 132))[:, :, 0 : self.max_h]

        # preparing for condition image
        cond_img = -np.ones((10, 108, 128), dtype=np.float32)
        nbpm = self.nbpms[song_id]

        start_measure = start_id // nbpm
        if start_measure == 0:
            return img, cond_img

        measure_phrase_roll = self.phrase_rolls[song_id][:, :: nbpm, 0]
        phrase_type_importance = phrase_roll[:, :, 0].sum(-1)  # (6, )

        measure_importance = phrase_type_importance[measure_phrase_roll.argmax(0)
                                                   ][0 : start_measure]

        cond_scope = 8  # 8 measures
        interval = 4  # 4 beats

        slices = []
        n_cond = np.random.choice(np.arange(0, 4), p=np.array([0.1, 0.1, 0.2, 0.6]))

        for i in range(n_cond):
            # go through the dataset and compute accumulate distance
            acc_measure_importance = np.array(
                [
                    measure_importance[i : i + cond_scope].sum()
                    for i in range(len(measure_importance))
                ]
            )
            measure = acc_measure_importance.argmax()
            if acc_measure_importance[measure] < 0:
                break
            measure_importance[measure : measure + cond_scope] = -1
            slices.append((measure, min(start_measure, measure + cond_scope)))

        cur_step = 0
        for slc in slices:
            seg_start_id, seg_end_id = slc[0] * nbpm, slc[1] * nbpm
            seg_lgth = seg_end_id - seg_start_id
            cond_seg = self.get_cond_img(song_id, seg_start_id, seg_end_id, shift)
            cond_img[:, cur_step : cur_step + seg_lgth] = cond_seg
            cur_step += seg_lgth + interval

        return img, cond_img

    def get_cond_img(self, song_id, start_id, end_id, shift):
        key_roll = self.key_rolls[song_id][:, start_id : end_id]  # (2, 128, 12)
        phrase_roll = self.phrase_rolls[song_id][:, start_id : end_id]  # (6, 128, 1)
        red_roll = self.reductions[song_id][:, start_id : end_id]  # (2, 128, 128)
        chord_roll = self.chord_rolls[song_id][:, start_id : end_id]  # (6, 128, 12)

        # pitch augmentation
        key_roll = np.roll(key_roll, shift=shift, axis=-1)
        red_roll = np.roll(red_roll, shift=shift, axis=-1)
        chord_roll = np.roll(chord_roll, shift=shift, axis=-1)
        actual_l = red_roll.shape[1]

        img = np.zeros((10, end_id - start_id, 132), dtype=np.float32)
        img[0 : 2, 0 : actual_l, 0 : 128] = red_roll
        img[0 : 2, 0 : actual_l, 36 : 48] = chord_roll[2 : 4]
        img[0 : 2, 0 : actual_l, 24 : 36] = chord_roll[4 : 6]

        img[4 : 10, 0 : actual_l] = phrase_roll

        img = img.reshape((10, end_id - start_id, 11, 12))
        img[2 : 4, 0 : actual_l] = key_roll[:, :, np.newaxis]
        img = img.reshape((10, end_id - start_id, 132))
        return img[:, :, 0 : self.max_h]

    def show(self, item, show_img=True):
        sample, cond_sample = self[item]

        print(
            f"sample[{item}].shape = {sample.shape}), cond_sample[{item}].shape = {cond_sample.shape}"
        )
        titles = ['red+rough_chd', 'key', 'phrase0-1', 'phrase2-3', 'phrase4-5']

        if show_img:
            fig, axs = plt.subplots(5, 2, figsize=(20, 30))
            for i in range(5):
                img = sample[2 * i : 2 * i + 2]
                img = np.pad(img, pad_width=((0, 1), (0, 0), (0, 0)), mode='constant')
                img = img.transpose((2, 1, 0))
                axs[i, 0].imshow(img, origin='lower', aspect='auto')
                axs[i, 0].title.set_text(titles[i])

                cond_img = cond_sample[2 * i : 2 * i + 2]
                cond_img = np.pad(
                    cond_img, pad_width=((0, 1), (0, 0), (0, 0)), mode='constant'
                )
                cond_img[2][cond_img[0] < 0] = 1
                cond_img[cond_img < 0] = 0
                cond_img = cond_img.transpose((2, 1, 0))

                axs[i, 1].imshow(cond_img, origin='lower', aspect='auto')
            plt.show()

    def get_whole_song(self, song_id):
        min_pitch, max_pitch = self.min_pitches[song_id], self.max_pitches[song_id]
        if min_pitch < 48:
            shift = 12 * int(np.ceil((48 - min_pitch) / 12))
        elif max_pitch > 108:
            shift = -12 * int(np.ceil((max_pitch - 108) / 12))
        else:
            shift = 0

        key_roll = self.key_rolls[song_id]
        phrase_roll = self.phrase_rolls[song_id]
        red_roll = self.reductions[song_id]
        chord_roll = self.chord_rolls[song_id]

        # pitch augmentation
        key_roll = np.roll(key_roll, shift=shift, axis=-1)
        red_roll = np.roll(red_roll, shift=shift, axis=-1)
        chord_roll = np.roll(chord_roll, shift=shift, axis=-1)
        actual_l = red_roll.shape[1]
        target_l = max(
            int(np.ceil(red_roll.shape[1] / (self.max_l // 2)) * self.max_l // 2),
            self.max_l
        )

        img = np.zeros((10, target_l, 132), dtype=np.float32)
        img[0 : 2, 0 : actual_l, 0 : 128] = red_roll
        img[0 : 2, 0 : actual_l, 36 : 48] = chord_roll[2 : 4]
        img[0 : 2, 0 : actual_l, 24 : 36] = chord_roll[4 : 6]

        img[4 : 10, 0 : actual_l] = phrase_roll

        img = img.reshape((10, target_l, 11, 12))
        img[2 : 4, 0 : actual_l] = key_roll[:, :, np.newaxis]
        img = img.reshape((10, target_l, 132))

        return img[:, :, 0 : self.max_h]


class Red2MelDataset(HierarchicalDatasetBase):
    def __init__(self, analyses, shift_high=0, shift_low=0, max_l=128, max_h=128):
        super(Red2MelDataset,
              self).__init__(analyses, shift_high, shift_low, max_l, max_h)
        self.key_rolls = [analysis['key_roll'] for analysis in analyses]
        self.expand_key_rolls()

        self.phrase_rolls = [
            analysis['phrase_roll'][:, :, np.newaxis] for analysis in analyses
        ]
        self.expand_phrase_rolls()

        self.rough_chord_rolls = [analysis['rough_chord_roll'] for analysis in analyses]
        self.expand_rough_chord_rolls()

        self.reductions = [analysis['mel_reduction'] for analysis in analyses]
        self.expand_reductions()

        self.mel_rolls = [analysis['mel_roll'] for analysis in analyses]
        self.chord_rolls = [analysis['chord_roll'] for analysis in analyses]
        self.expand_chord_rolls()

        self.lengths = np.array([mel.shape[1] for mel in self.mel_rolls])

        self.start_ids_per_song = [
            np.arange(0, lgth - self.max_l // 2, nspb * nbpm, dtype=np.int64)
            for lgth, nbpm, nspb in zip(self.lengths, self.nbpms, self.nspbs)
        ]

        self.indices = self._song_id_to_indices()

    def expand_key_rolls(self):
        self.key_rolls = [
            expand_roll(roll, nbpm * nspb)
            for roll, nbpm, nspb in zip(self.key_rolls, self.nbpms, self.nspbs)
        ]

    def expand_phrase_rolls(self):
        self.phrase_rolls = [
            expand_roll(roll, nbpm * nspb)
            for roll, nbpm, nspb in zip(self.phrase_rolls, self.nbpms, self.nspbs)
        ]

    def expand_rough_chord_rolls(self):
        self.rough_chord_rolls = [
            expand_roll(roll, nspb, contain_onset=True)
            for roll, nspb in zip(self.rough_chord_rolls, self.nspbs)
        ]

    def expand_chord_rolls(self):
        self.chord_rolls = [
            expand_roll(roll, nspb, contain_onset=True)
            for roll, nspb in zip(self.chord_rolls, self.nspbs)
        ]

    def expand_reductions(self):
        self.reductions = [
            expand_roll(roll, nspb, contain_onset=True)
            for roll, nspb in zip(self.reductions, self.nspbs)
        ]

    def get_data_sample(self, song_id, start_id, shift):
        # segment target data
        min_pitch, max_pitch = self.min_pitches[song_id] + shift, self.max_pitches[
            song_id] + shift
        if min_pitch < 48:
            shift = shift + 12 * int(np.ceil((48 - min_pitch) / 12))
        elif max_pitch > 108:
            shift = shift - 12 * int(np.ceil((max_pitch - 108) / 12))

        key_roll = self.key_rolls[song_id][:, start_id : start_id + self.max_l
                                          ]  # (2, 128, 12)
        phrase_roll = self.phrase_rolls[song_id][:, start_id : start_id + self.max_l
                                                ]  # (6, 128, 1)
        red_roll = self.reductions[song_id][:, start_id : start_id + self.max_l
                                           ]  # (2, 128, 128)
        rough_chord_roll = self.rough_chord_rolls[song_id][:, start_id : start_id +
                                                           self.max_l]  # (6, 128, 12)
        mel_roll = self.mel_rolls[song_id][:, start_id : start_id + self.max_l]
        chord_roll = self.chord_rolls[song_id][:, start_id : start_id + self.max_l]

        # pitch augmentation
        key_roll = np.roll(key_roll, shift=shift, axis=-1)
        red_roll = np.roll(red_roll, shift=shift, axis=-1)
        rough_chord_roll = np.roll(rough_chord_roll, shift=shift, axis=-1)
        mel_roll = np.roll(mel_roll, shift=shift, axis=-1)
        chord_roll = np.roll(chord_roll, shift=shift, axis=-1)
        actual_l = mel_roll.shape[1]

        img = np.zeros((12, self.max_l, 132), dtype=np.float32)

        img[0 : 2, 0 : actual_l, 0 : 128] = mel_roll
        img[0 : 2, 0 : actual_l, 36 : 48] = chord_roll[2 : 4]
        img[0 : 2, 0 : actual_l, 24 : 36] = chord_roll[4 : 6]

        img[2 : 4, 0 : actual_l, 0 : 128] = red_roll
        img[2 : 4, 0 : actual_l, 36 : 48] = rough_chord_roll[2 : 4]
        img[2 : 4, 0 : actual_l, 24 : 36] = rough_chord_roll[4 : 6]

        img[6 : 12, 0 : actual_l] = phrase_roll

        img = img.reshape((12, self.max_l, 11, 12))
        img[4 : 6, 0 : actual_l] = key_roll[:, :, np.newaxis]
        img = img.reshape((12, self.max_l, 132))[:, :, 0 : self.max_h]

        # preparing for condition image
        cond_img = -np.ones((12, 108, 128), dtype=np.float32)
        nbpm = self.nbpms[song_id]
        nspb = self.nspbs[song_id]

        start_measure = start_id // (nbpm * nspb)
        if start_measure == 0:
            return img, cond_img

        measure_phrase_roll = self.phrase_rolls[song_id][:, :: nbpm * nspb, 0]
        phrase_type_importance = phrase_roll[:, :, 0].sum(-1)  # (6, )

        measure_importance = phrase_type_importance[measure_phrase_roll.argmax(0)
                                                   ][0 : start_measure]

        cond_scope = 6  # 8 measures
        interval = 4  # 4 steps

        slices = []
        n_cond = np.random.choice(np.arange(0, 2), p=np.array([0.1, 0.9]))

        for i in range(n_cond):
            # go through the dataset and compute accumulate distance
            acc_measure_importance = np.array(
                [
                    measure_importance[i : i + cond_scope].sum()
                    for i in range(len(measure_importance))
                ]
            )
            measure = acc_measure_importance.argmax()
            if acc_measure_importance[measure] < 0:
                break
            measure_importance[measure : measure + cond_scope] = -1
            slices.append((measure, min(start_measure, measure + cond_scope)))

        cur_step = 0
        for slc in slices:
            seg_start_id, seg_end_id = slc[0] * nbpm * nspb, slc[1] * nbpm * nspb
            seg_lgth = seg_end_id - seg_start_id
            cond_seg = self.get_cond_img(song_id, seg_start_id, seg_end_id, shift)
            cond_img[:, cur_step : cur_step + seg_lgth] = cond_seg
            cur_step += seg_lgth + interval

        return img, cond_img

    def get_cond_img(self, song_id, start_id, end_id, shift):
        key_roll = self.key_rolls[song_id][:, start_id : end_id]  # (2, 128, 12)
        phrase_roll = self.phrase_rolls[song_id][:, start_id : end_id]  # (6, 128, 1)
        red_roll = self.reductions[song_id][:, start_id : end_id]  # (2, 128, 128)
        rough_chord_roll = self.rough_chord_rolls[song_id][:, start_id : end_id
                                                          ]  # (6, 128, 12)
        mel_roll = self.mel_rolls[song_id][:, start_id : end_id]
        chord_roll = self.chord_rolls[song_id][:, start_id : end_id]

        # pitch augmentation
        key_roll = np.roll(key_roll, shift=shift, axis=-1)
        red_roll = np.roll(red_roll, shift=shift, axis=-1)
        rough_chord_roll = np.roll(rough_chord_roll, shift=shift, axis=-1)
        mel_roll = np.roll(mel_roll, shift=shift, axis=-1)
        chord_roll = np.roll(chord_roll, shift=shift, axis=-1)
        actual_l = mel_roll.shape[1]

        img = np.zeros((12, end_id - start_id, 132), dtype=np.float32)
        img[0 : 2, 0 : actual_l, 0 : 128] = mel_roll
        img[0 : 2, 0 : actual_l, 36 : 48] = chord_roll[2 : 4]
        img[0 : 2, 0 : actual_l, 24 : 36] = chord_roll[4 : 6]

        img[2 : 4, 0 : actual_l, 0 : 128] = red_roll
        img[2 : 4, 0 : actual_l, 36 : 48] = rough_chord_roll[2 : 4]
        img[2 : 4, 0 : actual_l, 24 : 36] = rough_chord_roll[4 : 6]

        img[6 : 12, 0 : actual_l] = phrase_roll

        img = img.reshape((12, end_id - start_id, 11, 12))
        img[4 : 6, 0 : actual_l] = key_roll[:, :, np.newaxis]
        img = img.reshape((12, end_id - start_id, 132))[:, :, 0 : self.max_h]
        return img

    def show(self, item, show_img=True):
        sample, cond_sample = self[item]
        print(
            f"sample[{item}].shape = {sample.shape}), cond_sample[{item}].shape = {cond_sample.shape}"
        )
        titles = [
            'mel+chd', 'mel+rough_chd', 'key', 'phrase0-1', 'phrase2-3', 'phrase4-5'
        ]

        if show_img:
            fig, axs = plt.subplots(6, 2, figsize=(20, 40))
            for i in range(6):
                img = sample[2 * i : 2 * i + 2]
                img = np.pad(img, pad_width=((0, 1), (0, 0), (0, 0)), mode='constant')
                img = img.transpose((2, 1, 0))
                axs[i, 0].imshow(img, origin='lower', aspect='auto')
                axs[i, 0].title.set_text(titles[i])

                cond_img = cond_sample[2 * i : 2 * i + 2]
                cond_img = np.pad(
                    cond_img, pad_width=((0, 1), (0, 0), (0, 0)), mode='constant'
                )
                cond_img[2][cond_img[0] < 0] = 1
                cond_img[cond_img < 0] = 0
                cond_img = cond_img.transpose((2, 1, 0))
                axs[i, 1].imshow(cond_img, origin='lower', aspect='auto')

            plt.show()

    def get_whole_song(self, song_id):
        min_pitch, max_pitch = self.min_pitches[song_id], self.max_pitches[song_id]
        if min_pitch < 48:
            shift = 12 * int(np.ceil((48 - min_pitch) / 12))
        elif max_pitch > 108:
            shift = -12 * int(np.ceil((max_pitch - 108) / 12))
        else:
            shift = 0

        key_roll = self.key_rolls[song_id]  # (2, 128, 12)
        phrase_roll = self.phrase_rolls[song_id]  # (6, 128, 1)
        red_roll = self.reductions[song_id]  # (2, 128, 128 )
        rough_chord_roll = self.rough_chord_rolls[song_id]  # (6, 128, 12)
        mel_roll = self.mel_rolls[song_id]
        chord_roll = self.chord_rolls[song_id]

        # pitch augmentation
        key_roll = np.roll(key_roll, shift=shift, axis=-1)
        red_roll = np.roll(red_roll, shift=shift, axis=-1)
        rough_chord_roll = np.roll(rough_chord_roll, shift=shift, axis=-1)
        mel_roll = np.roll(mel_roll, shift=shift, axis=-1)
        chord_roll = np.roll(chord_roll, shift=shift, axis=-1)
        actual_l = mel_roll.shape[1]
        target_l = max(
            int(np.ceil(mel_roll.shape[1] / (self.max_l // 2)) * self.max_l // 2),
            self.max_l
        )

        img = np.zeros((12, target_l, 132), dtype=np.float32)

        img[0 : 2, 0 : actual_l, 0 : 128] = mel_roll
        img[0 : 2, 0 : actual_l, 36 : 48] = chord_roll[2 : 4]
        img[0 : 2, 0 : actual_l, 24 : 36] = chord_roll[4 : 6]

        img[2 : 4, 0 : actual_l, 0 : 128] = red_roll
        img[2 : 4, 0 : actual_l, 36 : 48] = rough_chord_roll[2 : 4]
        img[2 : 4, 0 : actual_l, 24 : 36] = rough_chord_roll[4 : 6]

        img[6 : 12, 0 : actual_l] = phrase_roll

        img = img.reshape((12, target_l, 11, 12))
        img[4 : 6, 0 : actual_l] = key_roll[:, :, np.newaxis]
        img = img.reshape((12, target_l, 132))

        return img[:, :, 0 : self.max_h]


class Mel2AccDataset(HierarchicalDatasetBase):
    def __init__(self, analyses, shift_high=0, shift_low=0, max_l=128, max_h=128):
        super(Mel2AccDataset,
              self).__init__(analyses, shift_high, shift_low, max_l, max_h)
        self.key_rolls = [analysis['key_roll'] for analysis in analyses]
        self.expand_key_rolls()

        self.phrase_rolls = [
            analysis['phrase_roll'][:, :, np.newaxis] for analysis in analyses
        ]
        self.expand_phrase_rolls()

        self.rough_chord_rolls = [analysis['rough_chord_roll'] for analysis in analyses]
        self.expand_rough_chord_rolls()

        self.reductions = [analysis['mel_reduction'] for analysis in analyses]
        self.expand_reductions()

        self.mel_rolls = [analysis['mel_roll'] for analysis in analyses]
        self.chord_rolls = [analysis['chord_roll'] for analysis in analyses]
        self.expand_chord_rolls()

        self.acc_rolls = [analysis['acc_roll'] for analysis in analyses]

        self.lengths = np.array([mel.shape[1] for mel in self.mel_rolls])

        self.start_ids_per_song = [
            np.arange(0, lgth - self.max_l // 2, nspb * nbpm, dtype=np.int64)
            for lgth, nbpm, nspb in zip(self.lengths, self.nbpms, self.nspbs)
        ]

        self.indices = self._song_id_to_indices()

    def expand_key_rolls(self):
        self.key_rolls = [
            expand_roll(roll, nbpm * nspb)
            for roll, nbpm, nspb in zip(self.key_rolls, self.nbpms, self.nspbs)
        ]

    def expand_phrase_rolls(self):
        self.phrase_rolls = [
            expand_roll(roll, nbpm * nspb)
            for roll, nbpm, nspb in zip(self.phrase_rolls, self.nbpms, self.nspbs)
        ]

    def expand_rough_chord_rolls(self):
        self.rough_chord_rolls = [
            expand_roll(roll, nspb, contain_onset=True)
            for roll, nspb in zip(self.rough_chord_rolls, self.nspbs)
        ]

    def expand_chord_rolls(self):
        self.chord_rolls = [
            expand_roll(roll, nspb, contain_onset=True)
            for roll, nspb in zip(self.chord_rolls, self.nspbs)
        ]

    def expand_reductions(self):
        self.reductions = [
            expand_roll(roll, nspb, contain_onset=True)
            for roll, nspb in zip(self.reductions, self.nspbs)
        ]

    def get_data_sample(self, song_id, start_id, shift):
        # segment target data
        acc_shift = shift
        min_pitch, max_pitch = self.min_pitches[song_id] + shift, self.max_pitches[
            song_id] + shift
        if min_pitch < 48:
            shift = shift + 12 * int(np.ceil((48 - min_pitch) / 12))
        elif max_pitch > 108:
            shift = shift - 12 * int(np.ceil((max_pitch - 108) / 12))

        key_roll = self.key_rolls[song_id][:, start_id : start_id + self.max_l
                                          ]  # (2, 128, 12)
        phrase_roll = self.phrase_rolls[song_id][:, start_id : start_id + self.max_l
                                                ]  # (6, 128, 1)
        red_roll = self.reductions[song_id][:, start_id : start_id + self.max_l
                                           ]  # (2, 128, 128)
        rough_chord_roll = self.rough_chord_rolls[song_id][:, start_id : start_id +
                                                           self.max_l]  # (6, 128, 12)
        mel_roll = self.mel_rolls[song_id][:, start_id : start_id + self.max_l]
        chord_roll = self.chord_rolls[song_id][:, start_id : start_id + self.max_l]
        acc_roll = self.acc_rolls[song_id][:, start_id : start_id + self.max_l]

        # pitch augmentation
        key_roll = np.roll(key_roll, shift=shift, axis=-1)
        red_roll = np.roll(red_roll, shift=shift, axis=-1)
        rough_chord_roll = np.roll(rough_chord_roll, shift=shift, axis=-1)
        mel_roll = np.roll(mel_roll, shift=shift, axis=-1)
        chord_roll = np.roll(chord_roll, shift=shift, axis=-1)
        acc_roll = np.roll(acc_roll, shift=acc_shift, axis=-1)
        actual_l = mel_roll.shape[1]

        img = np.zeros((14, self.max_l, 132), dtype=np.float32)

        # acc
        img[0 : 2, 0 : actual_l, 0 : 128] = acc_roll

        # melody and chord
        img[2 : 4, 0 : actual_l, 0 : 128] = mel_roll
        img[2 : 4, 0 : actual_l, 36 : 48] = chord_roll[2 : 4]
        img[2 : 4, 0 : actual_l, 24 : 36] = chord_roll[4 : 6]

        # red and rough chord
        img[4 : 6, 0 : actual_l, 0 : 128] = red_roll
        img[4 : 6, 0 : actual_l, 36 : 48] = rough_chord_roll[2 : 4]
        img[4 : 6, 0 : actual_l, 24 : 36] = rough_chord_roll[4 : 6]

        # phrase
        img[8 : 14, 0 : actual_l] = phrase_roll

        # key
        img = img.reshape((14, self.max_l, 11, 12))
        img[6 : 8, 0 : actual_l] = key_roll[:, :, np.newaxis]
        img = img.reshape((14, self.max_l, 132))[:, :, 0 : self.max_h]

        # preparing for condition image
        cond_img = -np.ones((14, 108, 128), dtype=np.float32)
        nbpm = self.nbpms[song_id]
        nspb = self.nspbs[song_id]

        start_measure = start_id // (nbpm * nspb)
        if start_measure == 0:
            return img, cond_img

        measure_phrase_roll = self.phrase_rolls[song_id][:, :: nbpm * nspb, 0]
        phrase_type_importance = phrase_roll[:, :, 0].sum(-1)  # (6, )

        measure_importance = phrase_type_importance[measure_phrase_roll.argmax(0)
                                                   ][0 : start_measure]

        cond_scope = 6  # 8 measures
        interval = 4  # 4 steps

        slices = []
        n_cond = np.random.choice(np.arange(0, 2), p=np.array([0.1, 0.9]))

        for i in range(n_cond):
            # go through the dataset and compute accumulate distance
            acc_measure_importance = np.array(
                [
                    measure_importance[i : i + cond_scope].sum()
                    for i in range(len(measure_importance))
                ]
            )
            measure = acc_measure_importance.argmax()
            if acc_measure_importance[measure] < 0:
                break
            measure_importance[measure : measure + cond_scope] = -1
            slices.append((measure, min(start_measure, measure + cond_scope)))

        cur_step = 0
        for slc in slices:
            seg_start_id, seg_end_id = slc[0] * nbpm * nspb, slc[1] * nbpm * nspb
            seg_lgth = seg_end_id - seg_start_id
            cond_seg = self.get_cond_img(
                song_id, seg_start_id, seg_end_id, shift, acc_shift
            )
            cond_img[:, cur_step : cur_step + seg_lgth] = cond_seg
            cur_step += seg_lgth + interval

        return img, cond_img

    def get_cond_img(self, song_id, start_id, end_id, shift, acc_shift):
        key_roll = self.key_rolls[song_id][:, start_id : end_id]  # (2, 128, 12)
        phrase_roll = self.phrase_rolls[song_id][:, start_id : end_id]  # (6, 128, 1)
        red_roll = self.reductions[song_id][:, start_id : end_id]  # (2, 128, 128)
        rough_chord_roll = self.rough_chord_rolls[song_id][:, start_id : end_id
                                                          ]  # (6, 128, 12)
        mel_roll = self.mel_rolls[song_id][:, start_id : end_id]
        chord_roll = self.chord_rolls[song_id][:, start_id : end_id]
        acc_roll = self.acc_rolls[song_id][:, start_id : end_id]

        # pitch augmentation
        key_roll = np.roll(key_roll, shift=shift, axis=-1)
        red_roll = np.roll(red_roll, shift=shift, axis=-1)
        rough_chord_roll = np.roll(rough_chord_roll, shift=shift, axis=-1)
        mel_roll = np.roll(mel_roll, shift=shift, axis=-1)
        chord_roll = np.roll(chord_roll, shift=shift, axis=-1)
        acc_roll = np.roll(acc_roll, shift=acc_shift, axis=-1)
        actual_l = mel_roll.shape[1]

        img = np.zeros((14, end_id - start_id, 132), dtype=np.float32)
        img[0 : 2, 0 : actual_l, 0 : 128] = acc_roll

        img[2 : 4, 0 : actual_l, 0 : 128] = mel_roll
        img[2 : 4, 0 : actual_l, 36 : 48] = chord_roll[2 : 4]
        img[2 : 4, 0 : actual_l, 24 : 36] = chord_roll[4 : 6]

        img[4 : 6, 0 : actual_l, 0 : 128] = red_roll
        img[4 : 6, 0 : actual_l, 36 : 48] = rough_chord_roll[2 : 4]
        img[4 : 6, 0 : actual_l, 24 : 36] = rough_chord_roll[4 : 6]

        img[8 : 14, 0 : actual_l] = phrase_roll

        img = img.reshape((14, end_id - start_id, 11, 12))
        img[6 : 8, 0 : actual_l] = key_roll[:, :, np.newaxis]
        img = img.reshape((14, end_id - start_id, 132))[:, :, 0 : self.max_h]
        return img

    def show(self, item, show_img=True):
        sample, cond_sample = self[item]
        print(
            f"sample[{item}].shape = {sample.shape}), cond_sample[{item}].shape = {cond_sample.shape}"
        )
        titles = [
            'acc', 'mel+chd', 'mel+rough_chd', 'key', 'phrase0-1', 'phrase2-3',
            'phrase4-5'
        ]

        if show_img:
            fig, axs = plt.subplots(7, 2, figsize=(20, 40))
            for i in range(6):
                img = sample[2 * i : 2 * i + 2]
                img = np.pad(img, pad_width=((0, 1), (0, 0), (0, 0)), mode='constant')
                img = img.transpose((2, 1, 0))
                axs[i, 0].imshow(img, origin='lower', aspect='auto')
                axs[i, 0].title.set_text(titles[i])

                cond_img = cond_sample[2 * i : 2 * i + 2]
                cond_img = np.pad(
                    cond_img, pad_width=((0, 1), (0, 0), (0, 0)), mode='constant'
                )
                cond_img[2][cond_img[0] < 0] = 1
                cond_img[cond_img < 0] = 0
                cond_img = cond_img.transpose((2, 1, 0))
                axs[i, 1].imshow(cond_img, origin='lower', aspect='auto')

            plt.show()

    def get_whole_song(self, song_id):
        min_pitch, max_pitch = self.min_pitches[song_id], self.max_pitches[song_id]
        if min_pitch < 48:
            shift = 12 * int(np.ceil((48 - min_pitch) / 12))
        elif max_pitch > 108:
            shift = -12 * int(np.ceil((max_pitch - 108) / 12))
        else:
            shift = 0

        key_roll = self.key_rolls[song_id]  # (2, 128, 12)
        phrase_roll = self.phrase_rolls[song_id]  # (6, 128, 1)
        red_roll = self.reductions[song_id]  # (2, 128, 128 )
        rough_chord_roll = self.rough_chord_rolls[song_id]  # (6, 128, 12)
        mel_roll = self.mel_rolls[song_id]
        chord_roll = self.chord_rolls[song_id]

        # pitch augmentation
        key_roll = np.roll(key_roll, shift=shift, axis=-1)
        red_roll = np.roll(red_roll, shift=shift, axis=-1)
        rough_chord_roll = np.roll(rough_chord_roll, shift=shift, axis=-1)
        mel_roll = np.roll(mel_roll, shift=shift, axis=-1)
        chord_roll = np.roll(chord_roll, shift=shift, axis=-1)
        actual_l = mel_roll.shape[1]
        target_l = max(
            int(np.ceil(mel_roll.shape[1] / (self.max_l // 2)) * self.max_l // 2),
            self.max_l
        )

        img = np.zeros((12, target_l, 132), dtype=np.float32)

        img[0 : 2, 0 : actual_l, 0 : 128] = mel_roll
        img[0 : 2, 0 : actual_l, 36 : 48] = chord_roll[2 : 4]
        img[0 : 2, 0 : actual_l, 24 : 36] = chord_roll[4 : 6]

        img[2 : 4, 0 : actual_l, 0 : 128] = red_roll
        img[2 : 4, 0 : actual_l, 36 : 48] = rough_chord_roll[2 : 4]
        img[2 : 4, 0 : actual_l, 24 : 36] = rough_chord_roll[4 : 6]

        img[6 : 12, 0 : actual_l] = phrase_roll

        img = img.reshape((12, target_l, 11, 12))
        img[4 : 6, 0 : actual_l] = key_roll[:, :, np.newaxis]
        img = img.reshape((12, target_l, 132))

        return img[:, :, 0 : self.max_h]


def create_train_and_valid_analyses(truncate_length=None):
    train_analyses = []
    valid_analyses = []
    train_ids, valid_ids = load_split_file(SPLIT_FILE_PATH)

    for i in tqdm(range(909), desc='loading dataset'):
        label = LABEL_SOURCE[i]
        num_beat_per_measure = 3 if i + 1 in TRIPLE_METER_SONG else 4
        folder_name = str(i + 1).zfill(3)
        data_fn = os.path.join(DATASET_PATH, folder_name)
        acc_fn = os.path.join(ACC_DATASET_PATH, folder_name)

        song = read_data(
            data_fn,
            acc_fn,
            num_beat_per_measure=num_beat_per_measure,
            num_step_per_beat=4,
            clean_chord_unit=num_beat_per_measure,
            song_name=folder_name,
            label=label
        )
        analysis = song.analyze()

        if i in train_ids:
            train_analyses.append(analysis)
        else:
            valid_analyses.append(analysis)

        if truncate_length is not None and i == truncate_length - 1:
            break
    return train_analyses, valid_analyses


def create_n2key_datatsets(
    train_analyses,
    valid_analyses,
    shift_high_t=5,
    shift_low_t=-6,
    shift_high_v=0,
    shift_low_v=0,
    max_l=256,
    max_h=12,
    truncate_length=None
):
    train_dataset = N2KeyDataset(
        train_analyses, shift_high_t, shift_low_t, max_l, max_h
    )
    valid_dataset = N2KeyDataset(
        valid_analyses, shift_high_v, shift_low_v, max_l, max_h
    )
    return train_dataset, valid_dataset


def create_key2red_datasets(
    train_analyses,
    valid_analyses,
    shift_high_t=5,
    shift_low_t=-6,
    shift_high_v=0,
    shift_low_v=0,
    max_l=128,
    max_h=128,
    truncate_length=None
):
    train_dataset = Key2RedDataset(
        train_analyses, shift_high_t, shift_low_t, max_l, max_h
    )
    valid_dataset = Key2RedDataset(
        valid_analyses, shift_high_v, shift_low_v, max_l, max_h
    )
    return train_dataset, valid_dataset


def create_red2mel_datasets(
    train_analyses,
    valid_analyses,
    shift_high_t=5,
    shift_low_t=-6,
    shift_high_v=0,
    shift_low_v=0,
    max_l=128,
    max_h=128,
    truncate_length=None
):

    train_dataset = Red2MelDataset(
        train_analyses, shift_high_t, shift_low_t, max_l, max_h
    )
    valid_dataset = Red2MelDataset(
        valid_analyses, shift_high_v, shift_low_v, max_l, max_h
    )
    return train_dataset, valid_dataset


def create_mel2acc_datasets(
    train_analyses,
    valid_analyses,
    shift_high_t=5,
    shift_low_t=-6,
    shift_high_v=0,
    shift_low_v=0,
    max_l=128,
    max_h=128,
    truncate_length=None
):

    train_dataset = Mel2AccDataset(
        train_analyses, shift_high_t, shift_low_t, max_l, max_h
    )
    valid_dataset = Mel2AccDataset(
        valid_analyses, shift_high_v, shift_low_v, max_l, max_h
    )
    return train_dataset, valid_dataset


def compute_min_max_pitch():
    min_pitch, max_pitch = 128, 0
    for i in tqdm(range(909), desc='checking min/max pitches'):
        label = LABEL_SOURCE[i]
        num_beat_per_measure = 3 if i + 1 in TRIPLE_METER_SONG else 4
        folder_name = str(i + 1).zfill(3)
        data_fn = os.path.join(DATASET_PATH, folder_name)

        song = read_data(
            data_fn,
            num_beat_per_measure=num_beat_per_measure,
            num_step_per_beat=4,
            clean_chord_unit=num_beat_per_measure,
            song_name=folder_name,
            label=label
        )

        pitches = song.melody[:, 1]
        if pitches.min() <= 48:
            pitches += 24
        # elif pitches.min() <= 60:
        #     pitches += 12
        max_pitch = max(max_pitch, pitches.max())
        min_pitch = min(min_pitch, pitches.min())
    print('max_pitch:', max_pitch, 'min_pitch:', min_pitch)


def song_length_stats():
    n_measures = []
    n_beats = []

    for i in tqdm(range(909), desc='checking song length distribution'):
        label = LABEL_SOURCE[i]
        num_beat_per_measure = 3 if i + 1 in TRIPLE_METER_SONG else 4
        folder_name = str(i + 1).zfill(3)
        data_fn = os.path.join(DATASET_PATH, folder_name)

        song = read_data(
            data_fn,
            num_beat_per_measure=num_beat_per_measure,
            num_step_per_beat=4,
            clean_chord_unit=num_beat_per_measure,
            song_name=folder_name,
            label=label
        )
        n_measures.append(song.total_measure)
        n_beats.append(song.total_beat)
    n_measures = np.array(n_measures, dtype=np.int64)
    n_beats = np.array(n_beats, dtype=np.int64)

    print(f"n_measures max: {n_measures.max()}, n_beats max: {n_beats.max()}")
    print(f"n_measures min: {n_measures.min()}, n_beats max: {n_beats.min()}")
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    axs[0].hist(n_measures, bins=100)
    axs[1].hist(n_beats, bins=100)
    plt.show()


def create_train_valid_dataloaders(batch_size, train_set, valid_set):
    train_dl = DataLoader(train_set, batch_size, True)
    valid_dl = DataLoader(valid_set, batch_size, True)
    return train_dl, valid_dl


if __name__ == '__main__':
    # compute_min_max_pitch()
    # song_length_stats()
    # todo: reduced chord
    # pad length and shift: how many
    train_analyses, valid_analyses = create_train_and_valid_analyses(truncate_length=6)

    train_set0, valid_set0 = create_n2key_datatsets(train_analyses, valid_analyses)
    train_set1, valid_set1 = create_key2red_datasets(train_analyses, valid_analyses)
    train_set2, valid_set2 = create_red2mel_datasets(train_analyses, valid_analyses)
    train_set3, valid_set3 = create_mel2acc_datasets(train_analyses, valid_analyses)

    # for dataset_id, dataset in enumerate([train_set0, valid_set0, train_set1, valid_set1, train_set2, valid_set2]):
    #     print(len(dataset), dataset[0].shape)
    #     shape = dataset[0].shape
    #     for i in tqdm(range(len(dataset)), desc=f"i={dataset_id}"):
    #         assert dataset[i].shape == shape

    # train_set0.show(0)
    train_set1.show(450)  # 450
    train_set2.show(233)
    train_set3.show(233)
