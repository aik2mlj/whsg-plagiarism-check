import os

from tqdm import tqdm

from data_utils.dataset import (
    ACC_DATASET_PATH,
    DATASET_PATH,
    LABEL_SOURCE,
    TRIPLE_METER_SONG,
    load_split_file,
    read_data,
)


def get_train_val_mels():
    train_mels = []
    valid_mels = []
    train_ids, valid_ids = load_split_file("./data/pop909_mel/split.npz")

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
        # analysis = song.analyze()

        if i in train_ids:
            train_mels.append(song.melody)
        else:
            valid_mels.append(song.melody)

    print(len(train_mels), len(valid_mels))
    return train_mels, valid_mels


if __name__ == "__main__":
    pass
