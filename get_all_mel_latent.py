import os

import torch
import utils

from tqdm import tqdm

from data_utils.dataset import (
    ACC_DATASET_PATH,
    DATASET_PATH,
    LABEL_SOURCE,
    TRIPLE_METER_SONG,
    load_split_file,
    read_data,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


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
        # total measure, num beat per measure
        n_bar, nbpm = song.total_measure, song.num_beat_per_measure
        if nbpm == 3:
            continue

        if i in train_ids:
            train_mels.append((song.melody, n_bar))
        else:
            valid_mels.append((song.melody, n_bar))

    print(f"train set: {len(train_mels)}, val set: {len(valid_mels)}")
    return train_mels, valid_mels


def get_melprmat(mels, seg_len=2):
    prmats = []
    for mel, num_bar in tqdm(mels, desc="nmat to melprmat"):
        # get 1-bar melprmat
        prmat = utils.nmat_to_melprmat(mel, num_bar)
        # rolling
        prmat_rolls = [prmat]
        for i in range(1, seg_len):
            prmat_rolls.append(torch.roll(prmat, -i, dims=0))
        # here is the seg with hopsize=1
        prmat_seg = torch.cat(prmat_rolls, dim=1)[:, :-(seg_len - 1), :]

        prmats.append(prmat_seg)
    prmats = torch.concat(prmats)
    print(prmats.shape)

    # clean empty 2bars
    print("cleaning empty 2-bars...")
    empty_index = torch.abs(prmats).sum(dim=(1, 2)) > 0
    prmats = prmats[empty_index]
    print(prmats.shape)

    return prmats.to(device)


def get_melchroma(mels, seg_len=2):
    prmats = []
    for mel, num_bar in tqdm(mels, desc="nmat to melchroma"):
        # get 1-bar melchroma
        prmat = utils.nmat_to_melchroma(mel, num_bar)
        # rolling
        prmat_rolls = [prmat]
        for i in range(1, seg_len):
            prmat_rolls.append(torch.roll(prmat, -i, dims=0))
        # here is the seg with hopsize=1
        prmat_seg = torch.cat(prmat_rolls, dim=1)[:-(seg_len - 1), :, :]

        prmats.append(prmat_seg)
    prmats = torch.concat(prmats)
    print(prmats.shape)

    # clean empty 2bars
    print("cleaning empty 2-bars...")
    empty_index = torch.abs(prmats).sum(dim=(1, 2)) > 0
    prmats = prmats[empty_index]
    print(prmats.shape)

    return prmats


def compute_notewise_copy_ratio(mel, train_mels):
    """
    mel, train_mels should be melchroma #[:, 16*seg_len, 12]
    """
    # truncate melprmat to only onset & pitch, without rest & sustain
    train_size = train_mels.shape[0]

    ratios = []
    for mel_2bar in tqdm(mel):
        mel_expand = mel_2bar.unsqueeze(0).expand(train_size, -1, -1)

        # notes the given melody contains
        num_mel_notes = torch.sum(mel_expand == 1, dim=(1, 2))
        # notes the training set contains
        num_train_notes = torch.sum(train_mels == 1, dim=(1, 2))

        # transpose to 12 keys
        ratio = -1.
        for i in range(12):
            train_transpose = torch.roll(train_mels, i, dims=2)
            # notes that the given melody match with the training set (pitch & onset identical)
            num_copy = torch.sum(
                (mel_expand == train_transpose) * (mel_expand == 1), dim=(1, 2)
            )
            ratio = max(
                ratio, (num_copy / (num_train_notes + num_mel_notes)).max().item()
            )
        ratios.append(ratio)
    print(ratios)
    ratio = sum(ratios) / len(ratios)
    print(ratio)
    return ratio


if __name__ == "__main__":
    train_mels, val_mels = get_train_val_mels()
    train_melchroma = get_melchroma(train_mels)
    val_melchroma = get_melchroma([val_mels[0]])

    print(compute_notewise_copy_ratio(val_melchroma, train_melchroma))
    exit(0)
