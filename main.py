import os
import argparse
import numpy as np

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


def get_train_val_mel_nmats():
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


def get_segment_w_rolling(prmat, seg_len):
    # rolling
    prmat_rolls = [prmat]
    for i in range(1, seg_len):
        prmat_rolls.append(torch.roll(prmat, -i, dims=0))
    # here is the seg with hopsize=1
    prmat_seg = torch.cat(prmat_rolls, dim=1)
    if seg_len > 1:
        prmat_seg = prmat_seg[:-(seg_len - 1), :, :]
    return prmat_seg


def get_melprmat(nmat, seg_len=2):
    prmats = []
    for mel, num_bar in tqdm(nmat, desc="nmat to melprmat"):
        # get 1-bar melprmat
        prmat = utils.nmat_to_melprmat(mel, num_bar)
        prmat_seg = get_segment_w_rolling(prmat, seg_len)
        prmats.append(prmat_seg)
    prmats = torch.concat(prmats)
    print(prmats.shape)

    print("cleaning empty segments...")
    empty_index = torch.abs(prmats).sum(dim=(1, 2)) > 0
    prmats = prmats[empty_index]
    print(prmats.shape)

    return prmats.to(device)


def get_melchroma(nmat, seg_len=2):
    prmats = []
    for mel, num_bar in tqdm(nmat, desc="nmat to melchroma"):
        # get 1-bar melchroma
        prmat = utils.nmat_to_melchroma(mel, num_bar)
        prmat_seg = get_segment_w_rolling(prmat, seg_len)
        prmats.append(prmat_seg)
    prmats = torch.concat(prmats)
    print(prmats.shape)

    print("cleaning empty segments...")
    empty_index = torch.abs(prmats).sum(dim=(1, 2)) > 0
    prmats = prmats[empty_index]
    print(prmats.shape)

    return prmats.to(device)


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
                ratio, (2 * num_copy / (num_train_notes + num_mel_notes)).max().item()
            )
        ratios.append(ratio)
    ratios = np.array(ratios)
    ratio_mean = ratios.mean(axis=0)
    ratio_std = ratios.std(axis=0)
    print(ratio_mean, ratio_std)
    return ratio_mean, ratio_std


def get_nmats_from_dir(dir, note_tracks):
    nmats = []
    for phrase_anno in os.scandir(dir):
        if phrase_anno.is_dir():
            phrase_config = utils.phrase_config_from_string(phrase_anno.name)
            num_bar = 0
            for phrase in phrase_config:
                num_bar += phrase[1]
            print(phrase_config, num_bar)
            for f in tqdm(os.scandir(phrase_anno.path)):
                fpath = f.path
                music = utils.get_music(fpath)
                nmat = utils.get_note_matrix(music, note_tracks)
                nmats.append((nmat, num_bar))
    return nmats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type")
    parser.add_argument("--seg-len", default=2)
    args = parser.parse_args()
    seg_len = int(args.seg_len)

    train_nmats, val_nmats = get_train_val_mel_nmats()
    train_melchroma = get_melchroma(train_nmats, seg_len)

    match args.type:
        case "ours":
            nmats = get_nmats_from_dir("./128samples/ours", [1])
            mel = get_melchroma(nmats, seg_len)
        case "polyff+phl":
            nmats = get_nmats_from_dir("./128samples/diff+phrase/mel+acc_samples", [0])
            mel = get_melchroma(nmats, seg_len)
        case "train":
            nmats = train_nmats[: 20]
            mel = get_melchroma(nmats, seg_len)

        case "val":
            nmats = val_nmats
            mel = get_melchroma(nmats, seg_len)
        case "remi+phl":
            nmats = get_nmats_from_dir("./128_remi/mel_32", [0])
            mel = get_melchroma(nmats, seg_len)
        case "copybot":
            prmats = []
            for mel, num_bar in train_nmats:
                prmats.append(utils.nmat_to_melchroma(mel, num_bar))
            prmats = torch.cat(prmats, dim=0)
            random_indices = torch.randint(len(prmats), (128 * 40, ))
            mel = get_segment_w_rolling(prmats[random_indices], seg_len)
            print(mel.shape)

        case _:
            raise RuntimeError

    compute_notewise_copy_ratio(mel, train_melchroma)
