import argparse
import os
from math import ceil

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import utils
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


def clean_sparse_segments(prmats):
    print(prmats.shape)
    print("cleaning empty/sparse segments...")
    empty_index = torch.abs(prmats).sum(dim=(1, 2)) > 1
    prmats = prmats[empty_index]
    print(prmats.shape)
    return prmats


def get_melchroma(nmat, seg_len=2):
    prmats = []
    for mel, num_bar in tqdm(nmat, desc="nmat to melchroma"):
        # get 1-bar melchroma
        prmat = utils.nmat_to_melchroma(mel, num_bar)
        prmat_seg = get_segment_w_rolling(prmat, seg_len)
        prmats.append(prmat_seg)
    prmats = torch.concat(prmats)
    prmats = clean_sparse_segments(prmats)

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
        ratios.append(float(ratio))
    ratios = np.array(ratios, dtype=float)
    return ratios


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
                nmat = get_nmat_from_midi(fpath, note_tracks)
                nmats.append((nmat, num_bar))
    return nmats


def get_nmat_from_midi(fpath, note_tracks):
    music = utils.get_music(fpath)
    nmat = utils.get_note_matrix(music, note_tracks)
    return nmat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type")
    parser.add_argument("--midi", help="midi file input")
    parser.add_argument("--note-track", default=0)
    parser.add_argument("--seg-len", default=2)
    args = parser.parse_args()
    seg_len = int(args.seg_len)

    train_nmats, val_nmats = get_train_val_mel_nmats()
    train_melchroma = get_melchroma(train_nmats, seg_len)

    def output_result(mel, run_type):
        ratios = compute_notewise_copy_ratio(mel, train_melchroma)
        ratio_mean = ratios.mean(axis=0)
        ratio_std = ratios.std(axis=0)
        plt.clf()
        plt.hist(ratios, bins=100, range=(0., 1.))
        plt.savefig(f"./plots/{run_type}_{seg_len}.png")
        with open("output.txt", "a") as f:
            f.write(f"\n{run_type}: {ratio_mean:.4f}, {ratio_std:.4f}")

    def ours():
        nmats = get_nmats_from_dir("./128samples/ours", [1])
        mel = get_melchroma(nmats, seg_len)
        output_result(mel, "ours")

    def ours_new():
        dir = "./128_new"
        for epoch in os.scandir(dir):
            if epoch.is_dir():
                subdir = epoch.path
                nmats = get_nmats_from_dir(subdir, [1])
                mel = get_melchroma(nmats, seg_len)
                output_result(mel, f"ours_new_epoch{epoch.name}")

    def polyff_phl():
        nmats = get_nmats_from_dir("./128samples/diff+phrase/mel+acc_samples", [0])
        mel = get_melchroma(nmats, seg_len)
        output_result(mel, "polyff_phl")

    def train():
        nmats = train_nmats[: 20]
        mel = get_melchroma(nmats, seg_len)
        output_result(mel, "train")

    def val():
        nmats = val_nmats
        mel = get_melchroma(nmats, seg_len)
        output_result(mel, "val")

    def remi_phl():
        nmats = get_nmats_from_dir("./128_remi/mel_32", [0])
        mel = get_melchroma(nmats, seg_len)
        output_result(mel, "remi_phl")

    def copybot():
        prmats = []
        for mel, num_bar in train_nmats:
            prmats.append(utils.nmat_to_melchroma(mel, num_bar))
        prmats = torch.concat(prmats)
        random_indices = torch.randint(len(prmats), (128 * 40, ))
        mel = get_segment_w_rolling(prmats[random_indices], seg_len)
        mel = clean_sparse_segments(mel)
        output_result(mel, "copybot")

    funcs = [ours, ours_new, polyff_phl, train, val, remi_phl, copybot]

    if args.midi is not None:
        fname = args.midi.split("/")[-1]
        nmat = get_nmat_from_midi(args.midi, [int(args.note_track)])
        num_bar = int(ceil((np.array(nmat)[:, 0] + np.array(nmat)[:, 2]).max() / 16))
        print("num_bar:", num_bar)
        nmats = [(nmat, num_bar)]
        mel = get_melchroma(nmats, seg_len)
        output_result(mel, f"{fname}")
    else:
        if args.type == "ours":
            ours()
        if args.type == "ours_new":
            ours_new()
        elif args.type == "polyff+phl":
            polyff_phl()
        elif args.type == "train":
            train()
        elif args.type == "val":
            val()
        elif args.type == "remi+phl":
            remi_phl()
        elif args.type == "copybot":
            copybot()
        else:
            for func in funcs:
                func()
