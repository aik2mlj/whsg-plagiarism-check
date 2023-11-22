import argparse
import os
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from ec2vae.model import EC2VAE

import utils
from data_utils.dataset import (
    ACC_DATASET_PATH,
    DATASET_PATH,
    LABEL_SOURCE,
    TRIPLE_METER_SONG,
    load_split_file,
    read_data,
)
from main import get_segment_w_rolling
from pretrained_encoders import Ec2VaeEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"
ec2vae_enc = Ec2VaeEncoder.create_2bar_encoder()
ec2vae_enc.load_state_dict(torch.load("./pretrained_models/ec2vae_enc_2bar.pt"))
ec2vae_enc = ec2vae_enc.to(device)

# initialize the model
ec2vae_model = EC2VAE.init_model()

# load model parameter
ec2vae_param_path = './ec2vae/model_param/ec2vae-v1.pt'
ec2vae_model.load_model(ec2vae_param_path)


def get_train_val_melchd_nmats():
    """
    return list[(melody, chord, num_bar], in nmats
    """
    train_nmat = []
    valid_nmat = []
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
            train_nmat.append((song.melody, song.chord, n_bar))
        else:
            valid_nmat.append((song.melody, song.chord, n_bar))

    print(f"train set: {len(train_nmat)}, val set: {len(valid_nmat)}")
    return train_nmat, valid_nmat


def get_ec2vae_inputs(nmats, seg_len=2):
    prmats = []
    chds = []
    for mel, chd, num_bar in tqdm(nmats, desc="nmat to ec2vae"):
        # get 1-bar melprmat
        prmat = utils.nmat_to_melprmat(mel, num_bar)
        chd_ec2vae = utils.nmat_to_chd_ec2vae(chd, num_bar)
        prmat_seg = get_segment_w_rolling(prmat, seg_len)
        chd_seg = get_segment_w_rolling(chd_ec2vae, seg_len)
        prmats.append(prmat_seg)
        chds.append(chd_seg)
    prmats = torch.concat(prmats)
    chds = torch.concat(chds)
    assert prmats.shape[0] == chds.shape[0]
    prmats, chds = clean_sparse_segments(prmats, chds)

    return prmats.to(device), chds.to(device)


def clean_sparse_segments(prmats, chds):
    print(prmats.shape)
    assert prmats.shape[0] == chds.shape[0]
    print("cleaning empty/sparse segments...")
    empty_index = torch.abs(prmats[:, :, : 128]).sum(dim=(1, 2)) > 1
    prmats = prmats[empty_index]
    chds = chds[empty_index]
    print(prmats.shape)
    return prmats, chds


def load_latent_npz(fpath):
    latent = np.load(fpath)
    zp, zr = latent["p"], latent["r"]
    zp = torch.from_numpy(zp).to(device)
    zr = torch.from_numpy(zr).to(device)
    return (zp, zr)


def compute_latent_copy_ratio(latent, train_latent):
    """
    mel, train_mels should be melchroma #[:, 16*seg_len, 12]
    """
    # truncate melprmat to only onset & pitch, without rest & sustain
    zp, zr = latent
    train_size = train_latent[0][0].shape[0]

    ratios_p = []
    ratios_r = []

    with torch.no_grad():
        for zp_2bar, zr_2bar in zip(tqdm(zp), zr):
            zp_sim = -1.
            zr_sim = -1.
            # for train_zp_batch, train_zr_batch in zip(train_zp, train_zr):
            # size = train_zp_batch.shape[0]
            zp_expand = zp_2bar.expand(train_size, -1)
            zr_expand = zr_2bar.expand(train_size, -1)

            for train_aug in train_latent:
                train_zp, train_zr = train_aug
                zp_sim = max(
                    zp_sim,
                    F.cosine_similarity(zp_expand, train_zp, dim=-1).max().item()
                )
                zr_sim = max(
                    zr_sim,
                    F.cosine_similarity(zr_expand, train_zr, dim=-1).max().item()
                )
            ratios_p.append(float(zp_sim))
            ratios_r.append(float(zr_sim))
    ratios_p = np.array(ratios_p, dtype=float)
    ratios_r = np.array(ratios_r, dtype=float)
    return ratios_p, ratios_r


def get_nmats_from_dir(dir, mel_tracks, chd_tracks):
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
                mel, chd = get_nmat_from_midi(fpath, mel_tracks, chd_tracks)
                nmats.append((mel, chd, num_bar))
    return nmats


def get_nmat_from_midi(fpath, mel_tracks, chd_tracks):
    music = utils.get_music(fpath)
    mel = utils.get_note_matrix(music, mel_tracks)
    chd = utils.get_note_matrix(music, chd_tracks)
    return mel, chd


def store_latent(melchd, fpath, batch_size=500):
    """
    mel, train_mels should be melchroma #[:, 16*seg_len, 12]
    """
    # truncate melprmat to only onset & pitch, without rest & sustain
    train_mels, train_chds = melchd
    train_mels = train_mels.split(batch_size, dim=0)
    train_chds = train_chds.split(batch_size, dim=0)

    zps = []
    zrs = []
    with torch.no_grad():
        for train_mel_batch, train_chd_batch in zip(train_mels, train_chds):
            dist_p_train, dist_r_train = ec2vae_enc.encoder(
                train_mel_batch, train_chd_batch
            )
            zps.append(dist_p_train.mean)
            zrs.append(dist_r_train.mean)
    zps = torch.concat(zps, dim=0)
    zrs = torch.concat(zrs, dim=0)
    zps_np = zps.detach().cpu().numpy()
    zrs_np = zrs.detach().cpu().numpy()
    # print(zps.shape, zrs.shape)
    savez = {"p": zps_np, "r": zrs_np}
    np.savez(fpath, **savez)
    return (zps, zrs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type")
    parser.add_argument("--midi", help="midi file input")
    parser.add_argument("--mel-track", default=0)
    parser.add_argument("--chd-track", default=1)
    parser.add_argument("--seg-len", default=2)
    parser.add_argument("--batch-size", default=500)
    parser.add_argument("--ns-p", default=0.1)
    parser.add_argument("--ns-r", default=0.1)
    args = parser.parse_args()
    seg_len = int(args.seg_len)
    batch_size = int(args.batch_size)

    train_nmats, val_nmats = get_train_val_melchd_nmats()
    train_mel, train_chd = get_ec2vae_inputs(train_nmats, seg_len)
    # train_melchd = train_mel[: 100], train_chd[: 100]
    # train_melchd = train_mel, train_chd
    train_latent = []
    for i in range(12):
        train_latent.append(load_latent_npz(f"./latents/train_aug{i}.npz"))

    def output_result(melchd, name, latent=None):
        latent_path = f"./latents/{name}.npz"
        if latent is None:
            if os.path.exists(latent_path):
                latent = load_latent_npz(latent_path)
            else:
                latent = store_latent(melchd, latent_path, batch_size)

        rp, rr = compute_latent_copy_ratio(latent, train_latent)
        rp_mean = rp.mean(axis=0)
        rp_std = rp.std(axis=0)
        plt.clf()
        plt.hist(rp, bins=100, range=(0., 1.))
        plt.savefig(f"./plots_latent/p_{name}_{seg_len}.png")
        with open("output_latent.txt", "a") as f:
            f.write(f"\np {name}: {rp_mean:.4f}, {rp_std:.4f}")
        rr_mean = rr.mean(axis=0)
        rr_std = rr.std(axis=0)
        plt.clf()
        plt.hist(rr, bins=100, range=(0., 1.))
        plt.savefig(f"./plots_latent/r_{name}_{seg_len}.png")
        with open("output_latent.txt", "a") as f:
            f.write(f"\nr {name}: {rr_mean:.4f}, {rr_std:.4f}")

    def ours():
        nmats = get_nmats_from_dir("./test_data/128samples/ours", [1], [2])
        melchd = get_ec2vae_inputs(nmats, seg_len)
        output_result(melchd, "ours")

    def ours_new():
        dir = "./test_data/128_new"
        for epoch in os.scandir(dir):
            if epoch.is_dir():
                subdir = epoch.path
                nmats = get_nmats_from_dir(subdir, [1], [2])
                melchd = get_ec2vae_inputs(nmats, seg_len)
                output_result(melchd, f"ours_new_epoch{epoch.name}")

    def polyff_phl():
        nmats = get_nmats_from_dir(
            "./test_data/128samples/diff+phrase/mel+chd_samples", [0], [1]
        )
        melchd = get_ec2vae_inputs(nmats, seg_len)
        output_result(melchd, "polyff_phl")

    def train():
        nmats = train_nmats[: 20]
        melchd = get_ec2vae_inputs(nmats, seg_len)
        output_result(melchd, "train")

    def val():
        nmats = val_nmats
        melchd = get_ec2vae_inputs(nmats, seg_len)
        output_result(melchd, "val")

    def remi_phl():
        nmats = get_nmats_from_dir("./test_data/128_remi/mel_32", [0], [1])
        melchd = get_ec2vae_inputs(nmats, seg_len)
        output_result(melchd, "remi_phl")

    def copybot():
        prmats = []
        chds = []
        for mel, chd, num_bar in train_nmats:
            prmats.append(utils.nmat_to_melprmat(mel, num_bar))
            chds.append(utils.nmat_to_chd_ec2vae(chd, num_bar))
        prmats = torch.concat(prmats)
        chds = torch.concat(chds)
        assert prmats.shape[0] == chds.shape[0]
        random_indices = torch.randint(len(prmats), (128 * 40, ))
        prmats = prmats[random_indices]
        chds = chds[random_indices]
        mel = get_segment_w_rolling(prmats, seg_len)
        chd = get_segment_w_rolling(chds, seg_len)
        mel, chd = clean_sparse_segments(mel, chd)
        melchd = mel.to(device), chd.to(device)
        output_result(melchd, "copybot")

    def copybot_z(ns_p, ns_r):
        train_zp, train_zr = train_latent[6]  # the untransposed training set
        random_indices = torch.randint(len(train_zp), (128 * 40, ))
        zp, zr = train_zp[random_indices], train_zr[random_indices]
        noise_zp = torch.normal(0., 1., zp.shape).to(device)
        noise_zr = torch.normal(0., 1., zr.shape).to(device)
        zp = noise_zp * ns_p + zp * (1. - ns_p)
        zr = noise_zr * ns_r + zr * (1. - ns_r)
        output_result(None, "copybot_z", (zp, zr))

        # output to midi
        chd = train_chd[random_indices]
        recon = ec2vae_model.decoder(zp, zr, chd)
        recon = recon.squeeze(0).cpu().numpy()
        print(recon.shape)
        utils.melprmat_to_midi_file(recon, "copybot_z.mid")
        # recon = ec2vae_model.__class__.note_array_to_notes(recon, bpm=120, start=0.)

    funcs = [ours, ours_new, polyff_phl, train, val, remi_phl, copybot, copybot_z]

    if args.midi is not None:
        fname = args.midi.split("/")[-1]
        nmat = get_nmat_from_midi(
            args.midi, [int(args.mel_track)], [int(args.chd_track)]
        )
        num_bar = int(
            ceil((np.array(nmat[1])[:, 0] + np.array(nmat[1])[:, 2]).max() / 16)
        )
        print("num_bar:", num_bar)
        nmats = [(*nmat, num_bar)]
        mel = get_ec2vae_inputs(nmats, seg_len)
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
        elif args.type == "copybot_z":
            copybot_z(float(args.ns_p), float(args.ns_r))
        else:
            for func in funcs:
                func()
