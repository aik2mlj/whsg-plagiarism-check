import argparse
import numpy as np
import torch

import main_latent
import utils


def test_midis(nmats):
    for mel, chd, num_bar in nmats:
        # print(mel.shape, chd.shape)
        utils.nmat_to_midi_file(mel, "./mel_nmat.mid")
        prmat = utils.nmat_to_melprmat(mel, num_bar)
        chd_ec2vae = utils.nmat_to_chd_ec2vae(chd, num_bar)
        # print(chd_ec2vae)
        utils.melprmat_to_midi_file(prmat, "./mel.mid")
        utils.chd_ec2vae_to_midi_file(chd_ec2vae, "./chd.mid")
        exit(0)


def augment_to_12keys(nmats):
    """
    chd : [:, 16]
        0: start time
        1: root
        2-13: absolute chroma
        14: bass
        15: dur
    """
    augmented = []
    for key in range(-6, 6):
        nmats_augmented = []
        for mel, chd, num_bar in nmats:
            mel_aug = mel.copy()
            mel_aug[:, 1] += key
            chd_aug = chd.copy()
            chd_aug[:, 1] = (chd_aug[:, 1] + key) % 12
            chd_aug[:, 2 : 14] = np.roll(chd_aug[:, 2 : 14], key)
            nmats_augmented.append((mel_aug, chd_aug, num_bar))
        augmented.append(nmats_augmented)
    return augmented


def store_latent_code(train_melchd, fpath, batch_size=500):
    """
    mel, train_mels should be melchroma #[:, 16*seg_len, 12]
    """
    # truncate melprmat to only onset & pitch, without rest & sustain
    train_mels, train_chds = train_melchd
    train_mels = train_mels.split(batch_size, dim=0)
    train_chds = train_chds.split(batch_size, dim=0)

    zps = []
    zrs = []
    with torch.no_grad():
        for train_mel_batch, train_chd_batch in zip(train_mels, train_chds):
            dist_p_train, dist_r_train = main_latent.ec2vae_enc.encoder(
                train_mel_batch, train_chd_batch
            )
            zps.append(dist_p_train.mean)
            zrs.append(dist_r_train.mean)
    zps = torch.concat(zps, dim=0).detach().cpu().numpy()
    zrs = torch.concat(zrs, dim=0).detach().cpu().numpy()
    print(zps.shape, zrs.shape)
    savez = {"p": zps, "r": zrs}
    np.savez(fpath, **savez)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=500)
    args = parser.parse_args()
    train_nmats, val_nmats = main_latent.get_train_val_melchd_nmats()
    train_augmented_nmats = augment_to_12keys(train_nmats)
    # test_midis(train_nmats)

    for aug, train_nmats in enumerate(train_augmented_nmats):
        train_melchd = main_latent.get_ec2vae_inputs(train_nmats, 2)
        store_latent_code(
            train_melchd,
            f"./latents/train_aug{aug}.npz",
            batch_size=int(args.batch_size)
        )

    # nmats = main_latent.get_nmats_from_dir("./test_data/128samples/ours", [1], [2])
    # test_midis(nmats)
