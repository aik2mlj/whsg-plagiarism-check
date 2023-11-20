import argparse
import numpy as np
import torch

import main_latent


def store_latent_code(train_melchd, batch_size=500):
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
    np.savez("train_latent", **savez)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=500)
    args = parser.parse_args()
    train_nmats, val_nmats = main_latent.get_train_val_melchd_nmats()
    train_melchd = main_latent.get_ec2vae_inputs(train_nmats, 2)
    store_latent_code(train_melchd, batch_size=int(args.batch_size))
