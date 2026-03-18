import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.vae.fid_vae import FIDVAE

from scipy.linalg import sqrtm

def compute_fid(real, fake):
    """
    real, fake: (N, D)
    """

    mu_r = np.mean(real, axis=0)
    mu_f = np.mean(fake, axis=0)

    sigma_r = np.cov(real, rowvar=False)
    sigma_f = np.cov(fake, rowvar=False)

    covmean = sqrtm(sigma_r @ sigma_f)

    # 数值稳定（必须）
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu_r - mu_f) ** 2) + np.trace(
        sigma_r + sigma_f - 2 * covmean
    )

    return float(fid)


# =========================
# Args
# =========================
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--real_path", type=str, required=True)
    parser.add_argument("--fake_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=128)

    # model config（必须一致）
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=64)

    # embedding 类型
    # parser.add_argument(
    #     "--mode",
    #     type=str,
    #     default="mean",
    #     choices=["mean", "full", "flatten"]
    # )

    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


# =========================
# Dataset
# =========================
def load_dataset(dict_path, dict_key, idx=-1):
    data = torch.load(dict_path, weights_only=False)[dict_key]
    if dict_key == "sampled_ts":
        if idx > -1:
            data = data[idx]
    if data.shape[1] > data.shape[2]:
        data = data.permute(0,2,1)
    # print(f"Loaded {dict_path}, key: {dict_key}: {data.shape}")
    return TensorDataset(data)


# =========================
# Extract Embedding
# =========================
@torch.no_grad()
def extract_embeddings(model, dataloader, device):

    model.eval()

    all_embeddings = []

    for batch in tqdm(dataloader, desc="Extracting"):
        x = batch[0].to(device)
        with torch.no_grad():
            out = model(x)
        mu = out["mu"]   # (B, C, T', latent)

        emb = mu

        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0).cpu().numpy()


# =========================
# Main
# =========================
def main(args):

    device = args.device if torch.cuda.is_available() else "cpu"

    real_dataset = load_dataset(args.real_path, "real_ts", idx=-1)
    real_dataloader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)

    sample = real_dataset[0][0]
    C, T = sample.shape

    # ===== load model =====
    model = FIDVAE(
        input_dim=C,
        output_dim=C,
        seq_len=T,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        latent_dim=args.latent_dim,
    ).to(device).eval()

    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    print(f"Loaded checkpoint from {args.ckpt_path}")

    real_embeddings = extract_embeddings(
        model,
        real_dataloader,
        device,
    )


    fid_list = []
    for i in range(10):
        fake_dataset = load_dataset(args.fake_path, "sampled_ts",idx=i)
        fake_dataloader = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False)

        # ===== extract =====
        fake_embeddings = extract_embeddings(
            model,
            fake_dataloader,
            device,
        ) # (N, seq_len, dim)

        # print("Real embeddings:", real_embeddings.shape)
        # print("Fake embeddings:", fake_embeddings.shape)

        # ===== compute FID =====
        fid = compute_fid(real_embeddings, fake_embeddings)
        fid_list.append(fid)

    fid_array = np.array(fid_list)
    mean_fid = np.mean(fid_array)
    std_fid = np.std(fid_array)
    print("\n==========================")
    print(f"FID: ${mean_fid:.6f} \pm {std_fid:.6f}$")
    print("==========================\n")


if __name__ == "__main__":
    args = get_args()
    main(args)