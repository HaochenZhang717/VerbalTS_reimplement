import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.vae.vae import TimeSeriesVAE


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
def load_dataset(dict_path, dict_key):
    data = torch.load(dict_path)[dict_key]
    if dict_key == "sampled_ts":
        data = data[0]
    print(f"Loaded {dict_path}, key: {dict_key}: {data.shape}")
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

        out = model(x)
        mu = out["mu"]   # (B, C, T', latent)

        emb = mu

        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)


# =========================
# Main
# =========================
def main(args):

    device = args.device if torch.cuda.is_available() else "cpu"

    real_dataset = load_dataset(args.real_path, "real_ts")
    fake_dataset = load_dataset(args.fake_path, "sample_ts")

    real_dataloader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)
    fake_dataloader = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False)

    sample = real_dataset[0][0]
    C, T = sample.shape

    # ===== load model =====
    model = TimeSeriesVAE(
        input_dim=C,
        output_dim=C,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        latent_dim=args.latent_dim,
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    print(f"Loaded checkpoint from {args.ckpt_path}")

    # ===== extract =====
    real_embeddings = extract_embeddings(
        model,
        real_dataloader,
        device,
    ) # (N, seq_len, dim)

    fake_embeddings = extract_embeddings(
        model,
        fake_dataloader,
        device,
    ) # (N, seq_len, dim)



if __name__ == "__main__":
    args = get_args()
    main(args)