import torch
from metrics.discriminative_torch import discriminative_score_metrics
import numpy as np

import json
import os
import numpy as np
import torch


def calculate_disc_two_paths(real_path, fake_path, save_path="disc_results.jsonl"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_dict = torch.load(real_path, map_location="cpu", weights_only=False)
    real = real_dict["real_ts"]

    samples_dict = torch.load(fake_path, map_location="cpu", weights_only=False)

    num_samples = min(len(real), len(samples_dict["sampled_ts"][0]))
    real = real[:num_samples]

    disc_score_list = []
    print(real.shape)
    print(samples_dict["sampled_ts"].shape)
    for i in range(samples_dict["sampled_ts"].shape[0]):
        fake = samples_dict["sampled_ts"][i, :num_samples]
        for _ in range(10):
            discriminative_score = discriminative_score_metrics(
                real, fake,
                real.shape[-1],
                device,
            )
            disc_score_list.append(discriminative_score)

    disc_score_arr = np.array(disc_score_list)
    disc_mean = float(disc_score_arr.mean())
    disc_std = float(disc_score_arr.std(ddof=1))

    # ===== 构造结果 =====
    result = {
        "real_path": real_path,
        "fake_path": fake_path,
        "num_samples": int(num_samples),
        "disc_scores": disc_score_list,
        "disc_mean": disc_mean,
        "disc_std": disc_std,
    }

    # ===== 写入 jsonl（append）=====
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None

    with open(save_path, "a") as f:
        f.write(json.dumps(result) + "\n")

    # ===== 仍然print（方便你看）=====
    print(fake_path)
    print(f"Disc Score: mean = {disc_mean:.4f}, std = {disc_std:.4f}")
    print("---" * 50)


if __name__ == "__main__":


    calculate_disc_two_paths(
        "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_vae_embed/text2ts_msmdiffmv/0/real_text_samples.pt",
        "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_vae_embed/text2ts_msmdiffmv/0/real_text_samples.pt"
    )

    calculate_disc_two_paths(
        "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen/text2ts_msmdiffmv/0/real_text_samples.pt",
        "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_qwen/text2ts_msmdiffmv/0/real_text_samples.pt"
    )

    calculate_disc_two_paths(
        "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u/text2ts_msmdiffmv/0/verbalts_caps_samples.pt",
        "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u/text2ts_msmdiffmv/0/verbalts_caps_samples.pt"
    )
    calculate_disc_two_paths(
        "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_u/text2ts_msmdiffmv/0/samples.pt",
        "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_u/text2ts_msmdiffmv/0/samples.pt"
    )

