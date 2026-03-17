import torch
from metrics.discriminative_torch import discriminative_score_metrics
import numpy as np

def calculate_disc_two_paths(real_path, fake_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_dict = torch.load(real_path, map_location="cpu", weights_only=False)
    real = real_dict["real_ts"]
    print(f"real shape = {real.shape}")
    samples_dict = torch.load(fake_path, map_location="cpu", weights_only=False)
    print(f"fake shape = {samples_dict['sampled_ts'].shape}")
    num_samples = min(len(real), len(samples_dict["sampled_ts"][0]))
    real = real[:num_samples]


    print(f"real shape: {real.shape}")
    disc_score_list = []
    for i in range(10):
        fake = samples_dict["sampled_ts"][0, :num_samples]

        discriminative_score = discriminative_score_metrics(
            real, fake,
            real.shape[-1],
            device,
        )
        disc_score_list.append(discriminative_score)


    disc_score_arr = np.array(disc_score_list)
    disc_mean = disc_score_arr.mean()
    disc_std = disc_score_arr.std(ddof=1)

    print(fake_path)
    print(f"Disc Score: mean = {disc_mean:.4f}, std = {disc_std:.4f}")
    print("---" * 50)


if __name__ == "__main__":
    calculate_disc_two_paths(
        "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u/text2ts_msmdiffmv/0/verbalts_caps_samples.pt",
        "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u/text2ts_msmdiffmv/0/verbalts_caps_samples.pt"
    )
    calculate_disc_two_paths(
        "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_u/text2ts_msmdiffmv/0/samples.pt",
        "/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/uncond_synth_u/text2ts_msmdiffmv/0/samples.pt"
    )