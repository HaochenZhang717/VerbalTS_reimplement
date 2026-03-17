import os
import json
import numpy as np
import random
from torch.utils.data import Dataset
import time
import torch


class CustomDataset:
    def __init__(self, folder, **kwargs):
        super().__init__()
        self.folder = folder
        self._load_meta()

    def _load_meta(self):
        self.meta = json.load(open(os.path.join(self.folder, "meta.json")))
        self.attr_list = self.meta["attr_list"]
        n_attr = len(self.attr_list)
        self.attr_ids = np.arange(n_attr)
        self.attr_n_ops = np.array(self.meta["attr_n_ops"])

    def get_split(self, split, *args):
        return CustomSplit(self.folder, split)


class CustomSplit(Dataset):
    def __init__(self, folder, split="train"):
        super().__init__()
        assert split in ("train", "valid", "test"), "Please specify a valid split."
        self.split = split            
        self.folder = folder
        self._load_data()

        print(f"Split: {self.split}, total samples {self.n_samples}.")

    def _load_data(self):
        ts = np.load(os.path.join(self.folder, self.split+"_ts.npy"))     # [n_samples, n_steps]
        attrs = np.load(os.path.join(self.folder, self.split+"_attrs_idx.npy"))  # [n_samples, n_attrs]
        caps = np.load(os.path.join(self.folder, self.split+fr"_text_caps.npy"), allow_pickle=True)

        self.ts, self.attrs, self.caps = ts, attrs, caps
        self.n_samples = self.ts.shape[0]
        self.n_steps = self.ts.shape[1]
        self.n_attrs = self.attrs.shape[1]
        self.time_point = np.arange(self.n_steps)

    def __getitem__(self, idx):
        cap_id = random.randint(0, len(self.caps[idx])-1)
        tmp_ts = self.ts[idx]
        if len(tmp_ts.shape) == 1:
            tmp_ts = tmp_ts[...,np.newaxis]
        return {"ts": tmp_ts,
                "ts_len": tmp_ts.shape[0],
                "attrs": self.attrs[idx],
                "cap": self.caps[idx][cap_id],
                "tp": self.time_point}

    def __len__(self):
        return self.n_samples

class MyDataset:
    """
    Wrapper class so that the block-causal dataset fits
    the GenerationDataset interface.
    """

    def __init__(
            self,
            ts_path,
            caps_path,
            seq_len,
            num_channels,
            num_segments=4,
            **kwargs
    ):
        self.ts_path = ts_path
        self.caps_path = caps_path
        self.seq_len = seq_len
        self.num_segments = num_segments
        self.num_channels = num_channels

        self.attr_n_ops = None

    def get_split(self, split, *args):
        return MySplit(
            ts_path=self.ts_path,
            caps_path=self.caps_path,
            seq_len=self.seq_len,
            num_channels=self.num_channels,
            num_segments=self.num_segments,
            split=split,
        )

class MySplit(Dataset):
    def __init__(
            self,
            ts_path,
            caps_path,
            seq_len,
            num_channels,
            num_segments=4,
            split="train",
    ):
        super().__init__()

        self.split = split
        self.num_segments = num_segments
        self.num_channels = num_channels

        self.caps_path = caps_path
        # ------------------------
        # load data
        # ------------------------
        self.ts = None
        if ts_path != "none":
            self.ts = np.load(f"{ts_path}/{split}_ts.npy", allow_pickle=True)  # (N,T,C)
            self.N, self.T, self.C = self.ts.shape
        else:
            self.N = -1
            self.T = seq_len
            self.C = num_channels


        self.caps = None
        if self.caps_path != "none":
            caps_dict = {}
            with open(f"{self.caps_path}/{split}_caps_ready.jsonl", "r") as f:
                for line in f:
                    item = json.loads(line)
                    caps_dict[item["id"]] = item["captions"]
            self.caps = caps_dict

        assert self.T % self.num_segments == 0

        self.segment_length = self.T // self.num_segments

        self.ids = sorted(
            self.text_embed.keys(),
            key=lambda x: int(x.replace("image", "")),
        )

        self.block_ids = list(range(self.num_segments))
        self.num_block_choices = len(self.block_ids)

        print(
            f"[CausalSplit:{self.split}] "
            f"N={self.N}, T={self.T}, C={self.C}, segments={self.num_segments}"
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        ts_id = int(image_id.replace("image", ""))

        if self.ts is not None:
            ts = self.ts[ts_id]  # (T,C)
            ts = torch.from_numpy(ts).float().transpose(0, 1)  # (C,T)
        else:
            ts = torch.zeros((self.C, self.T)).float()

        if self.caps is not None:
            caps = self.caps[image_id]

        else:
            caps = "caps not loaded."


        item = {
            "ts": ts,
            "ts_len": self.T,
            "image_id": image_id,
            "ts_id": ts_id,
            "caps": caps,
        }

        return item

    @staticmethod
    def collate_fn(batch):
        out = {}
        out["ts"] = torch.stack([b["ts"] for b in batch])
        out["ts_len"] = torch.tensor([b["ts_len"] for b in batch])
        out["text_embedding_all_segments"] = torch.stack([b["text_embedding_all_segments"] for b in batch])
        out["moment_embed"] = torch.stack([b["moment_embed"] for b in batch]) if batch[0][
                                                                                     "moment_embed"] is not None else None
        out["image_id"] = [b["image_id"] for b in batch]
        out["ts_id"] = torch.tensor([b["ts_id"] for b in batch])
        out["caps"] = [b["caps"] for b in batch]
        # out["attn_mask"] = torch.stack([b["attn_mask"] for b in batch])

        return out
