# dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


# Keep your original numeric mapping, but make it explicit and reusable.
LABEL_MAP = {
    "happy": 0,
    "sad": 1,
    "angry": 2,
    "relaxed": 3,
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    targets: torch.Tensor


class SongLyricsDataset(Dataset):
    def __init__(self, lyrics, moods, tokenizer: BertTokenizer, max_len: int):
        self.lyrics = lyrics
        self.moods = moods
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.lyrics)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        lyric = str(self.lyrics[idx])
        mood = str(self.moods[idx]).strip().lower()

        if mood not in LABEL_MAP:
            raise ValueError(f"Unknown mood label: {mood}")

        encoding = self.tokenizer(
            lyric,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),        # [max_len]
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "targets": torch.tensor(LABEL_MAP[mood], dtype=torch.long),
        }


def create_data_loader(df, tokenizer: BertTokenizer, max_len: int, batch_size: int) -> DataLoader:
    ds = SongLyricsDataset(
        lyrics=df["Lyrics"].to_numpy(),
        moods=df["Mood"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=0)
