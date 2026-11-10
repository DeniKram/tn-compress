"""SST-2 (binary sentiment) from GLUE: 67k train sentences, 872 validation sentences."""
from __future__ import annotations

from datasets import load_dataset
from torch.utils.data import DataLoader


def sst2_loader(tokenizer, split: str, batch_size: int, limit: int | None = None, shuffle=False):
    ds = load_dataset("nyu-mll/glue", "sst2", split=split)
    if limit is not None:
        ds = ds.shuffle(seed=0).select(range(min(limit, len(ds))))

    def collate(batch):
        enc = tokenizer([b["sentence"] for b in batch], padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc["labels"] = __import__("torch").tensor([b["label"] for b in batch])
        return enc

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
