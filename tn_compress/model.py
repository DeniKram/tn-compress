"""Load DistilBERT fine-tuned on SST-2 and swap its FFN layers for compressed ones.

DistilBERT has 6 transformer blocks; each FFN has
    lin1: Linear(768 -> 3072)   weight (3072, 768)
    lin2: Linear(3072 -> 768)   weight (768, 3072)
These 12 matrices hold ~28M of the model's ~67M parameters.
"""
from __future__ import annotations

import copy

import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tn_compress.layers import LowRankLinear, MPOLinear

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# 768 = 4*6*4*8,  3072 = 4*8*8*12
MODES_768 = [4, 6, 4, 8]
MODES_3072 = [4, 8, 8, 12]


def load_base():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tok, model


def ffn_layers(model) -> list[tuple[nn.Module, str, nn.Linear]]:
    """(parent module, attribute name, linear) for every FFN linear layer."""
    found = []
    for block in model.distilbert.transformer.layer:
        found.append((block.ffn, "lin1", block.ffn.lin1))
        found.append((block.ffn, "lin2", block.ffn.lin2))
    return found


def ffn_params(model) -> int:
    total = 0
    for _, _, layer in ffn_layers(model):
        if isinstance(layer, nn.Linear):
            total += layer.weight.numel()
        else:
            total += layer.n_params()
    return total


def compress(model, method: str, value: int, only: set[int] | None = None):
    """Return a deep copy of `model` with its FFN layers compressed.

    method="svd": value = rank r
    method="mpo": value = bond dimension chi
    only: indices into ffn_layers() to compress (0..11); None = all 12.
          Layer 2k is block k's lin1 (768 -> 3072), 2k+1 is its lin2 (3072 -> 768).
    """
    model = copy.deepcopy(model)
    for idx, (parent, name, lin) in enumerate(ffn_layers(model)):
        if only is not None and idx not in only:
            continue
        out_f, in_f = lin.weight.shape
        if method == "svd":
            new = LowRankLinear(lin, rank=value)
        elif method == "mpo":
            out_modes = MODES_3072 if out_f == 3072 else MODES_768
            in_modes = MODES_3072 if in_f == 3072 else MODES_768
            new = MPOLinear(lin, out_modes, in_modes, chi=value)
        else:
            raise ValueError(method)
        setattr(parent, name, new.to(lin.weight.device))
    return model
