from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval().to(device)
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        correct += (logits.argmax(-1) == batch["labels"]).sum().item()
        total += batch["labels"].numel()
    return correct / total
