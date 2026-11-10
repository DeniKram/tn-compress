"""Short fine-tune of a compressed model to recover accuracy ("healing")."""
from __future__ import annotations

import torch


def heal(model, loader, device, epochs: int = 1, lr: float = 3e-5) -> None:
    model.train().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    use_bf16 = device.type == "cuda"
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
