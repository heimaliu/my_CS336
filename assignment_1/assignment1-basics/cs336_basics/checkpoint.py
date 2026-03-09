from __future__ import annotations

import os
import pickle
from typing import BinaryIO, IO

import torch


def _dump_payload(payload: dict, out: str | os.PathLike | BinaryIO | IO[bytes]) -> None:
    if isinstance(out, (str, os.PathLike)):
        with open(out, "wb") as f:
            pickle.dump(payload, f)
    else:
        pickle.dump(payload, out)


def _load_payload(src: str | os.PathLike | BinaryIO | IO[bytes]) -> dict:
    if isinstance(src, (str, os.PathLike)):
        with open(src, "rb") as f:
            return pickle.load(f)
    return pickle.load(src)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    _dump_payload(payload, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    payload = _load_payload(src)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload["iteration"]
