from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cockroach.brain import Brain, BrainConfig
from cockroach.config import WorldConfig
from cockroach.world import OBS_NAMES, Action

FORMAT_VERSION = 1


def save(path: str | Path, brains: list[Brain], config: WorldConfig, training: dict):
    """Atomic, non-pickle checkpoint. A running viewer can safely reload it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": FORMAT_VERSION,
        "world": asdict(config),
        "observations": list(OBS_NAMES),
        "actions": [a.name for a in Action],
        "brains": [b.metadata() for b in brains],
        "training": training,
    }
    arrays = {"metadata": np.array(json.dumps(metadata))}
    for i, brain in enumerate(brains):
        arrays.update(brain.arrays(f"brain{i}"))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as file:
            temporary = file.name
            np.savez_compressed(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def load(path: str | Path, training: bool = False):
    try:
        with np.load(path, allow_pickle=False) as arrays:
            meta = json.loads(str(arrays["metadata"]))
            if (
                meta["version"] != FORMAT_VERSION
                or meta["observations"] != list(OBS_NAMES)
                or meta["actions"] != [a.name for a in Action]
                or len(meta["brains"]) != 2
            ):
                raise ValueError("Model biçimi bu sürümle uyumlu değil.")
            config = WorldConfig(**meta["world"])
            brains = []
            for i, data in enumerate(meta["brains"]):
                brain = Brain(i, BrainConfig(**data["config"]), training=training)
                brain.restore(data, arrays, f"brain{i}")
                brains.append(brain)
            return brains, config, meta["training"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Geçersiz model dosyası: {path}") from error
