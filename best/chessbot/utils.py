# chessbot/utils.py
import os
import json
import torch
from datetime import datetime


# -----------------------------
# Device helpers
# -----------------------------
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# JSON helpers
# -----------------------------
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# -----------------------------
# Model I/O
# -----------------------------
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[{timestamp()}] Saved model to {path}")


def load_model(model, path, device=None):
    device = device or get_device()
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"[{timestamp()}] Loaded model from {path}")
    return model


# -----------------------------
# Safe dictionary lookup
# -----------------------------
def safe_get(d, key, default=None):
    return d[key] if key in d else default


# -----------------------------
# Filesystem helpers
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_files(path, suffix=None):
    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if suffix is None or f.endswith(suffix)
    ]
    return sorted(files)


# -----------------------------
# Misc
# -----------------------------
def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")