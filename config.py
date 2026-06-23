"""Central configuration for the ZKML comparison benchmark.

One place for paths and hyper-parameters so every script (data loader, trainer,
exporter, runners, analysis) agrees on shapes and file locations.

The benchmark is an apples-to-apples comparison: a single LeNet5 CNN whose
architecture is *identical* to the one hardcoded in zkCNN
(``external/zkCNN/src/models.cpp``):
    Conv(1,6,5) -> ReLU -> MaxPool2 -> Conv(6,16,5) -> ReLU -> MaxPool2
    -> FC(400,120) -> ReLU -> FC(120,84) -> ReLU -> FC(84,10)
The same trained weights run on all three systems (ONNX for EZKL/DeepProve, a
flattened text stream for zkCNN). Input is a 32x32x1 "image" built from a
1024-hour consumption window; the label is a 10-bucket quantile class of
near-future demand, so accuracy is comparable across systems via argmax.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# --- external (vendored upstream repos; cloned by scripts/01_install_systems) -
EXTERNAL = REPO_ROOT / "external"
ZKPET_DIR = EXTERNAL / "zkpet"
ZKCNN_DIR = EXTERNAL / "zkCNN"
DEEPPROVE_DIR = EXTERNAL / "deep-prove"

# --- dataset ----------------------------------------------------------------
# Prefer a local copy under data/; fall back to the vendored zkpet dataset.
_DATA_LOCAL = REPO_ROOT / "data" / "electricity.csv"
_DATA_VENDORED = ZKPET_DIR / "dataset" / "electricity.csv"
ELECTRICITY_CSV = Path(os.environ.get("ELECTRICITY_CSV", _DATA_LOCAL if _DATA_LOCAL.exists() else _DATA_VENDORED))

# --- artifacts / results ----------------------------------------------------
ARTIFACTS_DIR = REPO_ROOT / "models" / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
FIGURES_DIR = RESULTS_DIR / "figures"

# ---------------------------------------------------------------------------
# LeNet5 classification (must match zkCNN's hardcoded lenet)
# ---------------------------------------------------------------------------
CNN_IMG = 32                 # 32x32 input (LeNet5 with no MNIST padding branch)
CNN_CHANNELS = 1
CNN_WINDOW = CNN_IMG * CNN_IMG   # 1024 consecutive hourly readings per sample
CNN_CLASSES = 10             # 10 quantile buckets of near-future demand
CNN_HORIZON = 24             # bucket is computed from the mean of the next 24h

# zkCNN internal quantization bit-width (informational; see neuralNetwork.hpp Q)
ZKCNN_Q = 9

# ---------------------------------------------------------------------------
# Benchmark defaults
# ---------------------------------------------------------------------------
DEFAULT_REPEATS = 5
NUM_SAMPLES = 30             # IO samples per run (matches deep-prove bench default)
EZKL_LOGROWS_MAX = 24        # calibration ceiling
# DeepProve quantization bit-width env var (ZKML_BIT_LEN); 8/10/12/16 typical.
DEEPPROVE_BIT_LEN = int(os.environ.get("ZKML_BIT_LEN", "12"))


def ensure_dirs() -> None:
    for d in (ARTIFACTS_DIR, RESULTS_DIR, RAW_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
