
# EZKL (Python/Windows), zkCNN(C++/WSL) and DeepProve (Rust/WSL)

from __future__ import annotations

import dataclasses
import json
import os
import platform
import socket
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

#
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
ARTIFACTS_DIR = REPO_ROOT / "models" / "artifacts"

# Valid enum-ish values (kept as plain strings for portability across languages).
SYSTEMS = ("ezkl", "zkcnn", "deepprove")
MODEL_TYPES = ("cnn",)
COMPUTE = ("cpu", "gpu")
STATUSES = ("ok", "error", "oom", "unsupported", "skipped")

SCHEMA_VERSION = 2


@dataclass
class RunResult:
    """One proving+verification run of one model on one system.

    Times are seconds, sizes are bytes, memory is MB. ``None`` means "not
    measured / not applicable for this system" (e.g. zkCNN has no GPU path, so
    its rows carry ``compute='cpu'`` and a note explaining the caveat).
    """

    system: str
    model: str
    model_type: str
    compute: str
    params: Optional[int] = None

    # --- timing (seconds) -------------------------------------------------
    setup_s: Optional[float] = None       # keygen / SRS load / preprocessing
    witness_s: Optional[float] = None     # witness / advice generation
    prove_s: Optional[float] = None       # proof generation (the headline metric)
    verify_s: Optional[float] = None      # verification
    total_s: Optional[float] = None       # end-to-end wall clock for the run

    # --- artifacts / resources -------------------------------------------
    proof_bytes: Optional[int] = None
    vk_bytes: Optional[int] = None            # verifying key size (where exposed)
    verifier_bytes: Optional[int] = None      # EZKL: .sol verifier source size
    verifier_bytecode_bytes: Optional[int] = None  # EZKL: compiled bytecode size
    peak_mem_mb: Optional[float] = None       # peak RSS (psutil or /usr/bin/time -v)
    cpu_pct: Optional[float] = None           # avg %CPU (>100 => multicore), like the paper

    # --- accuracy / fidelity ---------------------------------------------
    mean_abs_err: Optional[float] = None      # |float_out - quantized_out| mean
    median_abs_err: Optional[float] = None    # ... median
    verified: Optional[bool] = None           # did verification return true?

    # --- bookkeeping ------------------------------------------------------
    repeat: int = 0
    status: str = "ok"                # one of STATUSES
    notes: str = ""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    host: str = field(default_factory=socket.gethostname)
    os: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")
    schema_version: int = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)  # system-specific spillover

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if self.system not in SYSTEMS:
            raise ValueError(f"system must be one of {SYSTEMS}, got {self.system!r}")
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {self.model_type!r}")
        if self.compute not in COMPUTE:
            raise ValueError(f"compute must be one of {COMPUTE}, got {self.compute!r}")
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, raw_dir: Path | str = RAW_DIR) -> Path:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{self.system}_{self.model}_{self.compute}_r{self.repeat}_{self.run_id}.json"
        path = raw_dir / fname
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


# Column order for the collated CSV (stable, human-first).
CSV_COLUMNS = [f.name for f in dataclasses.fields(RunResult) if f.name != "extra"]


# ---------------------------------------------------------------------------
# Timing + memory helpers
# ---------------------------------------------------------------------------
class StageTimer:
    """Accumulates named wall-clock stage timings.

    Usage::

        t = StageTimer()
        with t.stage("prove"):
            ...
        result.prove_s = t["prove"]
    """

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = time.perf_counter() - start

    def __getitem__(self, name: str) -> Optional[float]:
        return self.timings.get(name)

    @property
    def total(self) -> float:
        return time.perf_counter() - self._t0


def peak_rss_mb() -> Optional[float]:
    """Best-effort peak resident memory of the current process, in MB.

    Used by the EZKL runner (in-process). zkCNN/DeepProve runners measure peak
    memory of the child process via ``/usr/bin/time -v`` instead (see
    ``parse_gnu_time``).
    """
    try:
        import psutil  # noqa: PLC0415

        proc = psutil.Process(os.getpid())
        # On Windows there is no RUSAGE peak; use current RSS as a floor.
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            import resource  # POSIX only  # noqa: PLC0415

            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports kB, macOS reports bytes.
            scale = 1024 if platform.system() == "Linux" else 1
            return ru * scale / (1024 * 1024)
        except Exception:
            return None


def parse_gnu_time(stderr_text: str) -> dict[str, Optional[float]]:
    """Parse ``/usr/bin/time -v`` stderr into {peak_mem_mb, cpu_pct, wall_s}.

    The WSL-side runners wrap their child process in ``/usr/bin/time -v`` so we
    get a real peak RSS for C++/Rust binaries we don't control.
    """
    out: dict[str, Optional[float]] = {"peak_mem_mb": None, "cpu_pct": None, "wall_s": None}
    for line in stderr_text.splitlines():
        line = line.strip()
        if line.startswith("Maximum resident set size"):
            kb = float(line.split(":")[-1].strip())
            out["peak_mem_mb"] = kb / 1024.0
        elif line.startswith("Percent of CPU this job got"):
            out["cpu_pct"] = float(line.split(":")[-1].strip().rstrip("%"))
        elif line.startswith("Elapsed (wall clock) time"):
            out["wall_s"] = _parse_walltime(line.split("):")[-1].strip())
    return out


def _parse_walltime(s: str) -> Optional[float]:
    # Formats: "h:mm:ss" or "m:ss.ss"
    try:
        parts = s.split(":")
        parts = [float(p) for p in parts]
        sec = 0.0
        for p in parts:
            sec = sec * 60 + p
        return sec
    except Exception:
        return None


def accuracy_vs_reference(float_out, quant_out) -> dict[str, float]:
    """Mean/median absolute error between float inference and quantized proof output.

    Matches EZKL's ``mean_abs_error`` / ``median_abs_error`` definition so all
    three systems report accuracy the same way. Accepts lists or numpy arrays.
    """
    import numpy as np  # noqa: PLC0415

    a = np.asarray(float_out, dtype=np.float64).ravel()
    b = np.asarray(quant_out, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    diff = np.abs(a[:n] - b[:n])
    return {
        "mean_abs_err": float(diff.mean()) if n else float("nan"),
        "median_abs_err": float(np.median(diff)) if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------
def collate(raw_dir: Path | str = RAW_DIR, out_csv: Path | str = RESULTS_DIR / "results.csv"):
    """Concatenate every ``results/raw/*.json`` into one tidy CSV (and return a DataFrame)."""
    import pandas as pd  # noqa: PLC0415

    raw_dir = Path(raw_dir)
    rows = []
    for jf in sorted(raw_dir.glob("*.json")):
        try:
            d = json.loads(jf.read_text())
            d.pop("extra", None)
            rows.append(d)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[collate] skipping {jf.name}: {exc}")
    df = pd.DataFrame(rows)
    if not df.empty:
        cols = [c for c in CSV_COLUMNS if c in df.columns]
        df = df[cols + [c for c in df.columns if c not in cols]]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[collate] {len(df)} rows -> {out_csv}")
    return df


if __name__ == "__main__":
    # `python -m bench.common` collates whatever raw rows exist.
    collate()
