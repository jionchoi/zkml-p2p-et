# runnder for deep prove. it runs cargo run --release [--features cuda] --bin bench -- -o model.onnx -i input.json --bench <csv> --num-samples N and emits two CSVs

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


import config as C  
from bench.common import RunResult, parse_gnu_time 

ZKML_CRATE = C.DEEPPROVE_DIR / "zkml" #find the path for the deep prove from the external library

def _read_metrics_csv(path: Path):
    #Return (accuracy, proof_kb) averaged over rows of the bench metrics CSV.

    accuracy, sizes = [], []

    with path.open() as file:
        for row in csv.DictReader(file):

            if "accuracy" in row and row["accuracy"]:

                accuracy.append(float(row["accuracy"]))

            if "proof_size" in row and row["proof_size"]:

                sizes.append(float(row["proof_size"]))


    def avg(xs):
        return sum(xs) / len(xs) if xs else None

    return avg(accuracy), avg(sizes)

# 
def _read_timed_csv(path: Path):

    #the output format for time comparison
    out = {"setup_s": None, "witness_s": None, "prove_s": None, "verify_s": None}


    #this is from the deep prove csv
    buckets = {"Context": [], "Inference": [], "Proving": [], "Verify": []}

    with path.open() as file:
        for row in csv.DictReader(file):

            try:
                elapsed_s = float(row["elapsed"]) / 1000.0 

            except (KeyError, ValueError):
                continue


            label = row["name"]

            for key in buckets:
                if label.startswith(key):
                    buckets[key].append(elapsed_s)

    def avg(xs):
        return sum(xs) / len(xs) if xs else None

    #set the output
    out["setup_s"] = avg(buckets["Context"])
    out["witness_s"] = avg(buckets["Inference"])
    out["prove_s"] = avg(buckets["Proving"])
    out["verify_s"] = avg(buckets["Verify"])
    return out

#run deep prove. 
def run_once(artifact: Path, compute: str, repeat: int, work: Path, num_samples: int, bit_len: int) -> RunResult:
    
    #locating the inputs 
    meta = json.loads((artifact / "meta.json").read_text())
    onnx = (artifact / "model.onnx").resolve()
    io = (artifact / "input.json").resolve()
    work.mkdir(parents=True, exist_ok=True)
    metrics_csv = (work / f"metrics_r{repeat}.csv").resolve()

    features = ["--features", "cuda"] if compute == "gpu" else []

    # command for building it. This runs the cargo run
    cmd = [
        "/usr/bin/time", "-v",
        "cargo", "run", "--release", *features, "--bin", "bench", "--",
        "-o", str(onnx), "-i", str(io),
        "--bench", str(metrics_csv),
        "--num-samples", str(num_samples),
    ]

    env = os.environ.copy()
    env["ZKML_BIT_LEN"] = str(bit_len)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ZKML_CRATE), env=env)

    # get the three data sources
    gnu = parse_gnu_time(proc.stderr)
    accuracy, proof_kb = _read_metrics_csv(metrics_csv)
    timed = _read_timed_csv(metrics_csv)

    status = "ok" if proc.returncode == 0 else "error"
    notes = [f"bit_len={bit_len}"]
    if status == "error":
        notes.append(f"rc={proc.returncode}; stderr_tail={proc.stderr[-300:]}")

    #combine the result for the output 
    return RunResult(
        system="deepprove", 
        model=meta["model"], 
        model_type=meta["model_type"],
        compute=compute, 
        params=meta.get("params"),
        setup_s=timed["setup_s"],
        witness_s=timed["witness_s"],
        prove_s=timed["prove_s"], 
        verify_s=timed["verify_s"], 
        total_s=gnu["wall_s"],
        proof_bytes=int(proof_kb * 1024) if proof_kb is not None else None,
        peak_mem_mb=gnu["peak_mem_mb"], 
        cpu_pct=gnu["cpu_pct"],
        verified=(status == "ok"), 
        repeat=repeat, 
        status=status,
        notes="; ".join(notes), 
        extra={"argmax_accuracy": accuracy},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--compute", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--repeats", type=int, default=C.DEFAULT_REPEATS)
    ap.add_argument("--num-samples", type=int, default=C.NUM_SAMPLES)
    ap.add_argument("--bit-len", type=int, default=C.DEEPPROVE_BIT_LEN)

    args = ap.parse_args()

    if not (ZKML_CRATE / "Cargo.toml").exists():
        raise SystemExit(f"deep-prove zkml crate not found at {ZKML_CRATE}. "f"Clone+build via scripts/01_install_systems.sh.")
    
    C.ensure_dirs()

    work = C.RESULTS_DIR / "_work" / "deepprove"

    for r in range(args.repeats):

        print(f"[deepprove] repeat {r+1}/{args.repeats} ({args.compute})")
        res = run_once(args.artifact, args.compute, r, work, args.num_samples, args.bit_len)
        path = res.save()
        print(f"  -> prove={res.prove_s}s verify={res.verify_s}s proof={res.proof_bytes}B "f"mem={res.peak_mem_mb}MB status={res.status}  [{path.name}]")


if __name__ == "__main__":
    main()
