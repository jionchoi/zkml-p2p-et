
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import config as C 
from bench.common import RunResult, parse_gnu_time

CAVEAT = "zkCNN"

#result scraping
def _parse_result_row(stdout: str):
    
    for line in reversed(stdout.strip().splitlines()):

        parts = [p.strip() for p in line.split(",")]

        parts = [p for p in parts if p != ""]

        if len(parts) >= 16:
            def f(i):
                try:
                    return float(parts[i])
                except ValueError:
                    return None
            return f(13), f(14), f(15)
        
    return None, None, None


def run_once(artifact: Path, zkcnn_bin: Path, repeat: int, work: Path) -> RunResult:
    #same thing again just input ready 
    meta = json.loads((artifact / "meta.json").read_text())
    in_file = artifact / "zkcnn_input.txt"
    cfg_file = artifact / "zkcnn_config.txt"
    if not in_file.exists():
        raise FileNotFoundError(f"{in_file} missing — run `python -m models.export`")
    work.mkdir(parents=True, exist_ok=True)
    out_csv = work / f"infer_r{repeat}.csv"

    #command to run the system
    cmd = [
        "/usr/bin/time", "-v",
        str(zkcnn_bin), str(in_file), str(cfg_file), str(out_csv), "1",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    gnu = parse_gnu_time(proc.stderr)
    prove_s, verify_s, proof_kb = _parse_result_row(proc.stdout)

    status = "ok" if proc.returncode == 0 and prove_s is not None else "error"
    notes = [CAVEAT]

    if status == "error":
        notes.append(f"rc={proc.returncode}; stderr_tail={proc.stderr[-300:]}")

    return RunResult(
        system="zkcnn", 
        model=meta["model"], 
        model_type="cnn",
        compute="cpu", 
        params=meta.get("params"),
        prove_s=prove_s, 
        verify_s=verify_s,
        proof_bytes=int(proof_kb * 1024) if proof_kb is not None else None,
        peak_mem_mb=gnu["peak_mem_mb"], 
        cpu_pct=gnu["cpu_pct"], 
        total_s=gnu["wall_s"],
        verified=(status == "ok"), 
        repeat=repeat, 
        status=status, 
        notes="; ".join(notes),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=C.ARTIFACTS_DIR / "cnn")
    ap.add_argument("--zkcnn-bin", type=Path, default=C.ZKCNN_DIR / "cmake-build-release" / "src" / "demo_lenet_run")
    ap.add_argument("--repeats", type=int, default=C.DEFAULT_REPEATS)
    args = ap.parse_args()

    if not args.zkcnn_bin.exists():
        raise SystemExit(f"zkCNN binary not found at {args.zkcnn_bin}. Build it via "f"scripts/01_install_systems.sh (runs script/demo_lenet.sh once).")
    
    C.ensure_dirs()

    work = C.RESULTS_DIR / "_work" / "zkcnn"

    for r in range(args.repeats):
        print(f"[zkcnn] repeat {r+1}/{args.repeats}")
        res = run_once(args.artifact, args.zkcnn_bin, r, work)
        path = res.save()
        print(f"  -> prove={res.prove_s}s verify={res.verify_s}s proof={res.proof_bytes}B "f"mem={res.peak_mem_mb}MB status={res.status}  [{path.name}]")


if __name__ == "__main__":
    main()
