from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import config as C 
from bench.common import RunResult, accuracy_vs_reference, parse_gnu_time 
 
EZKL = shutil.which("ezkl") or "ezkl" #finds the executable path 

def _run(cmd: list[str], timed: bool = False):
 
    if timed and shutil.which("/usr/bin/time"):
        cmd = ["/usr/bin/time", "-v", *cmd]

    start = time.perf_counter()

    p = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    gnu = parse_gnu_time(p.stderr) if timed else None


    return elapsed, p.stdout, p.stderr, p.returncode, gnu

#calculate the size
def _solc_bytecode_size(sol_path: Path):

    try:
        p = subprocess.run(["solc", "--bin", "--optimize", str(sol_path)],capture_output=True, text=True, timeout=120)

        m = re.search(r"Binary:\s*\n([0-9a-fA-F]+)", p.stdout)

        return len(m.group(1)) // 2 if m else None
    
    except Exception:
        return None


def run_once(artifact: Path, compute: str, repeat: int, work: Path, logrows_max: int) -> RunResult:
    #locating input
    meta = json.loads((artifact / "meta.json").read_text())
    model = str(artifact / "model.onnx")
    data = str(artifact / "input.json")

    #input information
    work.mkdir(parents=True, exist_ok=True)
    settings = str(work / "settings.json")
    compiled = str(work / "model.compiled")
    srs = str(work / "kzg.srs")
    vk = str(work / "vk.key")
    pk = str(work / "pk.key")
    witness = str(work / "witness.json")
    proof = str(work / "proof.json")
    sol = work / "verify.sol"
    abi = str(work / "verify.abi")

    notes, timings = [], {}

    #stage timing
    def step(name, cmd, timed=False):
        el, out, err, rc, gnu = _run(cmd, timed=timed)
        timings[name] = el
        if rc != 0:
            notes.append(f"{name} rc={rc}: {err.strip()[-200:]}")
        return out, err, rc, gnu

    #ezkl pipeline
    #settings with paper visibility
    step("gen_settings", [
        EZKL, "gen-settings", "-M", model, "-O", settings,
        "--input-visibility", "private",
        "--param-visibility", "fixed",
        "--output-visibility", "public",
    ])

    #runs the model on real data
    step("calibrate", [
        EZKL, "calibrate-settings", "-M", model, "-D", data, "-O", settings, "--max-logrows", str(logrows_max), "--target", "resources",
    ])

    #SRS sized from the calibrated settings
    step("get_srs", [EZKL, "get-srs", "--srs-path", srs, "--settings-path", settings])

    #compile
    step("compile", [EZKL, "compile-circuit", "-M", model, "--compiled-circuit", compiled, "-S", settings])

    #setup (keygen)
    _, _, _, _ = step("setup", [
        EZKL, "setup", "-M", compiled, "--srs-path", srs, "--vk-path", vk, "--pk-path", pk,
    ])


    #witness
    step("witness", [EZKL, "gen-witness", "-D", data, "-M", compiled, "-O", witness])

    #prove
    out, _, rc_prove, gnu = step("prove", [
        EZKL, "prove", "--witness", witness, "-M", compiled,
        "--pk-path", pk, "--proof-path", proof, "--srs-path", srs,
    ], timed=True)

    peak_mem = gnu["peak_mem_mb"] if gnu else None

    cpu_pct = gnu["cpu_pct"] if gnu else None

    # EZKL logs "proof took <sec>"; prefer it over wall clock when present.
    m = re.search(r"proof took (\d+\.\d+)", out)
    prove_s = float(m.group(1)) if m else timings["prove"]

    #verify
    _, _, rc_verify, _ = step("verify", [
        EZKL, "verify", "--proof-path", proof, "--settings-path", settings, "--vk-path", vk, "--srs-path", srs,
    ])
    verified = rc_verify == 0

    #EVM verifier sizes
    verifier_bytes = bytecode = None
    step("evm_verifier", [
        EZKL, "create-evm-verifier", "--vk-path", vk, "--srs-path", srs,
        "--settings-path", settings, "--sol-code-path", str(sol), "--abi-path", abi,
    ])

    if sol.exists():
        verifier_bytes = sol.stat().st_size
        bytecode = _solc_bytecode_size(sol)

    proof_bytes = Path(proof).stat().st_size if Path(proof).exists() else None
    vk_bytes = Path(vk).stat().st_size if Path(vk).exists() else None
    mean_err = median_err = None

    #format the output data
    try:
        pdata = json.loads(Path(proof).read_text())
        rescaled = pdata["pretty_public_inputs"]["rescaled_outputs"][0]
        ref = json.loads((artifact / "input.json").read_text())["pytorch_output"][0]
        acc = accuracy_vs_reference([float(x) for x in ref], [float(x) for x in rescaled])
        mean_err, median_err = acc["mean_abs_err"], acc["median_abs_err"]

    except Exception as exc:
        notes.append(f"acc skipped: {exc}")

    status = "ok" if verified else "error"

    return RunResult(
        system="ezkl", 
        model=meta["model"],
        model_type=meta["model_type"],
        compute=compute, 
        params=meta.get("params"),
        setup_s=timings.get("setup"), 
        witness_s=timings.get("witness"),
        prove_s=prove_s, 
        verify_s=timings.get("verify"),
        total_s=sum(timings.values()), 
        proof_bytes=proof_bytes, 
        vk_bytes=vk_bytes,
        verifier_bytes=verifier_bytes, 
        verifier_bytecode_bytes=bytecode,
        peak_mem_mb=peak_mem, 
        cpu_pct=cpu_pct,
        mean_abs_err=mean_err,
        median_abs_err=median_err,
        verified=verified, 
        repeat=repeat, 
        status=status, 
        notes="; ".join(notes),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--compute", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--repeats", type=int, default=C.DEFAULT_REPEATS)
    ap.add_argument("--logrows-max", type=int, default=C.EZKL_LOGROWS_MAX)
    args = ap.parse_args()

    C.ensure_dirs()
    work_root = C.RESULTS_DIR / "_work" / "ezkl"
    for r in range(args.repeats):
        print(f"[ezkl] repeat {r+1}/{args.repeats} on {args.artifact}")
        res = run_once(args.artifact, args.compute, r, work_root / f"r{r}", args.logrows_max)
        path = res.save()
        print(f"  -> prove={res.prove_s:.3f}s verify={res.verify_s}s proof={res.proof_bytes}B "f"verified={res.verified}  [{path.name}]")


if __name__ == "__main__":
    main()
