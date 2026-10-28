"""Rerun one country's two-dimensional comparison from the fixed inputs.

The stages are explicit because full native estimation can take days. Statistical
settings and estimator implementations are those used for the published analysis.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

from unpack_inputs import unpack

ROOT = Path(__file__).resolve().parent
SIZES = {"chile": ["300", "1000", "5000", "15000", "full"],
         "france": ["300", "1000", "full"], "brazil": ["300", "full"]}
TUNING_SIZE = {"chile": "5000", "france": "1000", "brazil": "full"}
MODELS = ["B1", "B2", "B3", "B4"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", choices=list(SIZES), required=True)
    parser.add_argument("--stage", choices=["prepare", "tune", "fit", "native", "evaluate", "all"], required=True)
    parser.add_argument("--rscript", default=os.environ.get("DYNAMAP_RSCRIPT", "Rscript"))
    parser.add_argument("--days", type=float, default=7,
                        help="Wall budget for native Bayesian estimation in this invocation.")
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be positive")
    unpack()
    country = args.country
    stages = ["prepare", "tune", "fit", "native", "evaluate"] if args.stage == "all" else [args.stage]
    deadline = time.time() + args.days * 86400
    environment = os.environ.copy()
    environment.update(DYNAMAP_RSCRIPT=args.rscript, CUDA_VISIBLE_DEVICES="-1",
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")

    def run(command):
        print(" ".join(map(str, command)), flush=True)
        subprocess.run(list(map(str, command)), cwd=ROOT, env=environment, check=True)

    def input_path(size):
        return ROOT / "nativeinputs" / country / size

    def prepared(size, variant="base"):
        return ROOT / "prepared" / country / size / variant

    for stage in stages:
        if stage == "prepare":
            for size in SIZES[country]:
                variants = ["base", "item_shrink4", "alternate_anchors"] if size == TUNING_SIZE[country] else ["base"]
                for variant in variants:
                    output = prepared(size, variant)
                    if not (output / "manifest.json").exists():
                        run([args.rscript, "--vanilla", "src/classic/prepare.R", input_path(size), output, variant])
        elif stage == "tune":
            size = TUNING_SIZE[country]
            for model in MODELS:
                run([sys.executable, "-m", "src.neural", "--stage", "tune", "--kind", model,
                     "--input", input_path(size), "--prepared", prepared(size),
                     "--out", ROOT / "results/tuning" / country / model])
        elif stage == "fit":
            for size in SIZES[country]:
                for model in MODELS:
                    run([sys.executable, "-m", "src.neural", "--stage", "fit", "--kind", model,
                         "--input", input_path(size), "--prepared", prepared(size),
                         "--selection", ROOT / "results/tuning" / country / model / "selection.json",
                         "--out", ROOT / "results/neural" / country / size / model,
                         "--seeds", "301,401,501"])
        elif stage == "native":
            for size in SIZES[country]:
                run([args.rscript, "--vanilla", "src/classic/wnominate_full.R", input_path(size),
                     prepared(size), ROOT / "results/wnominate" / country / size, "101"])
                variants = ["base", "item_shrink4", "alternate_anchors"] if size == TUNING_SIZE[country] else ["base"]
                for variant in variants:
                    run([sys.executable, "src/classic/condition.py", "--input", input_path(size),
                         "--prepared", prepared(size, variant), "--output",
                         ROOT / "results/classic" / country / size / variant, "--deadline", deadline])
        elif stage == "evaluate":
            # All country configurations and final fits must exist before test is opened.
            for size in SIZES[country]:
                for model in MODELS:
                    assert (ROOT / "results/neural" / country / size / model / "complete.json").exists()
            run([sys.executable, "src/finalize.py", "--root", ROOT, "--action", "seal", "--country", country])
            for size in SIZES[country]:
                run([sys.executable, "src/finalize.py", "--root", ROOT, "--action", "evaluate",
                     "--country", country, "--size", size])


if __name__ == "__main__":
    main()
