"""Evaluate every saved adapter checkpoint under one identical benchmark setup."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def checkpoint_paths(run_dir: Path) -> list[Path]:
    paths = sorted(
        path
        for path in run_dir.glob("checkpoint-*")
        if path.is_dir()
    )
    final_adapter = run_dir / "final_adapter"
    if final_adapter.is_dir():
        paths.append(final_adapter)
    if not paths:
        raise FileNotFoundError(f"No checkpoint-* or final_adapter found under {run_dir}.")
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--system-prompt-condition", default="neutral")
    parser.add_argument("--prompt-variants", default="original,swap_options")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/checkpoint_evaluations"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for adapter in checkpoint_paths(args.run_dir):
        command = [
            sys.executable,
            "-m",
            "sycophancy_rl.evaluation.run_benchmark",
            "--adapter",
            str(adapter),
            "--benchmark",
            str(args.benchmark),
            "--run-name",
            f"{args.run_dir.name}-{adapter.name}-seed{args.seed}",
            "--system-prompt-condition",
            args.system_prompt_condition,
            "--prompt-variants",
            args.prompt_variants,
            "--seed",
            str(args.seed),
            "--output-dir",
            str(args.output_dir),
        ]
        if args.max_examples is not None:
            command.extend(["--max-examples", str(args.max_examples)])
        if args.load_in_4bit:
            command.append("--load-in-4bit")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
