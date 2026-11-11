"""Command-line interface for LongMorph data, models, and evaluation."""

from __future__ import annotations

import argparse
from importlib import import_module
import sys

COMMANDS = {
    ("prepare", "reference"): (
        "longmorph.data.reference",
        "Build reference data from explicit source splits.",
    ),
    ("prepare", "ppg-dalia"): ("longmorph.data.ppg_dalia", "Preprocess PPG-DaLiA recordings."),
    ("prepare", "assets"): (
        "longmorph.generation.assets",
        "Build TRAIN-only scoring and retrieval assets.",
    ),
    ("train", "codec"): ("longmorph.training.codec", "Train coarse/detail cycle codecs."),
    ("train", "hybrid"): ("longmorph.training.hybrid", "Assemble a mixed-resolution hybrid codec."),
    ("train", "condition"): (
        "longmorph.training.condition",
        "Train the cycle-condition transition.",
    ),
    ("train", "flow"): ("longmorph.training.flow", "Train the coupled coarse/detail block Flow."),
    ("generate",): (
        "longmorph.generation.generate",
        "Generate recursively from observed contexts.",
    ),
    ("evaluate",): ("longmorph.evaluation.generation", "Evaluate paired generated futures."),
    ("downstream", "forecast"): (
        "longmorph.downstream.forecasting",
        "Evaluate forecasting protocols and models.",
    ),
    ("downstream", "heart-rate"): (
        "longmorph.downstream.heart_rate",
        "Build, train, and aggregate the HR task.",
    ),
    ("downstream", "export"): ("longmorph.downstream.export", "Export aligned generated cycles."),
    ("downstream", "summarize"): (
        "longmorph.downstream.reporting",
        "Aggregate downstream results.",
    ),
    ("baseline",): ("longmorph.baselines.cli", "List or train optional generative baselines."),
}


def parser():
    result = argparse.ArgumentParser(
        prog="longmorph",
        description="LongMorph: cycle-based PPG generation and evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n"
        + "\n".join("  " + " ".join(key).ljust(27) + value[1] for key, value in COMMANDS.items())
        + "\n\nUse COMMAND --help for its input schema and options.",
    )
    return result


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    root = parser()
    if not args or args[0] in ("-h", "--help"):
        root.print_help()
        return
    for size in (2, 1):
        key = tuple(args[:size])
        if key in COMMANDS:
            module = import_module(COMMANDS[key][0])
            previous = sys.argv
            try:
                sys.argv = ["longmorph " + " ".join(key), *args[size:]]
                return module.main()
            finally:
                sys.argv = previous
    group = args[0]
    matches = [key for key in COMMANDS if key[0] == group]
    if matches and args[1:] == ["--help"]:
        print(group + " commands:\n" + "\n".join("  " + " ".join(key) for key in matches))
        return
    root.error("Unknown command: " + " ".join(args[:2]))
