#!/usr/bin/env python3
"""Retrieve pinned, source-only optional baseline dependencies.

No network access or filesystem mutation occurs with --list or --dry-run.
Run explicitly with --fetch to download the selected upstream commit(s).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "pinned_sources.json"
DEFAULT_DESTINATION = HERE.parent / "longmorph" / "baselines" / "vendor"
BLOCKED_DIRECTORIES = {
    ".git",
    ".github",
    ".google_colab",
    "__pycache__",
    "assets",
    "cache",
    "checkpoints",
    "data",
    "dataset",
    "datasets",
    "figures",
    "images",
    "logs",
    "notebooks",
    "output",
    "outputs",
    "precomputed_outputs",
    "pretrained",
    "results",
    "samples",
    "tests",
    "wandb",
    "weights",
}
SOURCE_SUFFIXES = {".py", ".pyi", ".sh", ".toml", ".yaml", ".yml", ".cfg", ".ini"}


def selected_path(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe upstream source path")
    if any(part.lower() in BLOCKED_DIRECTORIES for part in path.parts[:-1]):
        return False
    lower = path.name.lower()
    is_notice = lower.startswith(("license", "licence", "copying", "notice"))
    is_requirements = lower.startswith("requirements") and path.suffix == ".txt"
    return is_notice or is_requirements or path.suffix.lower() in SOURCE_SUFFIXES


def git(repository: Path, *arguments: str) -> bytes:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if completed.returncode:
        # Do not echo environment-specific Git configuration or credential helpers.
        raise RuntimeError(
            f"Git operation {arguments[0]!r} failed with exit code {completed.returncode}"
        )
    return completed.stdout


def tree(repository: Path, revision: str) -> list[tuple[str, str, str]]:
    entries = []
    for record in git(repository, "ls-tree", "-rz", "--full-tree", revision).split(b"\0"):
        if not record:
            continue
        metadata, encoded_name = record.split(b"\t", 1)
        mode, kind, object_id = metadata.decode("ascii").split()
        name = encoded_name.decode("utf-8")
        if kind == "blob" and mode in ("100644", "100755") and selected_path(name):
            entries.append((name, mode, object_id))
    return entries


def export_sources(repository: Path, item: dict, destination: Path, dry_run: bool) -> None:
    revision = git(repository, "rev-parse", f"{item['commit']}^{{commit}}").decode().strip()
    if revision != item["commit"]:
        raise RuntimeError("Upstream commit identity mismatch")
    entries = tree(repository, revision)
    names = {name for name, _, _ in entries}
    missing = set(item["required_source_paths"]) - names
    if missing:
        raise RuntimeError(f"Required upstream source files missing: {sorted(missing)}")
    license_bytes = git(repository, "show", f"{revision}:{item['license_path']}")
    if hashlib.sha256(license_bytes).hexdigest() != item["license_sha256"]:
        raise RuntimeError("Pinned upstream license identity mismatch")
    target = destination / item["name"]
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to replace existing dependency directory: {target}")
    if dry_run:
        print(f"{item['name']}: {revision}, {len(entries)} source/metadata files, dry run")
        return
    target.mkdir(parents=True, exist_ok=False)
    exported = []
    # An interrupted export remains visible for inspection. No automatic deletion,
    # reset, overwrite, package installation, or training is performed.
    for name, mode, object_id in entries:
        payload = git(repository, "cat-file", "blob", object_id)
        path = target.joinpath(*PurePosixPath(name).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)
        if mode == "100755":
            path.chmod(0o755)
        exported.append(
            dict(path=name, git_blob=object_id, sha256=hashlib.sha256(payload).hexdigest())
        )
    receipt = dict(
        schema="LONGMORPH_OPTIONAL_UPSTREAM_SOURCE_V1",
        name=item["name"],
        repository=item["repository"],
        commit=revision,
        license=item["license"],
        export_scope="filtered source, dependency metadata, and license notices",
        excluded="datasets, model weights, checkpoints, saved experiment outputs, and Git history",
        files=exported,
    )
    with (target / "UPSTREAM_SOURCE.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(f"{item['name']}: exported {len(exported)} pinned files to {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models", nargs="*", help="Dependency names; 'all' selects the five licensed dependencies"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--list", action="store_true", help="List pins without writing or networking"
    )
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate a supplied local checkout without writing or networking",
    )
    action.add_argument(
        "--fetch", action="store_true", help="Explicitly retrieve and export pinned source"
    )
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--from-local",
        type=Path,
        help="Parent of existing named checkouts; reads Git objects without modifying them",
    )
    args = parser.parse_args()
    items = json.loads(MANIFEST.read_text())["dependencies"]
    available = {item["name"]: item for item in items}
    names = args.models or ["all"]
    if "all" in names:
        if names != ["all"]:
            parser.error("Use 'all' alone, or supply individual dependency names")
        names = [item["name"] for item in items if item["automated_source_retrieval"]]
    unknown = set(names) - available.keys()
    if unknown:
        parser.error(f"Unknown dependency names: {sorted(unknown)}")
    if len(names) != len(set(names)):
        parser.error("Duplicate dependency name")
    if args.list:
        for name in names:
            item = available[name]
            status = (
                "available"
                if item["automated_source_retrieval"]
                else "manual licensing review required"
            )
            print(f"{name}\t{item['commit']}\t{item['license']}\t{status}")
        return
    if args.dry_run and args.from_local is None:
        parser.error("--dry-run requires --from-local; it never accesses the network")
    for name in names:
        item = available[name]
        if not item["automated_source_retrieval"]:
            parser.error(
                f"{name}: no license file was found at the pin; resolve permission separately"
            )
        if not re.fullmatch(
            r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git", item["repository"]
        ):
            raise ValueError("Only fixed public GitHub HTTPS repository URLs are supported")
        if not re.fullmatch(r"[0-9a-f]{40}", item["commit"]):
            raise ValueError("A full immutable Git commit is required")
        if args.from_local is not None:
            export_sources(args.from_local / name, item, args.destination.resolve(), args.dry_run)
        else:
            with tempfile.TemporaryDirectory(prefix=f"longmorph-upstream-{name}-") as temporary:
                repository = Path(temporary)
                git(repository, "init", "--bare", "--quiet")
                git(
                    repository,
                    "fetch",
                    "--quiet",
                    "--depth=1",
                    "--no-tags",
                    item["repository"],
                    item["commit"],
                )
                export_sources(repository, item, args.destination.resolve(), False)


if __name__ == "__main__":
    main()
