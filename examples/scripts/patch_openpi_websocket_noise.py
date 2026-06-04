#!/usr/bin/env python3
"""Patch piperx-openpi's websocket server for DSRL noise steering.

piperx-openpi passes the unpacked websocket payload straight to
``policy.infer(obs)``.  DSRL sends a flat obs dict with an extra ``noise``
key; this script pops ``noise`` and forwards it as ``policy.infer(obs, noise=...)``.

Usage (on the rollout box, with the policy server repo checked out):

    python3 examples/scripts/patch_openpi_websocket_noise.py
    python3 examples/scripts/patch_openpi_websocket_noise.py ~/openpi
    OPENPI_ROOT=~/piperx-openpi python3 examples/scripts/patch_openpi_websocket_noise.py

Restart ``serve_policy.py`` after patching.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

TARGET = Path("src/openpi/serving/websocket_policy_server.py")

OLD_BLOCK = """                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(obs)"""

NEW_BLOCK = """                obs = msgpack_numpy.unpackb(await websocket.recv())
                noise = obs.pop("noise", None) if isinstance(obs, dict) else None

                infer_time = time.monotonic()
                action = self._policy.infer(obs, noise=noise)"""

ALREADY_PATCHED = 'noise = obs.pop("noise", None)'


def resolve_openpi_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    for candidate in (
        os.environ.get("OPENPI_ROOT"),
        "~/openpi",
        "~/piperx-openpi",
    ):
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / TARGET).is_file():
            return path
    raise FileNotFoundError(
        "Could not find websocket_policy_server.py. Pass the repo path explicitly, "
        "e.g. python3 examples/scripts/patch_openpi_websocket_noise.py ~/openpi"
    )


def patch_file(path: Path, dry_run: bool) -> None:
    text = path.read_text(encoding="utf-8")
    if ALREADY_PATCHED in text:
        print(f"Already patched: {path}")
        return
    if OLD_BLOCK not in text:
        raise RuntimeError(
            f"Unexpected contents in {path} — manual edit required.\n"
            "Insert before infer():\n"
            '    noise = obs.pop("noise", None) if isinstance(obs, dict) else None\n'
            "And call:\n"
            "    action = self._policy.infer(obs, noise=noise)"
        )
    updated = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    if dry_run:
        print(f"Would patch: {path}")
        return
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8")
    print(f"Patched: {path}")
    print(f"Backup:  {backup}")
    print("Restart the policy server (serve_policy.py) for the change to take effect.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "openpi_root",
        nargs="?",
        default=None,
        help="Path to piperx-openpi checkout (default: OPENPI_ROOT, ~/openpi, ~/piperx-openpi)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would change")
    args = parser.parse_args()

    root = resolve_openpi_root(args.openpi_root)
    target = root / TARGET
    if not target.is_file():
        print(f"Missing file: {target}", file=sys.stderr)
        return 1
    patch_file(target, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
