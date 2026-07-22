"""Hash images and report exact duplicates across dataset splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/manifests/duplicate_audit.json"))
    args = parser.parse_args()
    groups: dict[str, list[str]] = defaultdict(list)
    for root in args.paths:
        if not root.exists():
            raise FileNotFoundError(f"Input path does not exist: {root}")
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            groups[sha256(path)].append(path.as_posix())
    duplicates = {digest: paths for digest, paths in groups.items() if len(paths) > 1}
    report = {"files": sum(len(paths) for paths in groups.values()), "duplicate_groups": duplicates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"files": report["files"], "duplicate_groups": len(duplicates)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

