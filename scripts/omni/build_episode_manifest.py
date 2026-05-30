#!/usr/bin/env python3
"""Build a lightweight manifest for local Xperience-10M episode folders.

The manifest is intentionally metadata-only. It lets us decide how many
episodes fit on the H20 server before downloading or copying large media.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VIDEO_NAMES = [
    "fisheye_cam0.mp4",
    "fisheye_cam1.mp4",
    "fisheye_cam2.mp4",
    "fisheye_cam3.mp4",
    "stereo_left.mp4",
    "stereo_right.mp4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Xperience-10M episode folders.")
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        required=True,
        help="Root to scan. May be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/omni_exploration/episode_manifest.json"),
    )
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means no cap.")
    return parser.parse_args()


def size_or_zero(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def inspect_episode(annotation: Path) -> dict:
    episode_dir = annotation.parent
    files = [{"name": "annotation.hdf5", "bytes": size_or_zero(annotation), "exists": annotation.exists()}]
    for name in VIDEO_NAMES:
        path = episode_dir / name
        files.append({"name": name, "bytes": size_or_zero(path), "exists": path.exists()})
    rrd = episode_dir / "visualization.rrd"
    files.append({"name": "visualization.rrd", "bytes": size_or_zero(rrd), "exists": rrd.exists()})
    total_bytes = sum(item["bytes"] for item in files)
    train_bytes = sum(item["bytes"] for item in files if item["name"] != "visualization.rrd")
    return {
        "episode_id": episode_dir.name,
        "path": str(episode_dir),
        "annotation": str(annotation),
        "files": files,
        "total_bytes": total_bytes,
        "train_minimal_bytes": train_bytes,
        "has_annotation": annotation.exists(),
        "has_any_video": any((episode_dir / name).exists() for name in VIDEO_NAMES),
        "has_all_videos": all((episode_dir / name).exists() for name in VIDEO_NAMES),
        "has_rrd": rrd.exists(),
    }


def main() -> int:
    args = parse_args()
    annotations: list[Path] = []
    for root in args.data_root:
        annotations.extend(sorted(root.expanduser().resolve().rglob("annotation.hdf5")))
    if args.max_episodes > 0:
        annotations = annotations[: args.max_episodes]

    episodes = [inspect_episode(path) for path in annotations]
    summary = {
        "num_episodes": len(episodes),
        "total_bytes": sum(ep["total_bytes"] for ep in episodes),
        "train_minimal_bytes": sum(ep["train_minimal_bytes"] for ep in episodes),
        "notes": [
            "train_minimal_bytes excludes visualization.rrd because model training does not need it.",
            "This file is metadata-only; it does not copy or download raw data.",
        ],
    }
    payload = {"summary": summary, "episodes": episodes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
