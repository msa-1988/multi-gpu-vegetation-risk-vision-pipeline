from __future__ import annotations

import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def main() -> None:
    roots = sorted(Path("artifacts/runs").glob("*/metrics.json"))
    if not roots:
        print("No metrics files found under artifacts/runs/*/metrics.json")
        return

    print("| run | world_size | last val_iou | last val_f1 | images/sec |")
    print("|---|---:|---:|---:|---:|")
    for metrics_file in roots:
        payload = json.loads(metrics_file.read_text(encoding="utf-8"))
        last = payload["history"][-1]
        print(
            f"| {metrics_file.parent.name} | {payload['world_size']} | "
            f"{last['val_iou']:.4f} | {last['val_f1']:.4f} | {last['images_per_sec']:.1f} |"
        )


if __name__ == "__main__":
    main()

