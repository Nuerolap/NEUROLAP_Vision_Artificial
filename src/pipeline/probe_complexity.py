import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .common import artifacts_v2_dir, ensure_dir, utc_now_iso, videos_base, write_json


def run_extraction(args: argparse.Namespace, complexity: int, out_csv: Path, keypoints_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "Videos.pipeline.extract_features",
        "--base",
        str(Path(args.base).resolve()),
        "--index",
        str(Path(args.index).resolve()),
        "--out",
        str(out_csv),
        "--keypoints-dir",
        str(keypoints_dir),
        "--sample-per-label",
        str(args.sample_per_label),
        "--model-complexity",
        str(complexity),
        "--target-fps",
        str(args.target_fps),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--max-sampled-frames",
        str(args.max_sampled_frames),
        "--overwrite-output",
    ]
    if args.include_normalized_stems:
        cmd.append("--include-normalized-stems")
        cmd.extend(args.include_normalized_stems)
    subprocess.run(cmd, check=True)


def summarize(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    ok = df[df["extraction_status"].astype(str).str.lower() == "ok"].copy()
    if ok.empty:
        return {
            "rows": int(len(df)),
            "ok_rows": 0,
            "mean_max_miss": None,
            "mean_any_hand_coverage": None,
            "mean_processing_seconds": None,
        }
    return {
        "rows": int(len(df)),
        "ok_rows": int(len(ok)),
        "mean_max_miss": float(ok["max_miss"].mean()),
        "mean_any_hand_coverage": float(ok["any_hand_coverage"].mean()),
        "mean_processing_seconds": float(ok["processing_seconds"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe MediaPipe model_complexity=0 vs 1 on a small v2 sample.")
    parser.add_argument("--base", default=str(videos_base()), help="Videos directory.")
    parser.add_argument("--index", default=None, help="video_index.csv path.")
    parser.add_argument("--out", default=None, help="Probe summary JSON path.")
    parser.add_argument("--sample-per-label", type=int, default=1)
    parser.add_argument("--include-normalized-stems", nargs="*", default=["362"], help="Extra exact normalized_stem values to include.")
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--max-sampled-frames", type=int, default=180)
    parser.add_argument("--miss-tolerance", type=float, default=5.0, help="Allowed max-miss percentage-point increase for complexity 0.")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(artifacts_v2_dir(base) / "probe_complexity")
    args.index = args.index or str(artifacts_v2_dir(base) / "video_index.csv")
    summary_out = Path(args.out) if args.out else artifacts_v2_dir(base) / "probe_complexity.json"

    outputs = {
        0: out_dir / "features_complexity0.csv",
        1: out_dir / "features_complexity1.csv",
    }
    for complexity, out_csv in outputs.items():
        run_extraction(args, complexity, out_csv, out_dir / f"keypoints_c{complexity}")

    summary = {
        "generated_at": utc_now_iso(),
        "sample_per_label": int(args.sample_per_label),
        "include_normalized_stems": list(args.include_normalized_stems),
        "max_sampled_frames": int(args.max_sampled_frames),
        "miss_tolerance": float(args.miss_tolerance),
        "complexity": {str(k): summarize(v) for k, v in outputs.items()},
        "recommended_complexity": 1,
    }
    c0 = summary["complexity"]["0"]
    c1 = summary["complexity"]["1"]
    if (
        c0["ok_rows"] == c1["ok_rows"]
        and c0["mean_max_miss"] is not None
        and c1["mean_max_miss"] is not None
        and c0["mean_max_miss"] <= c1["mean_max_miss"] + args.miss_tolerance
    ):
        summary["recommended_complexity"] = 0

    write_json(summary_out, summary)
    print("=== COMPLEXITY PROBE ===")
    print(summary)
    print(f"[OK] summary -> {summary_out}")


if __name__ == "__main__":
    main()
