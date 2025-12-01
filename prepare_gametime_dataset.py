#!/usr/bin/env python3
"""
Utility to convert the Game-Time datasets into the format expected by
moshi-finetune.

It performs two actions:
1. Builds Moshi-style alignment JSON files (same stem as each stereo dialogue
   .wav) by merging the provided alignment JSONL files and renaming
   SPEAKER_AGENT -> SPEAKER_MAIN.
2. Writes train/eval manifest JSONLs that list every stereo dialogue file and
   its duration in seconds.

Usage:
    python prepare_gametime_dataset.py \
        --datasets-root /home/enpei/EnPei/Game-Time-Sushi/datasets \
        --output-dir /home/enpei/EnPei/Game-Time-Sushi/moshi-finetune/data
"""

from __future__ import annotations

import argparse
import contextlib
import json
import wave
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_datasets = repo_root / "datasets"
    default_output = Path(__file__).resolve().parent / "data"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=default_datasets,
        help="Root directory that contains the Game-Time-* dataset folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where gametime_train.jsonl and gametime_test.jsonl will be written.",
    )
    parser.add_argument(
        "--score-root",
        type=Path,
        default=None,
        help=(
            "Optional root that mirrors the dialogue/ tree and contains per-utterance "
            "JSON files with a `scores` object. If provided, examples missing scores or "
            "failing thresholds will be skipped."
        ),
    )
    parser.add_argument(
        "--min-score-per-dim",
        type=float,
        default=None,
        help="If set, require every score dimension to be at least this value.",
    )
    parser.add_argument(
        "--min-avg-score",
        type=float,
        default=None,
        help="If set, require the average of all score dimensions to be at least this value.",
    )
    parser.add_argument(
        "--top-per-id",
        action="store_true",
        help=(
            "If set, keep only one example per data id (stem before `_sample_`), "
            "selecting the highest scoring candidate. Requires --score-root."
        ),
    )
    return parser.parse_args()


def determine_split(dataset_dir: Path) -> tuple[str, str]:
    """Return (manifest_split, inner_split_name) based on the folder name."""
    name = dataset_dir.name.lower()
    if name.endswith("train"):
        return "train", "train"
    if name.endswith("test"):
        return "test", "test"
    raise ValueError(f"Cannot infer split from directory name: {dataset_dir}")


def load_alignment_entries(jsonl_path: Path) -> list[list]:
    """Load and normalize the alignments contained in a .jsonl file."""
    entries: list[list] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            for token, ts, speaker in payload["alignments"]:
                label = speaker
                if speaker == "SPEAKER_AGENT":
                    label = "SPEAKER_MAIN"
                entries.append([token, ts, label])
    # Keep chronological order to match audio frames.
    entries.sort(key=lambda item: (item[1][0], item[1][1]))
    return entries


def write_alignment_json(json_path: Path, alignments: list[list]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump({"alignments": alignments}, f, ensure_ascii=False)


def get_duration(audio_path: Path) -> float:
    with contextlib.closing(wave.open(str(audio_path), "rb")) as wf:
        return wf.getnframes() / float(wf.getframerate())


def iter_jsonl_files(alignment_root: Path, split_name: str) -> Iterable[Path]:
    for scenario_dir in sorted(alignment_root.iterdir()):
        candidate = scenario_dir / split_name
        if not candidate.is_dir():
            continue
        yield from sorted(candidate.glob("*.jsonl"))

def iter_wav_files(dialogue_root: Path, split_name: str) -> Iterable[Path]:
    for scenario_dir in sorted(dialogue_root.iterdir()):
        candidate = scenario_dir / split_name
        if not candidate.is_dir():
            continue
        yield from sorted(candidate.glob("*.wav"))


def find_alignment_root(dataset_dir: Path) -> Path:
    for candidate in ("alignment", "alignments", "alignments_whisper"):
        path = dataset_dir / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"{dataset_dir} must contain 'alignment/' or 'alignments_whisper/' folders."
    )


def load_scores(score_path: Path) -> dict | None:
    if not score_path.is_file():
        return None
    with score_path.open() as f:
        payload = json.load(f)
    return payload.get("scores")


def passes_score_filters(
    scores: dict | None, min_score_per_dim: float | None, min_avg_score: float | None
) -> bool:
    if scores is None:
        return False
    values = list(scores.values())
    if not values:
        return False
    if min_score_per_dim is not None and any(v < min_score_per_dim for v in values):
        return False
    if min_avg_score is not None:
        avg = sum(values) / len(values)
        if avg < min_avg_score:
            return False
    return True


def data_id_from_stem(stem: str) -> str:
    """Group variants like foo_sample_0, foo_sample_1 under the same id."""
    token = stem.split("_sample_")[0]
    return token


def score_priority(scores: dict) -> tuple[float, float]:
    """Return a tuple used to choose the best sample per id.

    Instruction following is up-weighted (1.5x) relative to other scores.
    """
    values = [float(v) for v in scores.values() if v is not None]
    if not values:
        return (-1.0, -1.0)
    instr_raw = scores.get("instruction_following")
    instr = float(instr_raw) if instr_raw is not None else 0.0
    other_vals = [float(v) for k, v in scores.items() if k != "instruction_following" and v is not None]
    weighted = 1.5 * instr + sum(other_vals)
    avg = sum(values) / len(values)
    return (weighted, avg)


def convert_dataset(
    dataset_dir: Path,
    split_name: str,
    score_root: Path | None,
    min_score_per_dim: float | None,
    min_avg_score: float | None,
    top_per_id: bool,
) -> list[dict]:
    """Convert every alignment file under dataset_dir into Moshi format."""
    print(f"Processing {dataset_dir} ({split_name}) ...")
    dialogue_root = dataset_dir / "dialogue"
    if not dialogue_root.is_dir():
        raise FileNotFoundError(f"{dataset_dir} must contain a 'dialogue/' folder.")
    alignment_root = find_alignment_root(dataset_dir)

    records: list[dict] = []
    kept = 0
    skipped = 0

    best_by_id: dict[str, tuple[float, float, Path, dict | None]] = {}

    for audio_path in iter_wav_files(dialogue_root, split_name):
        rel = audio_path.relative_to(dialogue_root)
        alignment_path = (
            alignment_root / rel.parent / f"{audio_path.stem}.jsonl"
        )

        # Optional score filtering
        if score_root is not None:
            score_path = score_root / rel.parent / f"{audio_path.stem}.json"
            scores = load_scores(score_path)
            if not passes_score_filters(scores, min_score_per_dim, min_avg_score):
                skipped += 1
                continue
        else:
            scores = None

        if top_per_id and score_root is not None:
            data_id = data_id_from_stem(audio_path.stem)
            primary, secondary = score_priority(scores or {})
            current = best_by_id.get(data_id)
            if current is None or (primary, secondary) > (current[0], current[1]):
                best_by_id[data_id] = (primary, secondary, audio_path, scores)
            continue

        # copy alignment to dialogue folder with Moshi format
        alignments = load_alignment_entries(alignment_path)
        output_alignment_path = audio_path.with_suffix(".json")
        write_alignment_json(output_alignment_path, alignments)

        scenario = rel.parts[0]
        filename = audio_path.stem

        duration = get_duration(audio_path)
        records.append(
            {
                "path": str(audio_path.resolve()),
                "duration": duration,
                "scenario": scenario,
                "dataset": dataset_dir.name,
                "scores": scores,
            }
        )
        kept += 1

    if top_per_id and score_root is not None:
        # Materialize the best candidates and process them now.
        kept = 0
        for _, _, audio_path, scores in best_by_id.values():
            rel = audio_path.relative_to(dialogue_root)
            alignment_path = alignment_root / rel.parent / f"{audio_path.stem}.jsonl"

            alignments = load_alignment_entries(alignment_path)
            output_alignment_path = audio_path.with_suffix(".json")
            write_alignment_json(output_alignment_path, alignments)

            scenario = rel.parts[0]
            duration = get_duration(audio_path)
            records.append(
                {
                    "path": str(audio_path.resolve()),
                    "duration": duration,
                    "scenario": scenario,
                    "dataset": dataset_dir.name,
                    "scores": scores,
                }
            )
            kept += 1

    if score_root is not None:
        print(
            f"  Score filtering: kept {kept}, skipped {skipped} "
            f"(root={score_root}, min_per_dim={min_score_per_dim}, min_avg={min_avg_score}, top_per_id={top_per_id})"
        )
    return records


def main():
    args = parse_args()
    args.datasets_root = args.datasets_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.score_root is not None:
        args.score_root = args.score_root.resolve()
    if not args.datasets_root.is_dir():
        raise FileNotFoundError(f"datasets_root {args.datasets_root} does not exist.")
    if args.top_per_id and args.score_root is None:
        raise ValueError("--top-per-id requires --score-root (scores are needed to rank).")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, list[dict]] = defaultdict(list)

    # Allow pointing directly at a dataset folder or at a root containing multiple datasets.
    candidate_dirs: list[Path] = []
    if (args.datasets_root / "dialogue").is_dir():
        candidate_dirs.append(args.datasets_root)
    else:
        candidate_dirs.extend([p for p in args.datasets_root.iterdir() if p.is_dir()])

    if not candidate_dirs:
        raise FileNotFoundError(
            f"No dataset folders found under {args.datasets_root} (expected 'dialogue/' subfolder)."
        )

    for dataset_dir in sorted(candidate_dirs):
        try:
            manifest_split, split_name = determine_split(dataset_dir)
        except ValueError:
            continue
        records = convert_dataset(
            dataset_dir,
            split_name,
            score_root=args.score_root,
            min_score_per_dim=args.min_score_per_dim,
            min_avg_score=args.min_avg_score,
            top_per_id=args.top_per_id,
        )
        manifests[manifest_split].extend(records)

    for split_name, records in manifests.items():
        records.sort(key=lambda item: item["path"])
        output_path = args.output_dir / f"gametime_{split_name}.jsonl"
        with output_path.open("w") as f:
            for record in records:
                json.dump(record, f)
                f.write("\n")
        total_hours = sum(r["duration"] for r in records) / 3600.0
        print(
            f"Wrote {len(records)} entries to {output_path} "
            f"({total_hours:.2f} hours of audio)."
        )


if __name__ == "__main__":
    main()
