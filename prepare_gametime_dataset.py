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


def convert_dataset(dataset_dir: Path, split_name: str) -> list[dict]:
    """Convert every alignment file under dataset_dir into Moshi format."""
    print(f"Processing {dataset_dir} ({split_name}) ...")
    dialogue_root = dataset_dir / "dialogue"
    if not dialogue_root.is_dir():
        raise FileNotFoundError(f"{dataset_dir} must contain a 'dialogue/' folder.")
    alignment_root = find_alignment_root(dataset_dir)

    records: list[dict] = []
    for audio_path in iter_wav_files(dialogue_root, split_name):
        rel = audio_path.relative_to(dialogue_root)
        alignment_path = (
            alignment_root / rel.parent / f"{audio_path.stem}.jsonl"
        )

        # copy alignment to dialogue folder with Moshi format
        alignments = load_alignment_entries(alignment_path)
        output_alignment_path = audio_path.with_suffix(".json")
        write_alignment_json(output_alignment_path, alignments)

        scenario = rel.parts[0]
        filename = audio_path.stem

        duration = get_duration(audio_path)
        records.append(
            {
                "path": str(audio_path),
                "duration": duration,
                "scenario": scenario,
                "dataset": dataset_dir.name,
            }
        )
    return records


def main():
    args = parse_args()
    if not args.datasets_root.is_dir():
        raise FileNotFoundError(f"datasets_root {args.datasets_root} does not exist.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, list[dict]] = defaultdict(list)

    for dataset_dir in sorted(args.datasets_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        try:
            manifest_split, split_name = determine_split(dataset_dir)
        except ValueError:
            continue
        if split_name not in ("train"):
            continue
        records = convert_dataset(dataset_dir, split_name)
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
