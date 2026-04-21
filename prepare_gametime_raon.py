"""Convert Game-Time Moshi-format alignments to Raon-Speech duplex JSONL.

Source (per line in a Moshi manifest):
    {"path": "<stereo.wav>", "duration": <sec>, "scenario": "...", "dataset": "..."}

with a sibling alignment JSONL at:
    <exps>/<dataset>/alignment/<scenario>/<split>/<stem>.jsonl
whose two lines (one per speaker) contain Moshi alignments:
    {"alignments": [[word, [start_sec, end_sec], "SPEAKER_AGENT"|"SPEAKER_USER"], ...]}

Target (per line, consumed by raon.utils.duplex_data.timeline_turns_to_metadata via
the "scripts" branch):
    {
      "audio_path": "<stereo.wav>",
      "sample_rate": 24000,
      "language": "eng",
      "channel": "full_duplex",
      "speak_first": [<ch0_first>, <ch1_first>],
      "include_in_training": [1, 1],
      "scripts": [[{"word": w, "start": s, "end": e}, ...],   # ch0 = AGENT
                  [{"word": w, "start": s, "end": e}, ...]]   # ch1 = USER
    }
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import soundfile as sf

logger = logging.getLogger("prepare_gametime_raon")

AGENT_CHANNEL = 0
USER_CHANNEL = 1


def remap_path(path: str, prefix_map: list[tuple[str, str]]) -> str:
    for src, dst in prefix_map:
        if path.startswith(src):
            return dst + path[len(src):]
    return path


def alignment_path_for(wav_path: Path) -> Path:
    parts = list(wav_path.parts)
    try:
        idx = parts.index("dialogue")
    except ValueError as e:
        raise ValueError(f"wav path missing 'dialogue' segment: {wav_path}") from e
    parts[idx] = "alignment"
    return Path(*parts).with_suffix(".jsonl")


def load_alignment(jsonl_path: Path) -> tuple[list[dict], list[dict]]:
    """Return (ch0_words, ch1_words) where ch0=AGENT, ch1=USER, in seconds."""
    ch0: list[dict] = []
    ch1: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for word, (start, end), speaker in obj["alignments"]:
                item = {"word": word, "start": float(start), "end": float(end)}
                if speaker == "SPEAKER_AGENT":
                    ch0.append(item)
                elif speaker == "SPEAKER_USER":
                    ch1.append(item)
                else:
                    raise ValueError(f"Unknown speaker {speaker!r} in {jsonl_path}")
    ch0.sort(key=lambda w: w["start"])
    ch1.sort(key=lambda w: w["start"])
    return ch0, ch1


def infer_speak_first(ch0: list[dict], ch1: list[dict]) -> list[int]:
    """[ch0_first, ch1_first] — exactly one is 1 unless both empty."""
    ch0_start = ch0[0]["start"] if ch0 else float("inf")
    ch1_start = ch1[0]["start"] if ch1 else float("inf")
    if ch0_start == float("inf") and ch1_start == float("inf"):
        return [1, 0]
    if ch0_start <= ch1_start:
        return [1, 0]
    return [0, 1]


def convert(
    manifest: Path,
    output: Path,
    prefix_map: list[tuple[str, str]],
    language: str,
    channel_type: str,
    require_stereo: bool,
    verify_audio: bool,
) -> tuple[int, int]:
    n_ok = n_skip = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open() as fin, output.open("w") as fout:
        for line_no, raw in enumerate(fin, 1):
            raw = raw.strip()
            if not raw:
                continue
            src = json.loads(raw)
            wav_path = Path(remap_path(src["path"], prefix_map))

            if not wav_path.is_file():
                logger.warning("skip (wav missing): %s", wav_path)
                n_skip += 1
                continue

            align_path = alignment_path_for(wav_path)
            if not align_path.is_file():
                logger.warning("skip (alignment missing): %s", align_path)
                n_skip += 1
                continue

            sample_rate = 24000
            duration_samples = None
            if verify_audio:
                info = sf.info(str(wav_path))
                if require_stereo and info.channels != 2:
                    logger.warning("skip (not stereo, channels=%d): %s", info.channels, wav_path)
                    n_skip += 1
                    continue
                sample_rate = int(info.samplerate)
                duration_samples = int(info.frames)

            try:
                ch0, ch1 = load_alignment(align_path)
            except Exception as exc:
                logger.warning("skip (alignment parse error %s): %s", exc, align_path)
                n_skip += 1
                continue

            if not ch0 and not ch1:
                logger.warning("skip (empty alignment): %s", align_path)
                n_skip += 1
                continue

            record = {
                "audio_path": str(wav_path),
                "sample_rate": sample_rate,
                "language": language,
                "channel": channel_type,
                "speak_first": infer_speak_first(ch0, ch1),
                "include_in_training": [1, 1],
                "scripts": [ch0, ch1],
            }
            if duration_samples is not None:
                record["duration_samples"] = duration_samples
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_ok += 1

    return n_ok, n_skip


def parse_prefix_map(pairs: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--path_prefix_map expects SRC=DST, got {pair!r}")
        src, dst = pair.split("=", 1)
        out.append((src, dst))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Input Moshi-format jsonl (e.g. gametime_train_all_basic.jsonl)")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output Raon duplex jsonl")
    ap.add_argument(
        "--path_prefix_map",
        action="append",
        default=[],
        help="SRC=DST prefix remap applied to manifest paths. Repeat for multiple rules. "
             "Default remaps /home/dmnph/Game-Time-Sushi/datasets/ to the local exps path.",
    )
    ap.add_argument("--language", default="eng", choices=["eng", "kor"])
    ap.add_argument("--channel_type", default="full_duplex",
                    choices=["full_duplex", "duplex_instruct"])
    ap.add_argument("--no_require_stereo", dest="require_stereo",
                    action="store_false", default=True,
                    help="Accept non-stereo wavs (default: skip them).")
    ap.add_argument("--no_verify_audio", dest="verify_audio",
                    action="store_false", default=True,
                    help="Skip soundfile.info probe (faster, no sample_rate/duration_samples).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    prefix_map = parse_prefix_map(args.path_prefix_map)
    if not prefix_map:
        prefix_map = [(
            "/home/dmnph/Game-Time-Sushi/datasets/",
            "/home/enpei/EnPei/Game-Time-Sushi/gametime/00_single_agent/exps/",
        )]

    n_ok, n_skip = convert(
        manifest=args.manifest,
        output=args.output,
        prefix_map=prefix_map,
        language=args.language,
        channel_type=args.channel_type,
        require_stereo=args.require_stereo,
        verify_audio=args.verify_audio,
    )
    logger.info("wrote %d records, skipped %d, -> %s", n_ok, n_skip, args.output)


if __name__ == "__main__":
    main()
