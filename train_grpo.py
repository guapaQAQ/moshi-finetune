import argparse
import dataclasses
import json
import logging
import os
import pprint
import random
import shutil
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator

import fire
import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
import torchaudio
from torch.optim import AdamW, lr_scheduler

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.data.interleaver import InterleavedTokenizer, Interleaver, Sample, Batch
from finetune.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)
from finetune.mixed_precision import (
    downcast_mixed_precision,
    prepare_mixed_precision,
    upcast_mixed_precision,
)
from finetune.monitoring.metrics_logger import MetricsLogger
from finetune.monitoring.utils import set_logger
from finetune.utils import TrainState, logged_closing, set_random_seed
from finetune.wrapped_model import get_fsdp_model
from moshi.models import loaders

logger = logging.getLogger("train_grpo")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for Moshi.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--reward_manifest", type=str, required=True)
    parser.add_argument("--group_size", type=int, default=4)
    # Two flags instead of argparse.BooleanOptionalAction, which needs Python
    # 3.9+ and would blow up at parse time on a stray 3.8 interpreter.
    parser.add_argument(
        "--normalize_advantage",
        dest="normalize_advantage",
        action="store_true",
        default=True,
        help="Divide group advantages by their std (default on).",
    )
    parser.add_argument(
        "--no_normalize_advantage",
        dest="normalize_advantage",
        action="store_false",
        help="Disable advantage normalization (Dr. GRPO-style) for ablations.",
    )
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument(
        "--clip_eps",
        type=float,
        default=0.2,
        help="PPO/GRPO clip range ε for the importance-ratio surrogate. The "
        "behavior-policy logprob is cached at each refresh; off-policy reuse "
        "(refresh_every>1) is then correctly clipped. 0 = plain REINFORCE (no "
        "ratio). At refresh_every=1 the ratio is ~1 so this is a no-op.",
    )
    parser.add_argument(
        "--clip_level",
        type=str,
        default="sequence",
        choices=["sequence", "token"],
        help="Granularity of the PPO ratio/clip. 'sequence' (default): one ratio "
        "per sample from the aggregated logprob. 'token': per-token ratio+clip "
        "then mask-averaged (canonical ms-swift/TRL GRPO; better credit "
        "assignment + length handling for Moshi's long audio).",
    )
    parser.add_argument(
        "--logp_pool",
        type=str,
        default="split",
        choices=["split", "token"],
        help="Per-sample logprob aggregation. 'split': mean(text)+mean(audio) "
        "(equal text/audio weight). 'token': pool all completion tokens, one "
        "mean (canonical GRPO; audio-dominant by token count).",
    )
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--min_group_size", type=int, default=2)
    parser.add_argument("--sample_with_replacement", action="store_true", default=True)
    parser.add_argument(
        "--select_by",
        type=str,
        default="uniform",
        choices=["uniform", "advantage"],
        help="How to pick which group to train on each step. 'uniform': random. "
        "'advantage': sample groups with probability ∝ within-group reward std, "
        "so zero-variance groups (no gradient) are skipped and high-signal "
        "groups are prioritized. Doesn't install missing behavior (floored "
        "tasks still need SFT) and can shift the train distribution off the "
        "eval mix -- use with that in mind.",
    )
    parser.add_argument(
        "--kl_coef",
        type=float,
        default=0.0,
        help="KL penalty coefficient β against the LoRA-disabled reference policy. "
        "0 reproduces the previous REINFORCE behaviour. DeepSeek default is 0.02.",
    )
    # ---- Online (in-process) GRPO -------------------------------------------
    # When --online is set, the resident model GENERATES its own rollouts every
    # --refresh_every steps (no per-iter model reload), scores them via the
    # whisper + judge servers, and trains on them. --reward_manifest is then the
    # PATH the freshly-generated rewards are written to (re-read each refresh)
    # instead of a static input. refresh_every=1 => fully on-policy.
    parser.add_argument("--online", action="store_true", default=False)
    parser.add_argument("--egs_file", type=str, default=None,
                        help="Prompt egs (jsonl) to page rollouts from (online).")
    parser.add_argument("--prompts_per_iter", type=int, default=8,
                        help="Prompt-groups generated per refresh. Each optimizer "
                        "step consumes one group, so this is how many fresh groups "
                        "a refresh provides.")
    parser.add_argument("--refresh_every", type=int, default=8,
                        help="Regenerate rollouts every N optimizer steps. Set "
                        "== prompts_per_iter to use each fresh group ~once; set "
                        "both to 1 for maximally on-policy (1 group, 1 step).")
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--start_cursor", type=int, default=0,
                        help="Resume paging offset (prompts consumed so far).")
    parser.add_argument("--audio_root", type=str, default=None)
    parser.add_argument("--gen_temp", type=float, default=0.8)
    parser.add_argument("--gen_temp_text", type=float, default=0.7)
    # Inference early-stop: halt generation once the agent has been silent for
    # this many seconds (past --gen_min_gen_sec), skipping the long silent tail.
    # 0 = off (generate the full prompt+tail). Big speedup + shorter rollouts.
    parser.add_argument("--gen_early_stop_sec", type=float, default=0.0)
    parser.add_argument("--gen_min_gen_sec", type=float, default=30.0)
    parser.add_argument("--gen_silence_thresh", type=float, default=1e-3)
    parser.add_argument("--whisper_url", type=str, default="http://127.0.0.1:8003")
    parser.add_argument("--judge_model", type=str, default="gemma-4-31B-it-FP8")
    parser.add_argument("--judge_base_url", type=str, default="http://127.0.0.1:8002/v1")
    parser.add_argument("--judge_api_key", type=str, default="dummy")
    parser.add_argument("--judge_prompt_file", type=str, default=None)
    parser.add_argument("--judge_max_tokens", type=int, default=2048)
    parser.add_argument("--judge_max_workers", type=int, default=16)
    parser.add_argument("--reward_key", type=str, default="applicable_avg")
    parser.add_argument("--repo_root", type=str, default=None,
                        help="Repo root for importing gametime generation + "
                        "reward helpers (online).")
    parser.add_argument("--resume_state", type=str, default=None,
                        help="Path to a train_state.pt (optimizer+scheduler+step+"
                        "rng+cursor) saved next to a checkpoint, for seamless "
                        "resume (no LR re-warmup / momentum reset). Pair with "
                        "--lora_weight = that checkpoint's lora and the same "
                        "max_steps as the original run.")
    return parser.parse_args()


def load_reward_manifest(path: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            group_id = payload.get("group_id")
            if not group_id:
                stem = Path(payload["path"]).stem
                group_id = stem.split("_sample_")[0]
            groups.setdefault(group_id, []).append(payload)
    return groups


def sample_group(
    group: list[dict[str, Any]],
    group_size: int,
    with_replacement: bool,
) -> list[dict[str, Any]]:
    if len(group) >= group_size:
        return random.sample(group, group_size)
    if with_replacement:
        return [random.choice(group) for _ in range(group_size)]
    return group


def _group_reward_std(items: list[dict[str, Any]]) -> float:
    rs = [float(it.get("reward", 0.0)) for it in items]
    if len(rs) < 2:
        return 0.0
    m = sum(rs) / len(rs)
    return (sum((r - m) ** 2 for r in rs) / len(rs)) ** 0.5


def pick_group(group_ids, groups, select_by: str) -> str:
    """Choose which group to train on. 'advantage' weights by within-group reward
    std (zero-variance groups give no gradient, so they're effectively skipped;
    high-signal groups are prioritized)."""
    if select_by == "advantage" and len(group_ids) > 1:
        weights = [_group_reward_std(groups[g]) + 1e-3 for g in group_ids]
        return random.choices(group_ids, weights=weights, k=1)[0]
    return random.choice(group_ids)


def _gametime_imports(repo_root: str):
    """Lazily import the gametime generation + reward helpers (they live outside
    moshi-finetune). Done lazily so non-online training never needs them."""
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from gametime.scripts.rl.collect_moshi_rollouts import (  # noqa: E402
        TARGET_SR,
        generate_dialogue,
        resolve_output_paths,
    )
    from gametime.scripts.rl.online_grpo import align_via_server  # noqa: E402

    return TARGET_SR, generate_dialogue, resolve_output_paths, align_via_server


def online_generate_score(
    cli: argparse.Namespace,
    model: Any,
    lm_gen: Any,
    mimi: Any,
    spm: Any,
    frame_size: int,
    window: list[dict[str, Any]],
    gen_dir: Path,
    group_size: int,
    seed_base: int,
) -> dict[str, list[dict[str, Any]]]:
    """Generate rollouts in-process with the resident policy, score them via the
    whisper + judge servers, and return the GRPO reward groups. No model reload:
    `lm_gen` wraps the model currently being trained."""
    import soundfile as sf

    repo_root = cli.repo_root or str(Path(__file__).resolve().parents[1])
    (TARGET_SR, generate_dialogue, resolve_output_paths,
     align_via_server) = _gametime_imports(repo_root)

    dialogue_dir = gen_dir / "dialogue"
    align_dir = gen_dir / "alignments_whisper"
    judge_dir = gen_dir / "judge_eval"

    # ---- generate (resident model, eval+no_grad inside stream_generate) ------
    model.eval()
    n = 0
    for idx, entry in enumerate(window):
        subcategory, split, base_id = (
            entry["subcategory"], entry.get("split", "train"), entry["id"])
        prompt_path = entry["path"]
        if cli.audio_root:
            marker = "/datasets/"
            pos = prompt_path.find(marker)
            if pos != -1:
                prompt_path = str(Path(cli.audio_root) / prompt_path[pos + len(marker):])
        for k in range(group_size):
            sample_id = f"{base_id}_sample_{k}"
            wav_path, inner_path = resolve_output_paths(
                gen_dir, subcategory, split, sample_id)
            if wav_path.exists():
                continue
            set_random_seed(seed_base + idx * group_size + k)
            stereo, inner_text = generate_dialogue(
                prompt_path, lm_gen, mimi, spm, frame_size, "cuda",
                early_stop_sec=cli.gen_early_stop_sec,
                min_gen_sec=cli.gen_min_gen_sec,
                silence_thresh=cli.gen_silence_thresh)
            sf.write(wav_path, stereo.T, TARGET_SR)
            inner_path.write_text(json.dumps({"agent_a": inner_text}, indent=2))
            n += 1
    model.train()
    main_logger_info(f"[online] generated {n} rollouts -> {dialogue_dir}")

    # ---- score: whisper align -> json -> judge -> reward manifest ------------
    align_via_server(dialogue_dir, align_dir, cli.whisper_url)
    rl = Path(repo_root) / "gametime/scripts/rl"
    judge_prompt = cli.judge_prompt_file or str(
        Path(repo_root) / "gametime/scripts/llm_eval/prompts/qwen3_omni_text.txt")
    subprocess.run([sys.executable, str(rl / "alignments_to_json.py"),
                    "--align_dir", str(align_dir),
                    "--dialogue_dir", str(dialogue_dir)], check=True)
    subprocess.run([sys.executable,
                    str(Path(repo_root) / "gametime/scripts/llm_eval/unified_eval.py"),
                    "--in_dir", str(align_dir), "--out_dir", str(judge_dir),
                    "--model", cli.judge_model, "--provider", "openai",
                    "--base_url", cli.judge_base_url, "--api_keys", cli.judge_api_key,
                    "--system_prompt_file", judge_prompt,
                    "--user_text_prefix",
                    "Please evaluate the following spoken dialogue:\n\n{alignment}",
                    "--constraint_in_system",
                    "--max_tokens", str(cli.judge_max_tokens),
                    "--max_workers", str(cli.judge_max_workers),
                    "--temperature", "0.0"], check=True)
    subprocess.run([sys.executable, str(rl / "build_reward_manifest.py"),
                    "--dialogue_root", str(dialogue_dir),
                    "--score_root", str(judge_dir),
                    "--reward_key", cli.reward_key], check=True)
    return load_reward_manifest(judge_dir / "rewards.jsonl")


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.cpu().numpy()


def build_batch(
    samples: list[dict[str, Any]],
    tokenizer: InterleavedTokenizer,
    target_sr: int,
) -> tuple[Batch, torch.Tensor]:
    batch_samples: list[Sample] = []
    rewards: list[float] = []
    for sample in samples:
        wav_path = Path(sample["path"])
        wav_np = load_audio(wav_path, target_sr)
        item = tokenizer(wav_np, start_sec=0.0, path=str(wav_path))
        batch_samples.append(item)
        rewards.append(float(sample["reward"]))
    return Batch.collate(batch_samples), torch.tensor(rewards, device="cuda")


def per_token_logprob(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (per-token logπ, float mask) — both shaped like `target`."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    target = target.long()
    # `target` (the codes) includes special tokens (zero_token_id / padding) at
    # positions that are masked out downstream, but their values can exceed the
    # logits' vocab dim and trip a CUDA "index out of bounds" in gather. The
    # model's *embedding* is sized to include those tokens, but the logits are
    # not. Clamp into range before gathering; those positions are zeroed by
    # `mask` anyway, so the clamped value is never used. Guard: an OOV target at
    # an UNMASKED position would be silently corrupted by the clamp -> assert it
    # never happens (would mean a real mask bug).
    vocab = log_probs.size(-1)
    oov_valid = ((target < 0) | (target >= vocab)) & mask.bool()
    assert not bool(oov_valid.any()), (
        "out-of-vocab target at an unmasked position — mask/logits mismatch"
    )
    safe_target = target.clamp(0, vocab - 1)
    gathered = torch.gather(log_probs, dim=-1, index=safe_target.unsqueeze(-1)).squeeze(-1)
    # Zero the logπ at masked positions BEFORE any downstream `* mask`. Masked
    # positions (special/forbidden tokens) can carry -inf logprob, and -inf * 0
    # = nan would poison the per-sample logp and the KL term. where() makes them
    # exactly 0 so they drop out cleanly.
    gathered = torch.where(mask.bool(), gathered, torch.zeros_like(gathered))
    return gathered, mask.float()


def seq_logprob(per_tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum per-token logπ within each sample. Shape: [batch].

    Kept for reference / future ablations; the canonical GRPO formulation
    uses per_sample_avg_logp instead -- see docs/grpo_loss_formulation.md
    for why summing tokens (the pre-2026-06-05 default) is not equivalent
    to any of the three published GRPO variants.
    """
    masked = per_tok * mask
    return masked.view(masked.size(0), -1).sum(dim=-1)


def per_sample_avg_logp(
    per_tok: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Average per-token logπ within each sample. Shape: [batch].

    This is the inner sum/|o_i| from the canonical GRPO loss
        L = -(1/G) * sum_i [ (1/|o_i|) * sum_t l_{i,t} ]
    (Shao et al. 2024, DeepSeekMath, eq. 21). Pairs with the per-token
    averaged KL k3 estimator so both terms in the loss are on the same
    scale and the documented beta = 0.02 actually exerts the intended
    regularization (see docs/grpo_loss_formulation.md).
    """
    return (per_tok * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)


def combine_logprob(
    output: Any, codes: torch.Tensor, model: Any, pool: str
) -> torch.Tensor:
    """Per-sample logπ [batch] from a model output, with text/audio aggregated
    per `pool` ('split' = equal text/audio weight, 'token' = pooled token mean).
    Shared by the training step (with grad) and logp_old caching (no_grad)."""
    text_target = codes[:, : model.audio_offset]
    audio_target = codes[:, model.audio_offset : model.audio_offset + model.dep_q]
    tlp, tm = per_token_logprob(output.text_logits, text_target, output.text_mask)
    alp, am = per_token_logprob(output.logits, audio_target, output.mask)
    tl, tm = tlp.flatten(1), tm.flatten(1)
    al, am = alp.flatten(1), am.flatten(1)
    if pool == "token":
        return ((tl * tm).sum(1) + (al * am).sum(1)) / (
            tm.sum(1) + am.sum(1)
        ).clamp(min=1.0)
    return per_sample_avg_logp(tl, tm) + per_sample_avg_logp(al, am)


def cache_logp_old(model, groups, tokenizer, target_sr, pool, clip_level) -> None:
    """Compute & store each sample's behavior-policy logπ (logp_old) at refresh
    time, so off-policy reuse within the refresh window is correctly ratio'd.
    sequence-level stores one scalar/sample; token-level stores the valid
    per-token logπ (text + audio) so the step can form per-token ratios. The
    valid-token sequence is deterministic per wav, so it realigns across the
    different batch padding at step time."""
    # eval mode for a deterministic behavior-policy logprob (matches generation,
    # which ran under eval; guards against future dropout). Restore afterwards.
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for items in groups.values():
            batch, _ = build_batch(items, tokenizer, target_sr=target_sr)
            cond = None
            if batch.condition_attributes is not None:
                cond = model.condition_provider.prepare(batch.condition_attributes)
            out = model(codes=batch.codes, condition_tensors=cond)
            if clip_level == "token":
                codes = batch.codes
                ttgt = codes[:, : model.audio_offset]
                atgt = codes[:, model.audio_offset : model.audio_offset + model.dep_q]
                tlp, tm = per_token_logprob(out.text_logits, ttgt, out.text_mask)
                alp, am = per_token_logprob(out.logits, atgt, out.mask)
                for b, it in enumerate(items):
                    it["logp_old_text"] = tlp[b][tm[b].bool()].cpu()
                    it["logp_old_audio"] = alp[b][am[b].bool()].cpu()
            else:
                lp = combine_logprob(out, batch.codes, model, pool)
                for it, v in zip(items, lp.tolist()):
                    it["logp_old"] = v
    if was_training:
        model.train()


def aggregate_scores(groups: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """Mean reward + per-dimension judge scores (IF/TT/RA) over all samples in
    `groups`, for wandb. Skips dims the judge left None (applicable_avg masking)
    so each dimension averages only where it applies."""
    dims = {
        "instruction_following": "IF",
        "turn_taking": "TT",
        "response_appropriateness": "RA",
    }
    acc: dict[str, list[float]] = {d: [] for d in dims}
    rewards: list[float] = []
    for items in groups.values():
        for it in items:
            rewards.append(float(it.get("reward", 0.0)))
            sc = it.get("scores") or {}
            for d in dims:
                v = sc.get(d)
                if v is not None:
                    acc[d].append(float(v))
    out: dict[str, float] = {}
    if rewards:
        out["rollout/reward"] = sum(rewards) / len(rewards)
    for d, short in dims.items():
        if acc[d]:
            out[f"rollout/{short}"] = sum(acc[d]) / len(acc[d])
    return out


def kl_k3(
    logp_pi: torch.Tensor, logp_ref: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Unbiased low-variance KL estimator from DeepSeek GRPO (Schulman k3)."""
    # Clamp the log-ratio before exp() so a large policy/ref divergence (or a
    # stray logprob) can't blow exp() up to inf/NaN (matches ms-swift's guard).
    log_ratio = (logp_ref - logp_pi).clamp(-20.0, 20.0)
    kl_per_tok = log_ratio.exp() - log_ratio - 1.0
    denom = mask.sum().clamp(min=1.0)
    return (kl_per_tok * mask).sum() / denom


@contextmanager
def lora_disabled(model: torch.nn.Module) -> Iterator[None]:
    """Temporarily zero every LoRALinear's scaling so forward = base policy.

    Relies on moshi.modules.lora.LoRALinear: forward returns
    `frozen_W(x) + lora * scaling`, and lora_B is zero-init. With scaling=0 the
    adapter contribution drops out, which is exactly the reference policy at
    step 0. Only valid for LoRA training (not full finetuning).
    """
    saved: list[tuple[torch.nn.Module, float]] = []
    for module in model.modules():
        if hasattr(module, "lora_A") and hasattr(module, "scaling"):
            saved.append((module, float(module.scaling)))
            module.scaling = 0.0
    try:
        yield
    finally:
        for module, prev in saved:
            module.scaling = prev


def train(config: str, reward_manifest: str, **kwargs):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    cli = parse_cli()
    if cli.config != config or cli.reward_manifest != reward_manifest:
        raise ValueError("Use --config and --reward_manifest in the CLI entrypoint.")

    set_logger(logging.INFO)
    with ExitStack() as exit_stack:
        _train(args, cli, exit_stack)
    logger.info("Closed everything!")


def _train(args: TrainArgs, cli: argparse.Namespace, exit_stack: ExitStack) -> None:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if "LOCAL_RANK" in os.environ:
        set_device()
        dist.init_process_group(backend=BACKEND)
    else:
        logger.error(
            "PyTorch environment is not correctly initialized. This message should only be displayed when testing."
        )

    # Differentiate the seed per rank so each rank samples a different group at
    # every step (data-parallel diversity). set_random_seed only seeds
    # torch/cuda; the GRPO group selection uses python's `random` (random.choice
    # / random.sample), so seed that too -- otherwise group selection is not
    # reproducible (it ran off OS entropy). Single-GPU is unaffected (rank 0).
    set_random_seed(args.seed + get_rank())
    random.seed(args.seed + get_rank())
    np.random.seed(args.seed + get_rank())

    run_dir = Path(args.run_dir)
    if is_torchrun():
        if run_dir.exists() and not args.overwrite_run_dir:
            raise RuntimeError(
                f"Run dir {run_dir} already exists. Make sure to either rename `run_dir` or remove {run_dir}."
            )
        elif run_dir.exists() and get_rank() == 0:
            # Only rank 0 clears the dir; the barrier below makes the other ranks
            # wait. Without the rank gate, ranks race in rmtree and hit
            # FileNotFoundError on entries a peer already deleted.
            main_logger_info(f"Removing run dir {run_dir}...")
            shutil.rmtree(run_dir)

    if args.full_finetuning:
        assert not args.lora.enable, "LoRA should not be enabled for full finetuning."
    else:
        assert args.lora.enable, "LoRA should be enabled for partial finetuning"

    dist.barrier()
    run_dir.mkdir(exist_ok=True, parents=True)
    args_path = run_dir / "args.yaml"
    if get_rank() == 0 and not args_path.exists():
        args.save(args_path)

    main_logger_info(f"TrainArgs: {pprint.pformat(dataclasses.asdict(args))}")

    metrics_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="train",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    main_logger_info("Loading Mimi and Moshi...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=args.moshi_paths.hf_repo_id,
        moshi_weights=args.moshi_paths.moshi_path,
        mimi_weights=args.moshi_paths.mimi_path,
        tokenizer=args.moshi_paths.tokenizer_path,
        config_path=args.moshi_paths.config_path,
    )

    lm_config = (
        loaders._lm_kwargs
        if checkpoint_info.raw_config is None
        else checkpoint_info.raw_config
    )
    lm_config["lora"] = args.lora.enable
    lm_config["lora_rank"] = args.lora.rank
    lm_config["lora_scaling"] = args.lora.scaling

    mimi = checkpoint_info.get_mimi(device="cuda")
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    model = get_fsdp_model(args, checkpoint_info)
    spm = checkpoint_info.get_text_tokenizer()

    # Manual data-parallel (online multi-GPU): the model is a raw replica on every
    # rank (no FSDP, so in-process generation works). The per-rank seed gave each
    # rank a DIFFERENT LoRA init, so broadcast rank 0's LoRA to make every replica
    # identical; grads are all-reduced each step (below) to keep them in sync.
    # Gate on MOSHI_NO_SHARD too: only that mode returns a raw replica. A plain
    # multi-GPU run (FSDP FULL_SHARD) must NOT manual-broadcast/all-reduce or it
    # double-syncs and corrupts FSDP's own reduction.
    is_dp = os.environ.get("MOSHI_NO_SHARD") == "1" and get_world_size() > 1
    if is_dp:
        for p in model.parameters():
            if p.requires_grad:
                dist.broadcast(p.data, src=0)
        main_logger_info(f"Manual DP: broadcast LoRA from rank 0 to {get_world_size()} ranks")

    interleaver = Interleaver(
        spm,
        mimi.frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver, duration_sec=args.duration_sec
    )

    # Online mode generates rollouts in-process each refresh; the static manifest
    # is only loaded for the offline (precomputed reward) path.
    online_state = None
    if cli.online:
        # In-process generation needs a RAW replicated model. Multi-GPU online
        # therefore REQUIRES MOSHI_NO_SHARD=1 (manual data-parallel); otherwise
        # get_fsdp_model returns an FSDP FULL_SHARD model whose params are sharded
        # and whose `.module` cannot drive LMGen -> generation breaks. Fail loud.
        if get_world_size() > 1 and os.environ.get("MOSHI_NO_SHARD") != "1":
            raise RuntimeError(
                "Online multi-GPU requires MOSHI_NO_SHARD=1 (manual data-parallel "
                "raw replicas). Without it the model is FSDP-sharded and in-process "
                "generation fails. Set MOSHI_NO_SHARD=1 (the pipeline does this)."
            )
        if cli.prompts_per_iter < get_world_size():
            raise RuntimeError(
                f"--prompts_per_iter ({cli.prompts_per_iter}) must be >= world_size "
                f"({get_world_size()}): each rank gets prompts[rank::world], so a "
                "smaller window leaves some rank with an empty shard. Increase "
                "PROMPTS or reduce ONLINE_GPUS."
            )
        repo_root = cli.repo_root or str(Path(__file__).resolve().parents[1])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from gametime.scripts.rl.collect_moshi_rollouts import maybe_set_temp
        from gametime.utils.moshi_utils import get_frame_size
        from moshi.models import LMGen

        # manual_dp / single-GPU: model is the raw LMModel (or has .module under a
        # wrapper); LMGen drives it directly.
        gen_model = getattr(model, "module", model)  # raw LMModel (manual DP / single GPU)
        lm_gen = LMGen(gen_model, **checkpoint_info.lm_gen_config)
        maybe_set_temp(lm_gen, cli.gen_temp, cli.gen_temp_text)
        with Path(cli.egs_file).open() as f:
            egs = [json.loads(x) for x in f if x.strip()]
        random.Random(cli.shuffle_seed).shuffle(egs)
        online_state = {
            "lm_gen": lm_gen,
            "frame_size": get_frame_size(mimi),
            "egs": egs,
            "cursor": cli.start_cursor,
            "since_refresh": 0,
        }
        main_logger_info(
            f"[online] {len(egs)} prompts, paging {cli.prompts_per_iter}/refresh, "
            f"refresh_every={cli.refresh_every} step(s), cursor={cli.start_cursor}"
        )
        groups, group_ids = {}, []
    else:
        groups = load_reward_manifest(Path(cli.reward_manifest))
        group_ids = [gid for gid, items in groups.items() if len(items) >= cli.min_group_size]
        if not group_ids:
            raise ValueError("No reward groups meet min_group_size.")
        if cli.clip_eps > 0:
            main_logger_info(
                f"[offline] clip_eps={cli.clip_eps} has NO effect: the static "
                "reward manifest has no cached logp_old, so the surrogate falls "
                "back to plain group-baseline REINFORCE (clipping is online-only)."
            )

    param_dtype = getattr(torch, args.param_dtype)
    optim_dtype = torch.float32
    optimizer = AdamW(
        model.parameters(),
        lr=args.optim.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.lr,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    )
    state = TrainState(args.max_steps)

    if args.do_ckpt:
        checkpointer = Checkpointer(
            model=model,
            state=state,
            config=lm_config,
            run_dir=run_dir,
            optimizer=optimizer,
            num_ckpt_keep=args.num_ckpt_keep,
            full_finetuning=args.full_finetuning,
        )

    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    model.train()
    torch.cuda.empty_cache()

    if args.batch_size != cli.group_size:
        main_logger_info(
            f"Overriding batch_size={args.batch_size} with group_size={cli.group_size}."
        )

    use_kl = cli.kl_coef > 0.0
    if use_kl and not args.lora.enable:
        raise NotImplementedError(
            "kl_coef>0 currently requires LoRA training; full-finetuning needs a "
            "second frozen ref model that this script does not load."
        )
    if use_kl:
        main_logger_info(
            f"GRPO with KL penalty β={cli.kl_coef} against LoRA-disabled reference."
        )
    else:
        main_logger_info("GRPO with KL disabled (REINFORCE + group baseline).")

    # Seamless resume: restore optimizer momentum, LR-schedule position, step
    # counter, RNG and paging cursor (the lora weights were already loaded via
    # lora_weight in get_fsdp_model). optimizer/scheduler/step are identical
    # across ranks (grads are synced), so rank-0's state loads correctly on all.
    if cli.resume_state and Path(cli.resume_state).exists():
        # Load to CPU; the RNG states must stay CPU ByteTensors and the optimizer
        # state is moved to the params' device explicitly below (loading straight
        # to "cuda" breaks both).
        sd = torch.load(cli.resume_state, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(sd["optimizer"])
        # foreach/fused AdamW requires state tensors on the params' cuda device
        # (the 'step' tensor may stay on CPU — leave it).
        dev = torch.device("cuda", torch.cuda.current_device())
        for st in optimizer.state.values():
            for kk, vv in st.items():
                if kk != "step" and torch.is_tensor(vv):
                    st[kk] = vv.to(dev)
        scheduler.load_state_dict(sd["scheduler"])
        state.step = int(sd["step"])
        try:
            torch.set_rng_state(sd["rng_torch"])  # CPU ByteTensor
            if sd.get("rng_cuda"):
                torch.cuda.set_rng_state(sd["rng_cuda"][0])  # this rank's 1 device
            random.setstate(sd["rng_python"])
            np.random.set_state(sd["rng_numpy"])
        except Exception as exc:  # noqa: BLE001
            main_logger_info(f"[resume] RNG restore skipped: {exc!r}")
        if online_state is not None and sd.get("cursor") is not None:
            online_state["cursor"] = sd["cursor"]
            online_state["since_refresh"] = sd.get("since_refresh", 0)
        main_logger_info(
            f"[resume] restored optimizer/scheduler/step={state.step} from "
            f"{cli.resume_state}"
        )

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps
        optimizer.zero_grad()

        # Online: regenerate rollouts from the CURRENT policy every refresh_every
        # steps (refresh_every=1 => fully on-policy). No model reload -- lm_gen
        # wraps the resident model that was just updated. Always refresh when
        # there is no batch yet (first step / after a skip) -- a step-modulo
        # alone would miss the first step since state.step does not start at 0.
        if online_state is not None and (
            not group_ids or online_state["since_refresh"] >= cli.refresh_every
        ):
            W, R = get_world_size(), get_rank()
            egs, cur, P = online_state["egs"], online_state["cursor"], cli.prompts_per_iter
            full_window = [egs[(cur + i) % len(egs)] for i in range(P)]
            window = full_window[R::W]  # this rank's shard (data-parallel gen)
            online_state["cursor"] = cur + P
            gen_dir = Path(args.run_dir) / "online" / f"step_{state.step:06d}" / f"rank_{R}"
            new_groups = online_generate_score(
                cli, model, online_state["lm_gen"], mimi, spm,
                online_state["frame_size"], window, gen_dir,
                cli.group_size, args.seed + state.step * 1000 + R * 131,
            )
            new_ids = [g for g, it in new_groups.items() if len(it) >= cli.min_group_size]
            online_state["since_refresh"] = 0
            if new_ids:
                groups, group_ids = new_groups, new_ids
                # Per-dimension rollout quality (IF/TT/RA + reward) for wandb.
                online_state["rollout_metrics"] = aggregate_scores(new_groups)
                # Freeze the behavior-policy logprob for the fresh batch so the
                # next refresh_every steps clip correctly against it (#1).
                if cli.clip_eps > 0:
                    cache_logp_old(
                        model, groups, interleaved_tokenizer,
                        int(mimi.sample_rate), cli.logp_pool, cli.clip_level,
                    )
            elif group_ids:
                # Keep the previous batch: every rank MUST take a step together or
                # FSDP/DDP grad all-reduce deadlocks. Never `continue` in the loop.
                main_logger_info(
                    f"[online] step {state.step} rank {R}: empty refresh, reusing prev batch"
                )
            else:
                raise RuntimeError(
                    f"[online] rank {R}: first refresh produced no usable groups"
                )
        if online_state is not None:
            online_state["since_refresh"] += 1

        group_id = pick_group(group_ids, groups, cli.select_by)
        selected = sample_group(
            groups[group_id], cli.group_size, cli.sample_with_replacement
        )
        batch, rewards = build_batch(
            selected, interleaved_tokenizer, target_sr=int(mimi.sample_rate)
        )
        codes = batch.codes
        condition_tensors = None
        if batch.condition_attributes is not None:
            condition_tensors = model.condition_provider.prepare(
                batch.condition_attributes
            )

        output = model(codes=codes, condition_tensors=condition_tensors)
        text_target = codes[:, : model.audio_offset]
        audio_target = codes[:, model.audio_offset : model.audio_offset + model.dep_q]
        text_logp_pi, text_mask = per_token_logprob(
            output.text_logits, text_target, output.text_mask
        )
        audio_logp_pi, audio_mask = per_token_logprob(
            output.logits, audio_target, output.mask
        )
        # Canonical GRPO uses per-sample mean-over-tokens, then group-mean.
        # See docs/grpo_loss_formulation.md for why the previous seq_logprob
        # path (sum over tokens with no per-sample normalization) does not
        # match any of the three published GRPO variants and made the
        # nominal beta = 0.02 KL penalty ~100x weaker than intended.
        # logp tensors are [batch, channels, T] (text: 1 channel, audio: dep_q=8
        # codebooks). Flatten the channel+time dims so the per-sample logprob is
        # [batch] -- otherwise audio keeps its 8-codebook dim and `advantages *
        # logprob` mismatches (advantages is [batch]).
        tl, tm = text_logp_pi.flatten(1), text_mask.flatten(1)
        al, am = audio_logp_pi.flatten(1), audio_mask.flatten(1)
        if cli.logp_pool == "token":
            # Canonical GRPO: pool ALL completion tokens, one mean per sample.
            # Audio (dep_q codebooks × frames) dominates text by token count.
            logprob = ((tl * tm).sum(1) + (al * am).sum(1)) / (
                tm.sum(1) + am.sum(1)
            ).clamp(min=1.0)
        else:
            # "split" (default): mean text + mean audio -> equal text/audio weight
            # regardless of token counts. See docs/grpo_loss_formulation.md.
            logprob = per_sample_avg_logp(tl, tm) + per_sample_avg_logp(al, am)

        kl_term = torch.zeros((), device=logprob.device)
        if use_kl:
            with torch.no_grad(), lora_disabled(model):
                ref_output = model(codes=codes, condition_tensors=condition_tensors)
                text_logp_ref, _ = per_token_logprob(
                    ref_output.text_logits, text_target, output.text_mask
                )
                audio_logp_ref, _ = per_token_logprob(
                    ref_output.logits, audio_target, output.mask
                )
            if cli.logp_pool == "token":
                # Pool text+audio tokens so the KL denominator matches the
                # token-pooled policy term (otherwise KL keeps equal text/audio
                # weight while the PG term is token-proportional).
                pi = torch.cat([text_logp_pi.flatten(1), audio_logp_pi.flatten(1)], 1)
                ref = torch.cat(
                    [text_logp_ref.flatten(1), audio_logp_ref.flatten(1)], 1
                )
                km = torch.cat([text_mask.flatten(1), audio_mask.flatten(1)], 1)
                kl_term = kl_k3(pi, ref.detach(), km)
            else:
                kl_term = kl_k3(
                    text_logp_pi, text_logp_ref.detach(), text_mask
                ) + kl_k3(audio_logp_pi, audio_logp_ref.detach(), audio_mask)

        rewards = rewards * cli.reward_scale
        baseline = rewards.mean()
        advantages = rewards - baseline
        if cli.normalize_advantage:
            advantages = advantages / (advantages.std() + cli.advantage_eps)
        advantages = advantages.detach()

        # Clipped GRPO/PPO surrogate when logp_old is cached (online refresh) and
        # clip_eps>0; otherwise plain group-baseline REINFORCE. At refresh_every=1
        # logp_old == current logprob (ratio≈1), so this is a no-op there.
        eps = cli.clip_eps
        if eps > 0 and cli.clip_level == "token" and all(
            "logp_old_text" in s for s in selected
        ):
            # Per-token ratio/clip (canonical). Scatter the cached valid-token
            # logp_old back into batch-shaped tensors (same per-sample order as
            # build_batch(selected)), form per-token ratios, clip, weight by the
            # per-sample advantage, then mask-average over all tokens per sample.
            A = advantages.view(-1, 1, 1)

            def tok_surrogate(logp_pi, mask, key):
                old = torch.zeros_like(logp_pi)
                for b, s in enumerate(selected):
                    old[b][mask[b].bool()] = s[key].to(logp_pi.device, logp_pi.dtype)
                # Clamp the log-ratio before exp so a single off token can't blow
                # the ratio (and the loss) up to ~1e2 (seen at step 185).
                ratio = torch.exp((logp_pi - old).clamp(-20.0, 20.0))
                clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
                surr = torch.min(ratio * A, clipped * A) * mask
                return surr.sum(dim=(1, 2)), mask.sum(dim=(1, 2))

            t_surr, t_n = tok_surrogate(text_logp_pi, text_mask, "logp_old_text")
            a_surr, a_n = tok_surrogate(audio_logp_pi, audio_mask, "logp_old_audio")
            per_sample = (t_surr + a_surr) / (t_n + a_n).clamp(min=1.0)
            pg_loss = -per_sample.mean()
        elif eps > 0 and all("logp_old" in s for s in selected):
            logp_old = torch.tensor(
                [s["logp_old"] for s in selected], device=logprob.device
            )
            ratio = torch.exp((logprob - logp_old).clamp(-20.0, 20.0))
            clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
            pg_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        else:
            pg_loss = -(advantages * logprob).mean()
        loss = pg_loss + cli.kl_coef * kl_term
        loss.backward()

        # Manual data-parallel: average LoRA grads across ranks so every replica
        # takes the same optimizer step and stays identical. (Each rank trained on
        # its own prompt shard, so this is standard data-parallel grad averaging.)
        if is_dp:
            ws = get_world_size()
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= ws

        upcast_mixed_precision(model.parameters(), optim_dtype=optim_dtype)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        downcast_mixed_precision(model.parameters(), param_dtype=param_dtype)
        last_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        loss_item = loss.item()
        avg_loss = avg_aggregate(loss_item)
        state.end_step(n_batch_tokens=codes.numel())

        if state.step % args.log_freq == 0 or is_last_step:
            logs = {
                "step": state.step,
                "loss": avg_loss,
                "pg_loss": pg_loss.item(),
                "kl": kl_term.item(),
                "reward_mean": rewards.mean().item(),
                "reward_std": rewards.std().item(),
                "adv_mean": advantages.mean().item(),
                "adv_std": advantages.std().item(),
                "lr": last_lr,
            }
            # Per-dimension rollout scores from the most recent refresh (online).
            # rank-0 shard only (the wandb logger is master-only anyway).
            if online_state is not None and "rollout_metrics" in online_state:
                logs.update(online_state["rollout_metrics"])
            if state.step % args.log_freq == 0:
                main_logger_info(
                    f"step={state.step} loss={avg_loss:.4f} pg={logs['pg_loss']:.4f} "
                    f"kl={logs['kl']:.4f} reward={logs['reward_mean']:.3f}"
                )
                metrics_logger.log(logs, step=state.step)
            # Drop the final metrics so online_grpo can forward them to its single
            # wandb run -- works even for short iters where step < log_freq.
            if is_last_step and get_rank() == 0:
                (Path(args.run_dir) / "last_metrics.json").write_text(json.dumps(logs))

        if args.do_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            checkpointer.save_checkpoint(
                save_only_lora=not args.full_finetuning and args.save_adapters,
                dtype=param_dtype,
            )
            # Save the full training state next to the lora so the run can resume
            # seamlessly (optimizer momentum, LR schedule, step, RNG, cursor).
            if get_rank() == 0:
                ckpt_dir = (
                    Path(args.run_dir) / "checkpoints"
                    / f"checkpoint_{state.step:06d}" / "consolidated"
                )
                if ckpt_dir.exists():
                    torch.save(
                        {
                            "step": state.step,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "rng_torch": torch.get_rng_state(),
                            "rng_cuda": torch.cuda.get_rng_state_all(),
                            "rng_python": random.getstate(),
                            "rng_numpy": np.random.get_state(),
                            "cursor": online_state["cursor"] if online_state else None,
                            "since_refresh": (
                                online_state["since_refresh"] if online_state else None
                            ),
                        },
                        ckpt_dir / "train_state.pt",
                    )

    main_logger_info("done!")


if __name__ == "__main__":
    fire.Fire(train)
