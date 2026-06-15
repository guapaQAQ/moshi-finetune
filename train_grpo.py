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
    parser.add_argument("--normalize_advantage", action="store_true", default=True)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--min_group_size", type=int, default=2)
    parser.add_argument("--sample_with_replacement", action="store_true", default=True)
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
                prompt_path, lm_gen, mimi, spm, frame_size, "cuda")
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
    # `mask` anyway, so the clamped value is never used.
    safe_target = target.clamp(0, log_probs.size(-1) - 1)
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


def kl_k3(
    logp_pi: torch.Tensor, logp_ref: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Unbiased low-variance KL estimator from DeepSeek GRPO (Schulman k3)."""
    log_ratio = logp_ref - logp_pi
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

    # Differentiate the seed per rank so each rank samples a different
    # group from the reward manifest at every step. Without the rank
    # offset every rank picks the same group_id (since `random.choice`
    # consumes the same Python random state) and FSDP just averages
    # identical gradients -- no data-parallel diversity benefit on
    # multi-GPU runs. Single-GPU runs are unaffected (get_rank() == 0).
    set_random_seed(args.seed + get_rank())

    run_dir = Path(args.run_dir)
    if is_torchrun():
        if run_dir.exists() and not args.overwrite_run_dir:
            raise RuntimeError(
                f"Run dir {run_dir} already exists. Make sure to either rename `run_dir` or remove {run_dir}."
            )
        elif run_dir.exists():
            main_logger_info(f"Removing run dir {run_dir}...")
            shutil.rmtree(run_dir)

    if args.full_finetuning:
        assert not args.lora.enable, "LoRA should not be enabled for full finetuning."
    else:
        assert args.lora.enable, "LoRA should be enabled for partial finetuning"

    dist.barrier()
    run_dir.mkdir(exist_ok=True, parents=True)
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
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
    is_dp = get_world_size() > 1
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
        repo_root = cli.repo_root or str(Path(__file__).resolve().parents[1])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from gametime.scripts.rl.collect_moshi_rollouts import maybe_set_temp
        from gametime.utils.moshi_utils import get_frame_size
        from moshi.models import LMGen

        # Multi-GPU: model is FSDP(NO_SHARD)-wrapped; LMGen needs the underlying
        # LMModel (full params present under NO_SHARD). Single-GPU: model is the
        # raw LMModel already.
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

        group_id = random.choice(group_ids)
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
        text_logp_avg = per_sample_avg_logp(text_logp_pi.flatten(1), text_mask.flatten(1))
        audio_logp_avg = per_sample_avg_logp(audio_logp_pi.flatten(1), audio_mask.flatten(1))
        logprob = text_logp_avg + audio_logp_avg

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
            kl_term = kl_k3(text_logp_pi, text_logp_ref.detach(), text_mask) + kl_k3(
                audio_logp_pi, audio_logp_ref.detach(), audio_mask
            )

        rewards = rewards * cli.reward_scale
        baseline = rewards.mean()
        advantages = rewards - baseline
        if cli.normalize_advantage:
            advantages = advantages / (advantages.std() + cli.advantage_eps)
        advantages = advantages.detach()

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

    main_logger_info("done!")


if __name__ == "__main__":
    fire.Fire(train)
