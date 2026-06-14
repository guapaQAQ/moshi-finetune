import argparse
import dataclasses
import json
import logging
import os
import pprint
import random
import shutil
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
