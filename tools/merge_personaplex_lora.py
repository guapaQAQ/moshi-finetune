#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import safetensors.torch
import torch
from moshi.modules.lora import LoRALinear

from finetune.model_loading import build_checkpoint_info, get_lm_config


def _parse_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _load_lora_weights(model: torch.nn.Module, lora_weight: Path) -> None:
    lora_state = safetensors.torch.load_file(str(lora_weight))
    model_state = model.state_dict()

    missing_shapes: list[str] = []
    unexpected_keys: list[str] = []
    loaded = 0

    with torch.no_grad():
        for key, tensor in lora_state.items():
            if key not in model_state:
                unexpected_keys.append(key)
                continue

            target = model_state[key]
            if target.shape != tensor.shape:
                missing_shapes.append(key)
                continue

            target.copy_(tensor.to(dtype=target.dtype, device=target.device))
            loaded += 1

    if unexpected_keys:
        print(f"[warn] unexpected LoRA keys: {unexpected_keys}")
    if missing_shapes:
        print(f"[warn] LoRA shape mismatches: {missing_shapes}")
    print(
        "[info] loaded LoRA tensors: "
        f"{loaded} (unexpected={len(unexpected_keys)}, shape_mismatch={len(missing_shapes)})"
    )


def _merged_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    def merge_lora(
        module: torch.nn.Module,
        destination: dict[str, torch.Tensor],
        prefix: str,
        *args,
    ):
        weight = module.merge_weight()  # type: ignore[attr-defined]
        destination[prefix + "weight"] = weight

    handles = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            handles.append(module._register_state_dict_hook(merge_lora))

    try:
        state_dict = model.state_dict()
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not any(blocked in k for blocked in ("lora", "frozen"))
        }
        return dict(sorted(state_dict.items()))
    finally:
        for handle in handles:
            handle.remove()


def _infer_lora_hparams(lora_weight: Path) -> tuple[int | None, float | None]:
    config_path = lora_weight.with_name("config.json")
    if not config_path.exists():
        return None, None
    try:
        with config_path.open() as f:
            cfg = json.load(f)
    except Exception:
        return None, None
    rank = cfg.get("lora_rank")
    scaling = cfg.get("lora_scaling")
    try:
        rank = int(rank) if rank is not None else None
    except Exception:
        rank = None
    try:
        scaling = float(scaling) if scaling is not None else None
    except Exception:
        scaling = None
    return rank, scaling


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge PersonaPlex LoRA adapter into full Moshi weights."
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="nvidia/personaplex-7b-v1",
        help="Base HF repo ID for PersonaPlex checkpoint.",
    )
    parser.add_argument(
        "--lora_weight",
        type=Path,
        required=True,
        help="Path to lora.safetensors.",
    )
    parser.add_argument(
        "--output_weight",
        type=Path,
        required=True,
        help="Output path for merged consolidated.safetensors.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used during merge (cpu/cuda).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype used during merge.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="LoRA rank used in training config.",
    )
    parser.add_argument(
        "--lora_scaling",
        type=float,
        default=None,
        help="LoRA scaling used in training config.",
    )
    args = parser.parse_args()

    if not args.lora_weight.exists():
        raise FileNotFoundError(f"LoRA weight not found: {args.lora_weight}")

    inferred_rank, inferred_scaling = _infer_lora_hparams(args.lora_weight.resolve())
    lora_rank = (
        args.lora_rank
        if args.lora_rank is not None
        else (inferred_rank if inferred_rank is not None else 128)
    )
    lora_scaling = (
        args.lora_scaling
        if args.lora_scaling is not None
        else (inferred_scaling if inferred_scaling is not None else 2.0)
    )

    dtype = _parse_dtype(args.dtype)
    output_weight = args.output_weight.resolve()
    output_weight.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_info = build_checkpoint_info(
        hf_repo=args.hf_repo,
        moshi_weights=None,
        mimi_weights=None,
        tokenizer=None,
        config_path=None,
    )

    print("[info] loading base model with LoRA modules...")
    model = checkpoint_info.get_moshi(
        device=args.device,
        dtype=dtype,
        lm_kwargs_overrides={
            "lora": True,
            "lora_rank": lora_rank,
            "lora_scaling": lora_scaling,
        },
        load_weight=True,
    )
    model.eval()

    print(f"[info] loading LoRA adapter: {args.lora_weight}")
    _load_lora_weights(model, args.lora_weight.resolve())

    print("[info] merging LoRA into base weights...")
    merged = _merged_state_dict(model)
    safetensors.torch.save_file(merged, str(output_weight))

    config_path = output_weight.with_name("config.json")
    with config_path.open("w") as f:
        json.dump(get_lm_config(checkpoint_info), f, indent=2)

    print(f"[ok] wrote merged model: {output_weight}")
    print(f"[ok] wrote config: {config_path}")


if __name__ == "__main__":
    main()
