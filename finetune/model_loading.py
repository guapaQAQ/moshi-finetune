import copy
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import safetensors.torch
import sentencepiece
import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from moshi.models.lm import LMModel


TEXT_TOKENIZER_NAME = getattr(loaders, "TEXT_TOKENIZER_NAME", "tokenizer_spm_32k_3.model")
MOSHI_NAME = getattr(loaders, "MOSHI_NAME", "model.safetensors")
MIMI_NAME = getattr(
    loaders, "MIMI_NAME", "tokenizer-e351c8d8-checkpoint125.safetensors"
)
DEFAULT_REPO = getattr(loaders, "DEFAULT_REPO", "kyutai/moshiko-pytorch-bf16")

PERSONAPLEX_COMPAT_LM_KWARGS = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 16,
    "card": 2048,
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_causal": True,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}


def is_safetensors(path: Path | str) -> bool:
    fn = getattr(loaders, "_is_safetensors", None)
    if fn is not None:
        return fn(path)
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def _load_raw_config(config_path: str | Path | None) -> dict[str, Any] | None:
    if config_path is None:
        return None
    path = Path(config_path)
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def get_lm_config(checkpoint_info: Any) -> dict[str, Any]:
    repo = str(getattr(checkpoint_info, "hf_repo", "") or "").lower()
    if "personaplex" in repo:
        return copy.deepcopy(PERSONAPLEX_COMPAT_LM_KWARGS)

    raw = getattr(checkpoint_info, "raw_config", None)
    if raw is not None:
        return copy.deepcopy(raw)
    return copy.deepcopy(getattr(loaders, "_lm_kwargs", {}))


def _patch_state_dict_for_personaplex(
    model: LMModel,
    state_dict: dict[str, torch.Tensor],
    copy_missing_weights: bool = True,
) -> dict[str, torch.Tensor]:
    model_sd = model.state_dict()

    # PersonaPlex checkpoints may need depformer attention weights expanded to
    # match the current model shape.
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
            if tensor.shape != model_sd[name].shape:
                missing = tensor if copy_missing_weights else model_sd[name][tensor.shape[0] :]
                state_dict[name] = torch.concat([tensor, missing], dim=0)

    # PersonaPlex can also omit the duplicated 8..15 depformer weights. Mirror
    # the inference loader behavior by copying 0..7 into the missing slots.
    if copy_missing_weights:
        to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
        for name in model_sd.keys():
            if name in state_dict:
                continue
            replaced = False
            for old, new in zip(range(8), range(8, 16)):
                for rep in to_replace:
                    needle = f"{rep}.{new}."
                    if needle in name:
                        src = name.replace(needle, f"{rep}.{old}.")
                        if src in state_dict:
                            state_dict[name] = state_dict[src]
                            replaced = True
                        break
                if replaced:
                    break

    return state_dict


@dataclass
class CompatCheckpointInfo:
    hf_repo: str | None
    moshi_weights: str | Path
    mimi_weights: str | Path
    tokenizer_path: str | Path
    config_path: str | Path | None = None
    raw_config: dict[str, Any] | None = None

    @classmethod
    def from_hf_repo(
        cls,
        hf_repo: str | None,
        moshi_weights: str | None,
        mimi_weights: str | None,
        tokenizer: str | None,
        config_path: str | None,
    ) -> "CompatCheckpointInfo":
        repo = hf_repo or DEFAULT_REPO

        if moshi_weights is None:
            moshi_weights = hf_hub_download(repo, MOSHI_NAME)
        if mimi_weights is None:
            mimi_weights = hf_hub_download(repo, MIMI_NAME)
        if tokenizer is None:
            tokenizer = hf_hub_download(repo, TEXT_TOKENIZER_NAME)
        if config_path is None:
            try:
                config_path = hf_hub_download(repo, "config.json")
            except Exception:
                config_path = None

        return cls(
            hf_repo=repo,
            moshi_weights=moshi_weights,
            mimi_weights=mimi_weights,
            tokenizer_path=tokenizer,
            config_path=config_path,
            raw_config=_load_raw_config(config_path),
        )

    def get_mimi(self, device: str | torch.device = "cpu"):
        return loaders.get_mimi(self.mimi_weights, device=device)

    def get_text_tokenizer(self):
        return sentencepiece.SentencePieceProcessor(str(self.tokenizer_path))

    def get_moshi(
        self,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        lm_kwargs_overrides: dict[str, Any] | None = None,
        load_weight: bool = True,
    ) -> LMModel:
        lm_kwargs = get_lm_config(self)

        sig = inspect.signature(LMModel)
        accepted = set(sig.parameters.keys())
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if lm_kwargs_overrides:
            unsupported: list[str] = []
            safe_overrides: dict[str, Any] = {}
            for key, value in lm_kwargs_overrides.items():
                if key in lm_kwargs or key in accepted:
                    safe_overrides[key] = value
                else:
                    unsupported.append(key)
            if unsupported:
                raise RuntimeError(
                    "The installed `moshi` package does not support the requested "
                    f"LM settings: {', '.join(sorted(unsupported))}. "
                    "Either use a compatible `moshi` version or remove those "
                    "settings from the training path."
                )
            lm_kwargs.update(safe_overrides)
        if accepts_var_kwargs:
            ctor_kwargs = dict(lm_kwargs)
        else:
            ctor_kwargs = {k: v for k, v in lm_kwargs.items() if k in accepted}
        if "device" in accepted:
            ctor_kwargs["device"] = device
        if "dtype" in accepted:
            ctor_kwargs["dtype"] = dtype

        if str(device) == "meta":
            with torch.device("meta"):
                model = LMModel(**ctor_kwargs)
        else:
            model = LMModel(**ctor_kwargs)
            model = model.to(device=device, dtype=dtype)

        if not load_weight:
            return model

        state_dict = safetensors.torch.load_file(str(self.moshi_weights))
        state_dict = _patch_state_dict_for_personaplex(model, state_dict)
        model.load_state_dict(state_dict, strict=False, assign=True)
        if str(device) != "meta":
            model = model.to(device=device, dtype=dtype)
        return model


def build_checkpoint_info(
    hf_repo: str | None,
    moshi_weights: str | None,
    mimi_weights: str | None,
    tokenizer: str | None,
    config_path: str | None,
) -> Any:
    repo_name = (hf_repo or "").lower()
    if "personaplex" in repo_name:
        # PersonaPlex config.json may not include fields expected by newer
        # upstream CheckpointInfo (e.g. dep_q). Use the local compat loader.
        return CompatCheckpointInfo.from_hf_repo(
            hf_repo=hf_repo,
            moshi_weights=moshi_weights,
            mimi_weights=mimi_weights,
            tokenizer=tokenizer,
            config_path=config_path,
        )

    checkpoint_cls = getattr(loaders, "CheckpointInfo", None)
    if checkpoint_cls is not None:
        return checkpoint_cls.from_hf_repo(
            hf_repo=hf_repo,
            moshi_weights=moshi_weights,
            mimi_weights=mimi_weights,
            tokenizer=tokenizer,
            config_path=config_path,
        )

    return CompatCheckpointInfo.from_hf_repo(
        hf_repo=hf_repo,
        moshi_weights=moshi_weights,
        mimi_weights=mimi_weights,
        tokenizer=tokenizer,
        config_path=config_path,
    )
