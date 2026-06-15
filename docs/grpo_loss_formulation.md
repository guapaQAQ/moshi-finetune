# GRPO Loss Formulation in `train_grpo.py`

Recording the algorithmic choice between the three canonical "GRPO" loss
aggregations so the paper claim and the code line up. Pre-2026-06-05 the
code shipped a fourth, non-canonical variant; this doc explains why we
switched and what the trade-offs are.

## The three canonical variants (all token-level)

Let `G` = group size, `|o_i|` = number of generated tokens in completion
`i`, and `l_{i,t}` = the inner per-token quantity
`-(advantage_i * (π_θ/π_θ_old) + β * KL_{i,t})` (with the convention that
positive KL adds penalty so the sign is `+β·KL` once we wrap the whole
expression in `-L`).

### 1. Original GRPO (Shao et al. 2024, DeepSeekMath)
$$L_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} l_{i,t}$$

Per-sample average over tokens, then group-mean. **Has length bias**: a
short completion contributes the same total weight as a long completion,
so per-token gradients on short outputs are inflated relative to long
outputs. This is the formula the original paper writes and the one
TRL / Verl reference as "GRPO".

### 2. DAPO (Yu et al. 2025; default in recent TRL)
$$L_{\text{DAPO}} = -\frac{1}{\sum_{i=1}^{G} |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} l_{i,t}$$

Sum every per-token loss, divide by total token count across the group.
Removes length bias (every token contributes the same gradient magnitude
regardless of which sample it came from). TRL's documentation calls this
out as "mitigates length bias in long CoT scenarios" and uses it as the
default `loss_agg_mode`.

### 3. Dr. GRPO (Liu et al. 2025)
$$L_{\text{Dr. GRPO}} = -\frac{1}{LG} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} l_{i,t}$$

Divide by `L · G` where `L` is a *constant* (max completion length) and
`G` is the group size. Also removes length bias; differs from DAPO in
that the denominator does not depend on the actual tokens sampled, so
the loss has a fixed scale across batches even when completions are
short.

## What the code shipped before 2026-06-05

```python
# Old per-sample logp: sum of per-token logπ over the sequence
logprob = seq_logprob(text_logp_pi, text_mask) + seq_logprob(audio_logp_pi, audio_mask)

pg_loss = -(advantages * logprob).mean()
loss    = pg_loss + cli.kl_coef * kl_term      # kl_term is per-token-averaged
```

Expanded, this is

$$L_{\text{old}} = -\frac{1}{G}\sum_{i=1}^{G} \sum_{t=1}^{|o_i|} l_{i,t}^{\text{pg}}
                  + \beta \cdot \frac{1}{\sum_i |o_i|} \sum_{i,t} \text{KL}_{i,t}$$

Two problems:

- **Sequence-level pg vs per-token KL scale mismatch.** The pg term sums
  tokens; the KL term averages tokens. With completion lengths of a few
  hundred to a few thousand tokens, the KL contribution is effectively
  shrunk by 100-1000x at the same `β`. With `β = 0.02` (DeepSeek
  recommendation) the KL penalty barely shows up in the gradient.
- **Severe length bias.** Long completions get gradient ∝ |o_i| (no
  per-sample normalization). On dialogue tasks where length varies by
  task family (OpenEnded vs TimeFast), this concentrates the update on
  whichever group happened to be longer this step.

Neither matches any of the three canonical variants above.

## Choice (effective 2026-06-05): variant 1 (original GRPO)

Three reasons:

1. **Paper claim cleanliness.** "We use GRPO (Shao et al. 2024)" in the
   methods section now corresponds exactly to the published formula.
   Switching to DAPO or Dr. GRPO would require a paragraph explaining
   why we deviate from the canonical formulation.
2. **Length variance is bounded.** Game-Time completions are
   `duration_sec`-bounded by the interleaver (audio frames at the codec
   frame rate plus an interleaved text channel). The completion-length
   range is ~3-5x across task families, not the 100x+ that motivates
   DAPO / Dr. GRPO in long-CoT reasoning settings. The length-bias
   downside of variant 1 is manageable here.
3. **One-line escape hatch.** If we later want to test variant 2 (DAPO),
   we can flip an `--loss_agg_mode` flag rather than reshape the whole
   training loop. Implementation is staged so the variant 1 result lands
   first and a DAPO ablation can be a follow-up.

The KL term remains per-token-averaged (already was), but the pg term
now matches that scale, so `β = 0.02` actually exerts the intended
amount of regularization.

## Concrete code change

```python
# NEW per-sample average-over-tokens helper
def per_sample_avg_logp(per_tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean per-token logπ within each sample. Shape: [batch]."""
    return (per_tok * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

# NEW logprob (per-sample average, summed across text and audio channels)
text_logp_avg  = per_sample_avg_logp(text_logp_pi,  text_mask)
audio_logp_avg = per_sample_avg_logp(audio_logp_pi, audio_mask)
logprob = text_logp_avg + audio_logp_avg          # shape [batch], same as before
pg_loss = -(advantages * logprob).mean()          # group-mean unchanged
loss    = pg_loss + cli.kl_coef * kl_term         # both terms now per-token-averaged
```

`seq_logprob` is kept in the file but unused so future ablations can flip
the aggregation mode without re-introducing the helper.

## Hyperparameter implications

With the corrected scale, `β = 0.02` is the right starting point per
DeepSeek's ablation. Recommended sweep:

| β | Behaviour |
|---|---|
| 0.0 | REINFORCE with group baseline (KL off; useful as ablation) |
| 0.02 | DeepSeek default; KL keeps π close to reference, mild regularization |
| 0.1 | Aggressive KL; π struggles to move much, useful only for short runs |

Log `kl` and `pg_loss` separately each step (already done at line 392-393)
and watch the ratio. If `β·kl_term` is consistently <1% of `pg_loss`, the
KL is too weak and β should go up; if >50%, it's dominating and β should
go down.

## Multi-rank seed (also fixed in the same commit)

Previously `set_random_seed(args.seed)` gave every rank the same seed,
so `random.choice(group_ids)` picked the same group on every rank.
Under FSDP that's not a correctness bug (parameter updates are still
synchronized) but it kills data parallelism: every rank trained on the
same batch, so going from 1 GPU to 4 didn't expose 4x more diverse
groups.

Fix: `set_random_seed(args.seed + get_rank())`. Each rank now samples a
different group per step. For single-GPU runs (the cluster's current
setup with `train_moshi.sh`) this is a no-op.

## Clipped surrogate, KL clamp, text/audio pooling (2026-06)

The objective is a **sequence-level** clipped surrogate (PPO/GRPO-style),
not bare REINFORCE — but note it is NOT the per-token canonical form:
ms-swift/TRL compute the ratio per token *before* aggregation, whereas
here the per-sample logπ is aggregated first and a single scalar ratio is
clipped per sample. At each online refresh the behaviour-policy per-sample
logπ (`logp_old`) is cached (`cache_logp_old`, eval mode); each optimizer
step forms the ratio `exp(logπ − logp_old)` and the clipped surrogate
`min(ratio·A, clip(ratio, 1±ε)·A)`, `ε = --clip_eps` (default 0.2). Moving
to a per-token ratio is a possible future refinement.
`--clip_eps 0` falls back to group-baseline REINFORCE. With
`--refresh_every 1` (fully on-policy) the ratio is ≈1 so the clip is a
no-op; it only bites when a rollout batch is reused off-policy
(`refresh_every > 1`).

The k3 KL log-ratio is clamped to `[-20, 20]` before `exp()` to avoid
inf/NaN under large divergence.

`--logp_pool` selects per-sample logπ aggregation: `split` (default)
averages text and audio separately then sums (equal text/audio weight
regardless of token counts); `token` pools all completion tokens into one
mean (token-proportional, audio-dominant for Moshi). Treat the choice as
an ablation knob; `split` is the project default.

Multi-GPU online uses **manual data-parallel** (raw replicated model, not
FSDP, since FSDP per-block wrapping breaks in-process LMGen generation).
Each rank generates its own prompt shard, so its local group baseline IS
the correct per-prompt advantage; LoRA grads are all-reduced. This is
equivalent to grouped GRPO, not a deviation.

## References

- Shao et al. 2024, "DeepSeekMath: Pushing the Limits of Mathematical
  Reasoning in Open Language Models" — original GRPO formulation
  (arxiv 2402.03300).
- Yu et al. 2025, "DAPO: Detached Advantage Policy Optimization" —
  length-bias-free GRPO variant; TRL's `loss_agg_mode` default.
- Liu et al. 2025, "Why GRPO Needs Normalization" — Dr. GRPO variant,
  arxiv 2503.20783.
- HuggingFace TRL GRPO loss variants docs:
  https://deepwiki.com/huggingface/trl/5.2-grpo-loss-variants-and-clipping
