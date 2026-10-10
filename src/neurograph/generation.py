"""Autoregressive generation conditioned on the cognitive graph workspace.

Key properties
--------------
* The workspace is run **once** over the prompt (``model.run_workspace``); its final
  nodes are then injected into every token through the model's adapter.
* Hugging Face backends use a KV cache: each step feeds only the newest token, so cost
  per token is O(1) in sequence length instead of re-encoding the whole sequence.
* ``stream()`` yields tokens one at a time and honours a ``threading.Event`` for
  cancellation and an optional wall-clock budget.
* ``last_stats`` records measured timings and the stop reason of the last call. These
  numbers are measured, never estimated.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn.functional as F

from neurograph.config import NeuroGraphConfig
from neurograph.model import GatedCrossAttentionAdapter, MockCausalLM, NeuroGraphModel


@dataclass
class GenerationStats:
    condition: str = "A"
    prompt_tokens: int = 0
    new_tokens: int = 0
    workspace_ms: float = 0.0
    first_token_ms: Optional[float] = None
    total_ms: float = 0.0
    tokens_per_s: Optional[float] = None
    kv_cache: bool = False
    stop_reason: str = ""
    workspace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = torch.zeros_like(remove).scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(mask, float("-inf"))
    return logits


def _apply_repetition_penalty(logits: torch.Tensor, seq: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, seq)
    score = torch.where(score < 0, score * penalty, score / penalty)
    return logits.scatter(1, seq, score)


def summarize_workspace(model: NeuroGraphModel, ws_out: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact, JSON-serialisable telemetry computed from real workspace outputs (batch 0)."""
    if ws_out is None:
        return {"active": False, "adapter_gate": model.adapter_gate()}
    adj = ws_out["adj_history"][-1][0].detach().float()  # [N, N]
    n = adj.shape[-1]
    row_entropy = -(adj.clamp_min(1e-12) * adj.clamp_min(1e-12).log()).sum(-1).mean().item()
    return {
        "active": True,
        "steps_executed": int(ws_out["steps_executed"][0].item()) if "steps_executed" in ws_out else None,
        "step_deltas": [round(float(d[0].item()), 5) for d in ws_out.get("step_deltas", [])],
        "adjacency": [[round(float(v), 4) for v in row] for row in adj.tolist()],
        "adjacency_row_entropy": round(row_entropy, 4),
        "adjacency_max_entropy": round(float(torch.log(torch.tensor(float(n))).item()), 4),
        "node_norms": [round(float(v), 4) for v in ws_out["final_nodes"][0].norm(dim=-1).tolist()],
        "adapter_gate": model.adapter_gate(),
        "dynamic_connectivity": not model.supports_static_connectivity,
    }


class CognitiveGenerator:
    """Generates token sequences conditioned on the recurrent cognitive workspace."""

    def __init__(self, model: NeuroGraphModel):
        self.model = model
        self.model.eval()
        self.last_stats = GenerationStats()

    # ---------------------------------------------------------------- internals
    def _context_limit(self) -> int:
        lm = self.model.base_lm
        if isinstance(lm, MockCausalLM):
            return lm.max_seq_len
        cfg = getattr(lm, "config", None)
        for attr in ("n_positions", "max_position_embeddings"):
            if cfg is not None and getattr(cfg, attr, None):
                return int(getattr(cfg, attr))
        return self.model.config.lm.max_seq_len

    def _embed(self, ids: torch.Tensor, ws_out: Optional[Dict[str, Any]]) -> torch.Tensor:
        emb = self.model.encode_context(ids)
        if ws_out is not None and isinstance(self.model.adapter, GatedCrossAttentionAdapter):
            emb = self.model.inject(emb, ws_out)
        return emb

    def _prefix(self, ws_out: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
        if ws_out is not None and not isinstance(self.model.adapter, GatedCrossAttentionAdapter):
            return self.model.adapter(ws_out["final_nodes"])
        return None

    # ------------------------------------------------------------------- public
    @torch.no_grad()
    def stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
        active_goal: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        condition: Optional[str] = None,
        steps: Optional[int] = None,
        halt_threshold: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
        max_time_s: Optional[float] = None,
        stop_on_eos: bool = True,
    ) -> Iterator[torch.Tensor]:
        """Yield the next token tensor [B, 1] at each step. Stats go to ``last_stats``."""
        t0 = time.perf_counter()
        cond = condition or self.model.config.condition
        stats = GenerationStats(condition=cond, prompt_tokens=int(input_ids.shape[1]))
        self.last_stats = stats

        ws_out = self.model.run_workspace(
            input_ids, condition=cond, steps=steps, active_goal=active_goal,
            uncertainty=uncertainty, halt_threshold=halt_threshold,
        )
        stats.workspace_ms = (time.perf_counter() - t0) * 1000
        stats.workspace = summarize_workspace(self.model, ws_out)

        prefix = self._prefix(ws_out)
        k_prefix = 0 if prefix is None else prefix.shape[1]
        limit = self._context_limit()
        eos = self.model.eos_token_id if stop_on_eos else None
        use_cache = not isinstance(self.model.base_lm, MockCausalLM)
        stats.kv_cache = use_cache

        seq = input_ids.clone()
        past = None
        stats.stop_reason = "max_tokens"
        for i in range(max_new_tokens):
            if cancel_event is not None and cancel_event.is_set():
                stats.stop_reason = "cancelled"
                break
            if max_time_s is not None and (time.perf_counter() - t0) > max_time_s:
                stats.stop_reason = "max_time"
                break
            if use_cache:
                if seq.shape[1] + k_prefix >= limit:
                    stats.stop_reason = "context_limit"
                    break
                if past is None:
                    emb = self._embed(seq, ws_out)
                    if prefix is not None:
                        emb = torch.cat([prefix, emb], dim=1)
                else:
                    emb = self._embed(seq[:, -1:], ws_out)
                out = self.model.lm_call(emb, past_key_values=past, use_cache=True)
                past = out["past"]
            else:
                window = seq[:, -(limit - k_prefix):]
                emb = self._embed(window, ws_out)
                if prefix is not None:
                    emb = torch.cat([prefix, emb], dim=1)
                out = self.model.lm_call(emb)
            logits = out["logits"][:, -1, :].float()
            logits = _apply_repetition_penalty(logits, seq, repetition_penalty)

            if not do_sample or temperature <= 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = _filter_logits(logits / max(temperature, 1e-5), top_k, top_p)
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            if i == 0:
                stats.first_token_ms = (time.perf_counter() - t0) * 1000
            if eos is not None and bool((next_token == eos).all()):
                stats.stop_reason = "eos"
                break
            seq = torch.cat([seq, next_token], dim=1)
            stats.new_tokens += 1
            yield next_token

        stats.total_ms = (time.perf_counter() - t0) * 1000
        gen_s = (stats.total_ms - stats.workspace_ms) / 1000
        stats.tokens_per_s = round(stats.new_tokens / gen_s, 2) if stats.new_tokens and gen_s > 0 else None

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generate and return the full sequence [B, S + generated_len].

        Accepts the same keyword arguments as :meth:`stream`. EOS is not appended.
        """
        tokens = list(self.stream(input_ids, **kwargs))
        if not tokens:
            return input_ids.clone()
        return torch.cat([input_ids] + tokens, dim=1)


def main():
    parser = argparse.ArgumentParser(description="Generate text using the Vyom NeuroGraph model.")
    parser.add_argument("--model", type=str, default="mock", help="Model path or 'mock'")
    parser.add_argument("--prompt", type=str, default="Explain how graph-based reasoning works.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--condition", type=str, default="C", choices=["A", "B", "C"])
    args = parser.parse_args()

    cfg = NeuroGraphConfig()
    cfg.lm.model_name_or_path = args.model
    cfg.condition = args.condition

    model = NeuroGraphModel(cfg)
    generator = CognitiveGenerator(model)
    prompt_tokens = model.encode_text(args.prompt)
    print(f"Prompt: {args.prompt}")
    output_ids = generator.generate(
        prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        condition=args.condition,
    )
    print(f"Decoded text:\n{model.decode_tokens(output_ids[:, prompt_tokens.shape[1]:])}")
    print(f"Stats: {generator.last_stats.to_dict()}")


if __name__ == "__main__":
    main()
