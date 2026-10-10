"""Autoregressive Language Generation Interface conditioned on Cognitive Graph Workspace."""

from typing import Optional
import argparse
import torch
import torch.nn.functional as F

from neurograph.config import NeuroGraphConfig
from neurograph.model import NeuroGraphModel


class CognitiveGenerator:
    """Generates autoregressive token sequences conditioned on the recurrent cognitive workspace."""

    def __init__(self, model: NeuroGraphModel):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = False,
        active_goal: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        condition: Optional[str] = None,
    ) -> torch.Tensor:
        """Autoregressively generate tokens conditioned on cognitive graph prefix.

        Args:
            input_ids (torch.Tensor): Prompt token IDs [B, S].
            max_new_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int): Top-k filtering threshold.
            top_p (float): Nucleus sampling probability threshold.
            do_sample (bool): If False, greedy decoding is used.
            active_goal (Optional[torch.Tensor]): Active goal embedding [B, D_goal].
            uncertainty (Optional[torch.Tensor]): Uncertainty embedding [B, D_unc].
            condition (Optional[str]): Condition A, B, C, or D.

        Returns:
            generated_ids (torch.Tensor): Full sequence [B, S + generated_len].
        """
        B, S = input_ids.shape
        cond = condition or self.model.config.condition

        curr_input_ids = input_ids.clone()

        # Step 1: Precompute cognitive workspace prefix tokens once
        prefix_embeds = None
        if cond != "A":
            token_embeds = self.model.encode_context(input_ids)
            steps = 1 if cond == "B" else self.model.config.workspace.recurrent_steps
            ws_out = self.model.workspace(
                context_features=token_embeds,
                active_goal=active_goal,
                uncertainty=uncertainty,
                steps=steps,
            )
            prefix_embeds = self.model.adapter(ws_out["final_nodes"])  # [B, K, D_lm]

        # Step 2: Autoregressive decoding loop
        for _ in range(max_new_tokens):
            token_embeds = self.model.encode_context(curr_input_ids)  # [B, curr_len, D_lm]

            if prefix_embeds is not None:
                full_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            else:
                full_embeds = token_embeds

            lm_out = self.model.base_lm(inputs_embeds=full_embeds)
            next_token_logits = lm_out["logits"][:, -1, :]  # [B, vocab_size]

            if not do_sample or temperature <= 0.0:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                # Temperature scaling
                logits = next_token_logits / max(temperature, 1e-5)

                # Top-K filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                    logits[indices_to_remove] = -float("Inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift indices right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    for b in range(B):
                        indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
                        logits[b, indices_to_remove] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)
            # Early stop if EOS token is encountered (DialoGPT/GPT2 EOS is 50256)
            if (next_token == 50256).all():
                break

        return curr_input_ids


def main():
    parser = argparse.ArgumentParser(description="Generate text using NeuroGraph-LLM.")
    parser.add_argument("--model", type=str, default="mock", help="Model path or 'mock'")
    parser.add_argument("--prompt", type=str, default="Explain how graph-based reasoning works.", help="Prompt string")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling instead of greedy decoding")
    parser.add_argument("--condition", type=str, default="C", choices=["A", "B", "C", "D"], help="Experimental condition")
    args = parser.parse_args()

    cfg = NeuroGraphConfig()
    cfg.lm.model_name_or_path = args.model
    cfg.condition = args.condition

    model = NeuroGraphModel(cfg)
    generator = CognitiveGenerator(model)

    # Encode prompt using tokenizer or deterministic fallback
    prompt_tokens = model.encode_text(args.prompt)
    print(f"Prompt: {args.prompt}")
    print(f"Condition: {args.condition} | Generating {args.max_new_tokens} tokens...")

    output_ids = generator.generate(
        input_ids=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        condition=args.condition,
    )
    print(f"Generated token IDs: {output_ids.squeeze().tolist()}")
    decoded_text = model.decode_tokens(output_ids)
    print(f"Decoded text:\n{decoded_text}")
    print("Generation complete.")


if __name__ == "__main__":
    main()
