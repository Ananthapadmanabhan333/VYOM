"""NeuroGraph-LLM Core Architecture Module.

Integrates:
1. Base Causal Language Model (with optional LoRA).
2. Dynamic Cognitive Workspace (with recurrent graph updates M(t+1) = GraphUpdate(M(t), H, G(t))).
3. Persistent Episodic Memory.
4. World-State Predictor.
5. Goal-Directed Planning Interface.
6. Graph-to-Language Adapter: zero-initialised gated cross-attention (default) or the
   legacy virtual-prefix adapter (`workspace.adapter_mode = "prefix"`).
"""

from typing import Optional, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurograph.config import NeuroGraphConfig, LMConfig
from neurograph.cognitive_workspace import DynamicCognitiveWorkspace
from neurograph.memory import EpisodicMemory
from neurograph.world_model import WorldStatePredictor


class LoRALinear(nn.Module):
    """Built-in lightweight LoRA adapter for linear layers."""

    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.randn(r, in_features) * (1.0 / math.sqrt(r)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = F.linear(self.dropout(x), self.lora_B @ self.lora_A) * self.scaling
        return base_out + lora_out


class TinyCausalAttention(nn.Module):
    """Causal Multi-Head Self-Attention for mock/tiny LM."""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 512):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

        # Precompute causal mask
        mask = torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv_proj(x).view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2)  # [B, H, S, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]

        # Apply causal mask
        causal = self.causal_mask[:S, :S].unsqueeze(0).unsqueeze(1)  # [1, 1, S, S]
        scores = scores + causal

        if attention_mask is not None:
            # attention_mask: [B, S] -> [B, 1, 1, S]
            pad_mask = (1.0 - attention_mask[:, None, None, :].float()) * -1e9
            scores = scores + pad_mask

        probs = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v)  # [B, H, S, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(attn_out)


class TinyTransformerBlock(nn.Module):
    """Transformer decoder block for tiny/mock causal LM."""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 512):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TinyCausalAttention(d_model, num_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class MockCausalLM(nn.Module):
    """Tiny local causal language model for tests and fast smoke execution."""

    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.d_model)

        self.layers = nn.ModuleList([
            TinyTransformerBlock(config.d_model, config.num_heads, config.max_seq_len)
            for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.tok_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if inputs_embeds is None:
            assert input_ids is not None
            B, S = input_ids.shape
            inputs_embeds = self.tok_embeddings(input_ids)
        else:
            B, S, _ = inputs_embeds.shape

        positions = torch.arange(S, device=inputs_embeds.device).unsqueeze(0).expand(B, -1)
        hidden = inputs_embeds + self.pos_embeddings(positions)

        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attention_mask)

        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits, "hidden_states": hidden}


class GraphLanguageAdapter(nn.Module):
    """Injects cognitive workspace states into causal LM representations.

    Projects workspace node states M into virtual prefix prompt tokens [P_1, ..., P_K],
    which are prepended to the sequence embeddings. Causal attention ensures that prefix
    tokens condition all subsequent language generation without future-token leakage.
    """

    def __init__(self, node_dim: int, lm_dim: int, num_prefix_tokens: int = 4):
        super().__init__()
        self.node_dim = node_dim
        self.lm_dim = lm_dim
        self.num_prefix_tokens = num_prefix_tokens

        self.prefix_proj = nn.Sequential(
            nn.Linear(node_dim, lm_dim),
            nn.LayerNorm(lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim),
        )

    def forward(self, workspace_nodes: torch.Tensor) -> torch.Tensor:
        """Args:

            workspace_nodes (torch.Tensor): [B, N, D_node]

        Returns:
            prefix_embeds (torch.Tensor): [B, K, D_lm]
        """
        B, N, _ = workspace_nodes.shape
        K = min(self.num_prefix_tokens, N)
        selected_nodes = workspace_nodes[:, :K, :]  # [B, K, D_node]
        prefix_embeds = self.prefix_proj(selected_nodes)  # [B, K, D_lm]
        return prefix_embeds


class GatedCrossAttentionAdapter(nn.Module):
    """Injects workspace state into token embeddings via gated cross-attention.

    For every token embedding e_i (prompt *and* generated tokens):

        e_i' = e_i + tanh(g) * W_o · Attn(q = W_q LN(e_i), K/V = W_k,v P(M))

    where M are the final workspace nodes, P projects nodes to the LM width and g is a
    scalar gate initialised to 0. Properties:

    * Exact identity at initialisation: an untrained workspace cannot corrupt a
      pretrained LM (the legacy prefix adapter raised DialoGPT perplexity ~23x).
    * No sequence-length change and no positional shift, so it composes with KV caching:
      each newly generated token is injected independently against the fixed nodes.
    * Token-specific: each position retrieves different workspace content.
    """

    def __init__(self, node_dim: int, lm_dim: int, num_heads: int = 4, gate_init: float = 0.0):
        super().__init__()
        if lm_dim % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.head_dim = lm_dim // num_heads
        self.node_proj = nn.Sequential(nn.Linear(node_dim, lm_dim), nn.LayerNorm(lm_dim))
        self.query_norm = nn.LayerNorm(lm_dim)
        self.q_proj = nn.Linear(lm_dim, lm_dim, bias=False)
        self.k_proj = nn.Linear(lm_dim, lm_dim, bias=False)
        self.v_proj = nn.Linear(lm_dim, lm_dim, bias=False)
        self.out_proj = nn.Linear(lm_dim, lm_dim)
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))

    def gate_value(self) -> float:
        """Current effective gate tanh(g); 0.0 means the workspace has no effect."""
        return float(torch.tanh(self.gate.detach()).item())

    def forward(self, token_embeds: torch.Tensor, workspace_nodes: torch.Tensor) -> torch.Tensor:
        """Args:

            token_embeds (torch.Tensor): [B, S, D_lm]
            workspace_nodes (torch.Tensor): [B, N, D_node]

        Returns:
            torch.Tensor: injected embeddings [B, S, D_lm]
        """
        B, S, D = token_embeds.shape
        N = workspace_nodes.shape[1]
        nodes = self.node_proj(workspace_nodes.to(token_embeds.dtype))  # [B, N, D]
        q = self.q_proj(self.query_norm(token_embeds)).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(nodes).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(nodes).view(B, N, self.num_heads, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, D)
        return token_embeds + torch.tanh(self.gate) * self.out_proj(ctx)


class NeuroGraphModel(nn.Module):
    """End-to-end NeuroGraph-LLM Architecture."""

    def __init__(self, config: NeuroGraphConfig):
        super().__init__()
        self.config = config

        # 1. Base Language Model
        self.lm_backend = "mock"
        self.lm_load_error: Optional[str] = None
        if config.lm.model_name_or_path == "mock":
            self.base_lm = MockCausalLM(config.lm)
        else:
            try:
                from transformers import AutoModelForCausalLM
                self.base_lm = AutoModelForCausalLM.from_pretrained(config.lm.model_name_or_path)
                self.lm_backend = "huggingface"
            except Exception as exc:
                # Fallback keeps tests/offline use working, but is recorded and surfaced
                # via `lm_backend` / `lm_load_error` instead of failing silently.
                self.lm_load_error = f"{type(exc).__name__}: {exc}"
                self.base_lm = MockCausalLM(config.lm)

        # Handle freezing and LoRA
        if config.lm.freeze_base_model:
            for p in self.base_lm.parameters():
                p.requires_grad = False

        if config.lm.use_lora:
            try:
                from peft import LoraConfig, get_peft_model
                peft_config = LoraConfig(
                    r=config.lm.lora_r,
                    lora_alpha=config.lm.lora_alpha,
                    lora_dropout=config.lm.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.base_lm = get_peft_model(self.base_lm, peft_config)
            except Exception:
                # Built-in lightweight LoRALinear fallback
                for name, module in self.base_lm.named_children():
                    if isinstance(module, nn.Linear):
                        setattr(
                            self.base_lm,
                            name,
                            LoRALinear(
                                module,
                                r=config.lm.lora_r,
                                alpha=config.lm.lora_alpha,
                                dropout=config.lm.lora_dropout,
                            ),
                        )

        # 2. Tokenizer initialization
        self.tokenizer = None
        if config.lm.model_name_or_path != "mock":
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(config.lm.model_name_or_path, use_fast=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception:
                self.tokenizer = None

        # 3. Dynamic Cognitive Workspace
        # Dynamically inspect LM embedding dimension from base model config
        if hasattr(self.base_lm, "config") and hasattr(self.base_lm.config, "hidden_size"):
            lm_dim = self.base_lm.config.hidden_size
        elif hasattr(self.base_lm, "config") and hasattr(self.base_lm.config, "d_model"):
            lm_dim = self.base_lm.config.d_model
        else:
            lm_dim = config.lm.d_model

        self.lm_dim = lm_dim
        self.workspace = DynamicCognitiveWorkspace(config.workspace, context_dim=lm_dim)

        # 4. Persistent Episodic Memory
        self.memory = EpisodicMemory(config.memory)

        # 5. World-State Predictor
        self.world_model = WorldStatePredictor(config.world_model)

        # 6. Graph-to-Language Adapter
        if config.workspace.adapter_mode == "prefix":
            self.adapter: nn.Module = GraphLanguageAdapter(
                node_dim=config.workspace.node_dim,
                lm_dim=lm_dim,
                num_prefix_tokens=min(4, config.workspace.num_nodes),
            )
        else:
            self.adapter = GatedCrossAttentionAdapter(
                node_dim=config.workspace.node_dim,
                lm_dim=lm_dim,
                gate_init=config.workspace.adapter_gate_init,
            )

        # 6. Action selection head from pooled workspace state
        self.action_head = nn.Linear(config.workspace.node_dim, config.world_model.action_dim)

    # ------------------------------------------------------------------ helpers
    @property
    def eos_token_id(self) -> Optional[int]:
        if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
            return int(self.tokenizer.eos_token_id)
        return None

    @property
    def supports_static_connectivity(self) -> bool:
        """Condition D needs a workspace built with dynamic_connectivity=False."""
        return not self.workspace.connectivity.dynamic

    def adapter_gate(self) -> Optional[float]:
        """Effective injection gate (gated_cross mode) or None for the legacy prefix adapter."""
        return self.adapter.gate_value() if isinstance(self.adapter, GatedCrossAttentionAdapter) else None

    def encode_context(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return raw input token embeddings from the base LM (no positional information)."""
        embed_fn = self.base_lm.get_input_embeddings()
        return embed_fn(input_ids)

    def lm_call(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """Run the base LM uniformly for HF and mock backends.

        Returns a dict with keys: logits, loss, hidden (final layer or None), past.
        KV caching is only available for Hugging Face backends.
        """
        if isinstance(self.base_lm, MockCausalLM):
            out = self.base_lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            return {"logits": out["logits"], "loss": out["loss"], "hidden": out["hidden_states"], "past": None}
        out = self.base_lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        hidden = out.hidden_states[-1] if output_hidden_states and out.hidden_states is not None else None
        return {
            "logits": out.logits,
            "loss": getattr(out, "loss", None),
            "hidden": hidden,
            "past": getattr(out, "past_key_values", None) if use_cache else None,
        }

    def workspace_context(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Features the workspace reads, per `workspace.context_source`.

        'hidden' runs the base LM once over the prompt and returns final hidden states,
        which are contextual and order-aware. 'embeddings' returns raw embeddings.
        """
        embeds = self.encode_context(input_ids)
        if self.config.workspace.context_source == "embeddings":
            return embeds
        lm_frozen = not any(p.requires_grad for p in self.base_lm.parameters())
        with torch.set_grad_enabled(torch.is_grad_enabled() and not lm_frozen):
            out = self.lm_call(embeds, attention_mask=attention_mask, output_hidden_states=True)
        return out["hidden"]

    def run_workspace(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        condition: Optional[str] = None,
        steps: Optional[int] = None,
        active_goal: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        halt_threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run the cognitive workspace for a condition; returns None for condition A."""
        cond = condition or self.config.condition
        if cond == "A":
            return None
        if cond == "D" and not self.supports_static_connectivity:
            raise ValueError(
                "Condition D requires a model built with workspace.dynamic_connectivity=False"
            )
        if cond == "C" and self.supports_static_connectivity:
            raise ValueError(
                "Condition C requires a model built with workspace.dynamic_connectivity=True"
            )
        n_steps = 1 if cond == "B" else (steps or self.config.workspace.recurrent_steps)
        context = self.workspace_context(input_ids, attention_mask)
        return self.workspace(
            context_features=context,
            context_mask=attention_mask,
            active_goal=active_goal,
            uncertainty=uncertainty,
            steps=n_steps,
            halt_threshold=halt_threshold if cond != "B" else None,
        )

    def inject(self, token_embeds: torch.Tensor, workspace_outputs: Dict[str, Any]) -> torch.Tensor:
        """Apply the gated cross-attention adapter (gated_cross mode only)."""
        assert isinstance(self.adapter, GatedCrossAttentionAdapter)
        return self.adapter(token_embeds, workspace_outputs["final_nodes"])

    def encode_text(self, text: str, max_length: Optional[int] = None) -> torch.Tensor:
        """Encode text prompt into input token IDs using tokenizer or deterministic fallback."""
        if self.tokenizer is not None:
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length or self.config.lm.max_seq_len,
                truncation=True,
            )
            return enc["input_ids"]
        else:
            words = text.split()
            v = self.config.lm.vocab_size
            return torch.tensor([[abs(hash(w)) % (v - 1) + 1 for w in words]], dtype=torch.long)

    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to human-readable text."""
        if self.tokenizer is not None:
            flat_ids = token_ids.squeeze().tolist()
            if isinstance(flat_ids, int):
                flat_ids = [flat_ids]
            return self.tokenizer.decode(flat_ids, skip_special_tokens=True)
        else:
            return " ".join(f"tok_{t}" for t in token_ids.squeeze().tolist())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        active_goal: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        condition: Optional[str] = None,
        current_state: Optional[torch.Tensor] = None,
        target_action: Optional[torch.Tensor] = None,
        target_next_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """End-to-end forward pass supporting all 4 experimental conditions.

        Conditions:
            'A': Base pretrained LLM (bypasses cognitive workspace).
            'B': Base LLM + One-pass graph module (steps=1).
            'C': Base LLM + Recurrent dynamic graph workspace (steps=T, dynamic edges).
            'D': Base LLM + Recurrent workspace with static connectivity (steps=T, static edges).
        """
        cond = condition or self.config.condition
        B, S = input_ids.shape
        device = input_ids.device

        token_embeds = self.encode_context(input_ids)  # [B, S, D_lm]

        action_logits = None
        reg_loss = torch.tensor(0.0, device=device)
        full_embeds, full_mask, full_labels = token_embeds, attention_mask, labels

        workspace_outputs = self.run_workspace(
            input_ids,
            attention_mask=attention_mask,
            condition=cond,
            active_goal=active_goal,
            uncertainty=uncertainty,
            halt_threshold=self.config.workspace.halt_threshold,
        )

        if workspace_outputs is not None:
            reg_loss = workspace_outputs["reg_loss"]
            action_logits = self.action_head(workspace_outputs["pooled_state"])

            if isinstance(self.adapter, GatedCrossAttentionAdapter):
                # Workspace -> every token via gated cross-attention (no length change).
                full_embeds = self.inject(token_embeds, workspace_outputs)
            else:
                # Legacy prefix mode: prepend K virtual tokens; mask them out of the loss.
                prefix_embeds = self.adapter(workspace_outputs["final_nodes"])  # [B, K, D_lm]
                K = prefix_embeds.shape[1]
                full_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
                prefix_mask = torch.ones(B, K, dtype=torch.long, device=device)
                full_mask = (
                    torch.cat([prefix_mask, attention_mask], dim=1)
                    if attention_mask is not None
                    else torch.ones(B, K + S, dtype=torch.long, device=device)
                )
                if labels is not None:
                    prefix_labels = torch.full((B, K), -100, dtype=torch.long, device=device)
                    full_labels = torch.cat([prefix_labels, labels], dim=1)

        lm_out = self.lm_call(full_embeds, attention_mask=full_mask, labels=full_labels)
        lm_loss = lm_out["loss"]
        logits = lm_out["logits"]

        # World model loss if state/action pairs provided
        world_loss = torch.tensor(0.0, device=device)
        if current_state is not None and target_action is not None and target_next_state is not None:
            wm_out = self.world_model.compute_loss(current_state, target_action, target_next_state)
            world_loss = wm_out["loss"]

        # Action prediction loss if target actions provided
        action_loss = torch.tensor(0.0, device=device)
        if action_logits is not None and target_action is not None:
            action_loss = F.cross_entropy(action_logits, target_action)

        # Combined multi-objective loss with documented weighting
        weights = self.config.training.loss_weights
        total_loss = (
            (lm_loss if lm_loss is not None else 0.0) * weights.get("lm", 1.0)
            + world_loss * weights.get("world", 0.5)
            + action_loss * weights.get("action", 0.5)
            + reg_loss * weights.get("reg", 0.01)
        )

        return {
            "total_loss": total_loss,
            "lm_loss": lm_loss,
            "world_loss": world_loss,
            "action_loss": action_loss,
            "reg_loss": reg_loss,
            "logits": logits,
            "action_logits": action_logits,
            "workspace_outputs": workspace_outputs,
        }
