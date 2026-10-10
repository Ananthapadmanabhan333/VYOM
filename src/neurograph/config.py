"""Configuration dataclasses and loaders for NeuroGraph-LLM."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class LMConfig:
    """Configuration for the base language model."""
    model_name_or_path: str = "mock"
    freeze_base_model: bool = True
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_seq_len: int = 128
    vocab_size: int = 1000
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4


@dataclass
class WorkspaceConfig:
    """Configuration for the dynamic cognitive workspace."""
    num_nodes: int = 8
    node_dim: int = 64
    recurrent_steps: int = 3
    top_k_edges: Optional[int] = None
    dynamic_connectivity: bool = True
    use_layer_norm: bool = True
    dropout: float = 0.1
    goal_dim: int = 32
    uncertainty_dim: int = 16
    num_heads: int = 2
    gate_type: str = "gru"  # 'gru' or 'residual'


@dataclass
class MemoryConfig:
    """Configuration for persistent episodic memory."""
    capacity: int = 1000
    memory_dim: int = 64
    retrieval_top_k: int = 3
    similarity_metric: str = "cosine"  # 'cosine' or 'dot'
    decay_rate: float = 0.0  # optional temporal decay


@dataclass
class WorldModelConfig:
    """Configuration for the predictive world model."""
    state_dim: int = 32
    action_dim: int = 5  # default 5 actions for GridWorld
    hidden_dim: int = 64
    predict_uncertainty: bool = True


@dataclass
class PlannerConfig:
    """Configuration for goal-directed planning."""
    max_depth: int = 15
    discount: float = 0.95
    heuristic_weight: float = 1.0


@dataclass
class TrainingConfig:
    """Configuration for training and optimization."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_steps: int = 100
    warmup_steps: int = 10
    device: str = "cpu"
    seed: int = 42
    mixed_precision: bool = False
    gradient_checkpointing: bool = False
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "lm": 1.0,
        "world": 0.5,
        "action": 0.5,
        "mem": 0.1,
        "reg": 0.01,
    })


@dataclass
class NeuroGraphConfig:
    """Root configuration for NeuroGraph-LLM."""
    experiment_name: str = "default_experiment"
    condition: str = "C"  # A: Base LLM, B: One-pass Graph, C: Recurrent Dynamic Graph, D: Recurrent Static Graph
    lm: LMConfig = field(default_factory=LMConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeuroGraphConfig":
        lm_cfg = LMConfig(**data.get("lm", {}))
        ws_cfg = WorkspaceConfig(**data.get("workspace", {}))
        mem_cfg = MemoryConfig(**data.get("memory", {}))
        wm_cfg = WorldModelConfig(**data.get("world_model", {}))
        pl_cfg = PlannerConfig(**data.get("planner", {}))
        tr_cfg = TrainingConfig(**data.get("training", {}))

        return cls(
            experiment_name=data.get("experiment_name", "default_experiment"),
            condition=data.get("condition", "C"),
            lm=lm_cfg,
            workspace=ws_cfg,
            memory=mem_cfg,
            world_model=wm_cfg,
            planner=pl_cfg,
            training=tr_cfg,
        )

    def save_yaml(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "NeuroGraphConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})
