# Limitations & Model Card

## 1. Research Disclaimer & Scope

NeuroGraph-LLM is an experimental research prototype created to investigate the integration of recurrent graph memory workspaces with causal language models.

- **No Claim of AGI:** This system is not artificial general intelligence and does not exhibit consciousness, autonomous self-awareness, or human-level cognitive faculties.
- **Not a Biological Simulation:** The "cognitive workspace", "nodes", and "recurrent message passing" are mathematical abstractions inspired by cognitive science hypotheses (such as Global Workspace Theory); they do not simulate biological neural circuits, synaptic plasticity, or neuromorphic dynamics.

---

## 2. Technical Limitations

1. **Environment Realism:**
   Current experiments rely on synthetic 2D grid-world environments and controlled state-transition spaces. Real-world planning (e.g. robotic manipulation or open-ended web tasks) involves continuous observation spaces, stochastic dynamics, and partial observability that exceed the scope of the synthetic benchmark.

2. **Scaling to Multi-Billion Parameter LLMs:**
   While the architecture natively supports any causal model via Hugging Face Transformers (`AutoModelForCausalLM`), full-scale research training on 7B–70B parameter models requires multi-GPU distributed clusters, FSDP/DeepSpeed, and substantial VRAM. The current configuration is optimized for consumer hardware (8–16 GB RAM / 8 GB VRAM) using small/mock models.

3. **Inference Latency Trade-Off:**
   Recurrent cognitive message passing introduces $O(T \cdot N^2)$ additional computation per forward pass. For tasks where one-pass feed-forward intuition suffices, recurrent iterations incur latency without proportional accuracy gains.

4. **Sample Efficiency:**
   Untrained or minimally trained neural rollout planners cannot compete with exact search algorithms (BFS/A*) in deterministic graph domains without adequate demonstration datasets or reinforcement learning.

---

## 3. Hardware Requirements & Out-of-Memory Troubleshooting

- **Minimum:** 4 GB RAM, Dual-core CPU.
- **Recommended:** 16 GB System RAM, NVIDIA GPU with 8 GB+ VRAM.
- **OOM Troubleshooting:**
  - Reduce `workspace.num_nodes` (e.g., from 16 to 4).
  - Reduce `workspace.recurrent_steps` (e.g., from 5 to 2).
  - Decrease batch size and increase `gradient_accumulation_steps`.
  - Enable mixed precision (`mixed_precision: true`).
