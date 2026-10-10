# Research Hypotheses & Theoretical Framework

## 1. Core Research Question

> **Primary Hypothesis:**
> Does an iterative, recurrent graph-based cognitive workspace improve multi-step reasoning, adaptive planning, memory utilization, and out-of-distribution generalization compared to a conventional frozen pretrained LLM or one-pass graph adapter?

Treat this question as an open empirical inquiry rather than a predetermined conclusion.

---

## 2. Competing Hypotheses & Null Baselines

### $H_0$ (The Parameter / Compute Parity Hypothesis)
Any performance improvement observed in a recurrent graph cognitive architecture is solely an artifact of:
1. Additional trainable parameters ($+ \Delta \theta$).
2. Increased forward compute / latency during inference ($O(T \cdot N^2)$).
3. Retrieval augmentation from external memory stores.

Under $H_0$, matching parameters and inference FLOPs with a standard feed-forward layer or wider transformer will achieve parity with the recurrent graph model.

### $H_1$ (The Relational Induction Hypothesis)
Explicit graph message-passing over structured slots enforces an inductive bias that benefits relational reasoning, spatial navigation, and state-transition modeling over unstructured flat context representations.

### $H_2$ (The Recurrent Iteration Hypothesis)
Recurrent processing steps ($T > 1$) allow iterative constraint satisfaction and dynamic routing that cannot be collapsed into a single feed-forward pass ($T = 1$) without significant parameter expansion.

---

## 3. Four Controlled Experimental Conditions

| Condition | Description | Recurrent Steps ($T$) | Connectivity Topology | Primary Role |
| :--- | :--- | :--- | :--- | :--- |
| **Condition A** | Base Pretrained LLM | 0 | None (Bypassed) | Control Baseline |
| **Condition B** | Base LLM + One-Pass Graph | 1 | Dynamic ($Q K^T$) | Feedforward Graph Baseline |
| **Condition C** | Recurrent Dynamic Graph Workspace | $T > 1$ (e.g. 3) | Dynamic ($Q K^T$, top-$k$) | Core Proposed Architecture |
| **Condition D** | Recurrent Static Graph Workspace | $T > 1$ (e.g. 3) | Static Learned Adjacency | Topology Ablation |

In addition, a non-neural search baseline (Breadth-First Search / A*) is evaluated under identical deterministic environment rules to establish an upper-bound performance anchor.
