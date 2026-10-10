# Empirical Experiments & Benchmark Results

This document records the empirical results obtained from the automated benchmarks executed on the prototype architecture.

## 1. Experimental Setup

- **Hardware:** Local CPU execution environment (PyTorch 2.12.1+cpu).
- **Test Task:** Synthetic 6x6 GridWorld navigation across 10 held-out maps generated with obstacle density 0.15 under fixed random seeds (500-509).
- **Evaluation Metrics:**
  - **Goal Completion %:** Fraction of episodes where the agent successfully reaches the goal coordinate.
  - **Invalid Action %:** Fraction of executed actions that collided with obstacles or grid boundaries.
  - **Average Steps:** Number of environment steps taken per episode (max cap = 25).
  - **Train Loss:** Multi-objective composite loss after 25 optimization steps.
  - **Parameter Count:** Total number of model parameters.
  - **Latency:** Average inference forward-pass latency in milliseconds.

---

## 2. Benchmark Comparison Results

The following table reflects actual executed runs from `benchmarks.compare_baselines`:

| Condition | Goal Comp % | Invalid Act % | Avg Steps | Train Loss | Params | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Condition A (Base LLM)** | 0.0% | 0.0% | 25.00 | 7.0908 | 371,526 | 0.88 |
| **Condition B (One-pass Graph)** | 0.0% | 0.0% | 25.00 | 7.7318 | 379,654 | 1.84 |
| **Condition C (Recurrent Dynamic Graph)** | 0.0% | 0.0% | 25.00 | 7.9785 | 379,654 | 2.95 |
| **Condition D (Recurrent Static Graph)** | 0.0% | 0.0% | 25.00 | 7.8924 | 371,526 | 3.54 |
| **Conventional Search (BFS)** | 100.0% | 0.0% | 10.00 | 0.0000 | 0 | 0.15 |

---

## 3. Scientific Analysis & Discussion

1. **Comparison with Algorithmic Search:**
   Under minimal training (25 steps), all neural models fail to solve complex obstacle navigation on held-out maps (0.0% goal completion, timing out at step 25). By contrast, conventional Breadth-First Search achieves a 100% completion rate in an average of 10.0 steps with 0.15 ms latency.
   *Takeaway:* Neural cognitive workspaces cannot replace exact combinatorial search algorithms without extensive pre-training or fine-tuning on diverse pathfinding demonstrations.

2. **Latency & Compute Overhead:**
   Moving from Condition A (Base LLM) to Condition C (Recurrent Dynamic Graph) introduces an additional 8,128 parameters (+2.2%) and increases inference latency from 0.88 ms to 2.95 ms (+235%), reflecting the iterative message-passing across recurrent steps $T=3$.

3. **World Model Transition Learning:**
   In independent transition evaluation (`test_world_model_training_and_uncertainty`), the world-state predictor converged reliably, reducing transition MSE below 0.05 on held-out transitions and producing well-calibrated epistemic uncertainty estimates.
