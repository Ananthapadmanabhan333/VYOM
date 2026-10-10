"""Benchmark metrics computation and reporting utilities."""

from typing import Dict, Any
import time
import torch


def measure_model_efficiency(
    model: torch.nn.Module,
    sample_input_ids: torch.Tensor,
    condition: str = "C",
    num_runs: int = 10,
) -> Dict[str, Any]:
    """Measure parameter counts, inference latency, and peak memory usage."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    device = sample_input_ids.device
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model(sample_input_ids, condition=condition)

    # Latency timing
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(sample_input_ids, condition=condition)
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000.0

    peak_memory_mb = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "avg_latency_ms": round(avg_latency_ms, 3),
        "peak_cuda_memory_mb": round(peak_memory_mb, 2),
    }


def format_comparison_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Format structured results as a clean Markdown table."""
    headers = [
        "Condition",
        "Goal Comp %",
        "Invalid Act %",
        "Avg Steps",
        "Train Loss",
        "Params",
        "Latency (ms)",
    ]

    rows = []
    for cond_name, r in results.items():
        comp = f"{r.get('goal_completion_rate', 0.0) * 100:.1f}%"
        inv = f"{r.get('invalid_action_rate', 0.0) * 100:.1f}%"
        steps = f"{r.get('avg_steps_to_goal', 0.0):.2f}"
        loss = f"{r.get('final_total_loss', 0.0):.4f}"
        params = f"{r.get('total_parameters', 0):,}"
        latency = f"{r.get('avg_latency_ms', 0.0):.2f}"
        rows.append([cond_name, comp, inv, steps, loss, params, latency])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in col_widths) + " |"
    data_lines = ["| " + " | ".join(str(val).ljust(w) for val, w in zip(row, col_widths)) + " |" for row in rows]

    return "\n".join([header_line, sep_line] + data_lines)
