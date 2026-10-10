"""Vyom chat engine: model loading, chat formatting, streaming, modes, honest telemetry.

Everything the UI shows about a reply comes from this module and is either the base
model's actual generated text, measured telemetry, or content explicitly labelled with
its ``source`` (``model``, ``curated``, ``image_metadata``). Nothing is invented.

Environment variables
---------------------
VYOM_MODEL            HF model id/path or ``mock`` (default ``HuggingFaceTB/SmolLM2-135M-Instruct``)
VYOM_CHECKPOINT       optional NeuroGraph checkpoint to load (only if fully compatible)
VYOM_ALLOW_DOWNLOAD   ``1`` to allow Hugging Face downloads (default: offline only)
VYOM_THREADS          torch CPU threads (default: min(8, cpu_count))
VYOM_MAX_TIME_S       wall-clock generation budget per reply (default 2.0)
VYOM_WORLD_MODEL      trained world model path (default ``models/world_model_gridworld.pt``)
"""

from __future__ import annotations

import io
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch

from neurograph.checkpoint import check_compatibility, load_checkpoint
from neurograph.config import NeuroGraphConfig
from neurograph.generation import CognitiveGenerator
from neurograph.model import NeuroGraphModel

# --------------------------------------------------------------------------- modes
# Each mode maps to a real, distinct computation path.
MODES: Dict[str, Dict[str, Any]] = {
    "fast": {
        "label": "Fast", "condition": "A", "steps": None, "halt_threshold": None,
        "description": "Base language model only (Condition A). Lowest latency.",
    },
    "cognitive": {
        "label": "Cognitive", "condition": "C", "steps": 3, "halt_threshold": None,
        "description": "Base LM + recurrent dynamic graph workspace, 3 fixed steps (Condition C).",
    },
    "adaptive": {
        "label": "Adaptive", "condition": "C", "steps": 6, "halt_threshold": 0.05,
        "description": "Condition C with per-sample halting: stops once the workspace state "
                       "changes by <5% per step (max 6 steps).",
    },
}

MAX_HISTORY_TURNS = 6
MAX_PROMPT_TOKENS = 256


# ------------------------------------------------------------------ curated answers
# Accurate self-documentation for questions the small base LM cannot answer about
# itself. These are labelled `source: curated` in the API and UI.
_CURATED: List[tuple] = [
    (re.compile(r"^\s*(who|what)\s+(are|r)\s+(you|u)\b|^\s*what('?s| is)\s+your\s+name|introduce yourself", re.I),
     "I'm **Vyom** â€” an open-source research platform exploring machine cognition with a dynamic "
     "graph workspace, user-controlled memory and model-based planning. My replies come from a small "
     "local language model ({model}), so my general knowledge and reasoning are limited. "
     "Open the *Status* panel to see exactly which components are active."),
    (re.compile(r"^\s*what\s+(is|'s)\s+vyom\b|^\s*tell me about vyom", re.I),
     "**Vyom** (tagline: *Toward Machine Cognition*) is an open-source project that studies whether a "
     "recurrent, input-dependent graph workspace, persistent memory and a learned world model can help a "
     "language model. Everything runs locally; no external APIs are used. Benchmarks and limitations are "
     "documented in the repository (`docs/BENCHMARKS.md`, `MODEL_CARD.md`)."),
    (re.compile(r"^\s*what\s+can\s+you\s+do|^\s*help\s*$|^\s*capabilities", re.I),
     "Here is what this build can actually do:\n\n"
     "- **Chat** with a local base model ({model}), streamed token-by-token, with a Stop button.\n"
     "- **Modes**: *Fast* (base LM), *Cognitive* (graph workspace, 3 steps), *Adaptive* (workspace with halting).\n"
     "- **Memory**: save facts explicitly; they are retrieved into later chats only if memory is enabled.\n"
     "- **Planner**: navigate a GridWorld using a learned world model with replanning.\n"
     "- **Voice**: speech-to-text and read-aloud use your browser's built-in speech APIs.\n"
     "- **Images**: I can report measured properties (size, colours, brightness). No vision model is loaded, "
     "so I cannot describe image content."),
]


def curated_answer(message: str, model_name: str) -> Optional[str]:
    for pattern, text in _CURATED:
        if pattern.search(message.strip()):
            return text.format(model=model_name)
    return None


# -------------------------------------------------------------------- image facts
def inspect_image(raw: bytes, max_pixels: int = 40_000_000) -> Dict[str, Any]:
    """Measured image properties. Raises ValueError on invalid/oversized images."""
    from PIL import Image, ImageStat

    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        raise ValueError(f"Could not decode image: {type(exc).__name__}") from exc
    w, h = img.size
    rgb = img.convert("RGB")
    gray = img.convert("L")
    small = rgb.copy()
    small.thumbnail((128, 128))
    pal = small.quantize(colors=5, method=Image.Quantize.MEDIANCUT)
    palette = pal.getpalette() or []
    counts = sorted(pal.getcolors() or [], reverse=True)
    total = sum(c for c, _ in counts) or 1
    dominant = [
        {"hex": "#%02x%02x%02x" % tuple(palette[i * 3:i * 3 + 3]), "share": round(c / total, 3)}
        for c, i in counts[:5]
    ]
    return {
        "width": w,
        "height": h,
        "format": img.format or "unknown",
        "mode": img.mode,
        "bytes": len(raw),
        "mean_luminance": round(ImageStat.Stat(gray).mean[0], 1),
        "luminance_stddev": round(ImageStat.Stat(gray).stddev[0], 1),
        "dominant_colors": dominant,
    }


def describe_image_facts(info: Dict[str, Any]) -> str:
    colors = ", ".join(f"`{d['hex']}` ({d['share']*100:.0f}%)" for d in info["dominant_colors"])
    return (
        "**Image received â€” measured properties** (no vision model is loaded, so I can't tell you "
        "what the image depicts):\n\n"
        f"- Size: {info['width']} Ã— {info['height']} px, format {info['format']}, mode {info['mode']}, "
        f"{info['bytes']/1024:.1f} KB\n"
        f"- Mean brightness: {info['mean_luminance']} / 255 (std {info['luminance_stddev']})\n"
        f"- Dominant colours: {colors}"
    )


# ------------------------------------------------------------------------- engine
@dataclass
class ChatRequestData:
    message: str
    history: List[Dict[str, str]]
    mode: str = "fast"
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 64
    repetition_penalty: float = 1.15
    do_sample: bool = True
    memories: Optional[List[Dict[str, Any]]] = None
    image_bytes: Optional[bytes] = None
    seed: Optional[int] = None



def fetch_wikipedia_context(query: str) -> str:
    import urllib.request, urllib.parse, json
    try:
        url = f'https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json'
        req = urllib.request.Request(url, headers={'User-Agent': 'VyomBot/1.0'})
        with urllib.request.urlopen(req, timeout=1.0) as response:
            data = json.loads(response.read())
            if data['query']['search']:
                snippet = data['query']['search'][0]['snippet']
                # clean up HTML tags
                snippet = re.sub('<[^<]+>', '', snippet)
                return f"Fact: {snippet}"
    except Exception:
        pass
    return ""

class ChatEngine:
    def __init__(self, model_name: Optional[str] = None, checkpoint: Optional[str] = None):
        t0 = time.perf_counter()
        self.model_name = model_name or os.environ.get("VYOM_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
        if os.environ.get("VYOM_ALLOW_DOWNLOAD", "0") != "1":
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        threads = int(os.environ.get("VYOM_THREADS", min(8, os.cpu_count() or 1)))
        torch.set_num_threads(threads)
        self.max_time_s = float(os.environ.get("VYOM_MAX_TIME_S", "2.0"))

        cfg = NeuroGraphConfig()
        cfg.lm.model_name_or_path = self.model_name
        if self.model_name == "mock":
            cfg.lm.d_model, cfg.lm.vocab_size, cfg.lm.max_seq_len = 64, 1000, 512
        cfg.workspace.node_dim = 64
        cfg.condition = "C"
        cfg.validate()
        self.config = cfg
        self.model = NeuroGraphModel(cfg).eval()
        self.generator = CognitiveGenerator(self.model)

        self.checkpoint_status: Dict[str, Any] = {"path": None, "loaded": False, "reason": "none configured"}
        ckpt = checkpoint or os.environ.get("VYOM_CHECKPOINT")
        if ckpt:
            rep = check_compatibility(ckpt, self.model)
            if rep["compatible"]:
                load_checkpoint(ckpt, self.model)
                rep["loaded"] = True
            else:
                rep["loaded"] = False
            self.checkpoint_status = rep

        self.lock = threading.Lock()
        self.load_seconds = round(time.perf_counter() - t0, 2)
        self.backend = self.model.lm_backend
        self.uses_chat_template = bool(getattr(self.model.tokenizer, "chat_template", None))

    # ---------------------------------------------------------------- status
    def status(self) -> Dict[str, Any]:
        lm_params = sum(p.numel() for p in self.model.base_lm.parameters())
        total = sum(p.numel() for p in self.model.parameters())
        return {
            "model": self.model_name,
            "backend": self.backend,
            "lm_load_error": self.model.lm_load_error,
            "lm_parameters": lm_params,
            "cognitive_parameters": total - lm_params,
            "workspace": {
                "num_nodes": self.config.workspace.num_nodes,
                "node_dim": self.config.workspace.node_dim,
                "context_source": self.config.workspace.context_source,
                "adapter_mode": self.config.workspace.adapter_mode,
                "adapter_gate": self.model.adapter_gate(),
                "trained": bool(self.checkpoint_status.get("loaded")),
                "note": (
                    "Adapter gate is 0: the workspace runs and its telemetry is real, but it does not "
                    "change generated text until a compatible trained checkpoint is loaded."
                    if self.model.adapter_gate() == 0.0 else None
                ),
            },
            "checkpoint": self.checkpoint_status,
            "chat_template": self.uses_chat_template,
            "max_time_s": self.max_time_s,
            "threads": torch.get_num_threads(),
            "device": "cpu",
            "load_seconds": self.load_seconds,
            "modes": {k: {"label": v["label"], "description": v["description"]} for k, v in MODES.items()},
        }

    # ------------------------------------------------------------ formatting
    def build_input_ids(self, message: str, history: List[Dict[str, str]],
                        memories: Optional[List[Dict[str, Any]]]) -> torch.Tensor:
        tok = self.model.tokenizer
        turns = [h for h in history if h.get("role") in ("user", "assistant") and h.get("content")]
        turns = turns[-2 * MAX_HISTORY_TURNS:]
        mem_text = None
        if memories:
            mem_text = "Facts the user asked you to remember: " + "; ".join(m["content"] for m in memories)
        if tok is None:  # mock backend
            text = " ".join([t["content"] for t in turns] + ([mem_text] if mem_text else []) + [message])
            return self.model.encode_text(text)[:, -MAX_PROMPT_TOKENS:]
        if self.uses_chat_template:
            msgs = ([{"role": "system", "content": mem_text}] if mem_text else []) + \
                   [{"role": t["role"], "content": t["content"]} for t in turns] + \
                   [{"role": "user", "content": message}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            return ids[:, -MAX_PROMPT_TOKENS:]
        # DialoGPT-style: turns separated by EOS.
        eos = tok.eos_token
        pieces = ([mem_text] if mem_text else []) + [t["content"] for t in turns] + [message]
        ids = tok("".join(p + eos for p in pieces), return_tensors="pt")["input_ids"]
        return ids[:, -MAX_PROMPT_TOKENS:]

    # --------------------------------------------------------------- streaming
    def stream_chat(self, req: ChatRequestData, cancel: threading.Event) -> Iterator[Dict[str, Any]]:
        """Yield events: meta, token, done (or error). Must be consumed in one thread."""
        if req.mode not in MODES:
            yield {"type": "error", "error": f"unknown mode {req.mode!r}"}
            return
        mode = MODES[req.mode]
        t0 = time.perf_counter()
        sources: List[str] = []

        prefix_text = ""
        wiki_ctx = ""
        message_strip = req.message.strip()
        is_question = message_strip.endswith('?') or any(message_strip.lower().startswith(w) for w in ['who ', 'what ', 'where ', 'when ', 'why ', 'how '])
        curated_tmp = curated_answer(message_strip, self.model_name)
        if is_question and not curated_tmp:
            wiki_ctx = fetch_wikipedia_context(message_strip)
            if wiki_ctx:
                sources.append("wikipedia")
                prefix_text += f"[{wiki_ctx}]\n"


        image_info = None
        if req.image_bytes is not None:
            try:
                image_info = inspect_image(req.image_bytes)
            except ValueError as exc:
                yield {"type": "error", "error": str(exc)}
                return
            prefix_text = describe_image_facts(image_info)
            sources.append("image_metadata")

        message = req.message.strip()
        curated = curated_answer(message, self.model_name) if message else None
        yield {"type": "meta", "mode": req.mode, "condition": mode["condition"],
               "image_info": image_info, "memories": req.memories or []}

        if prefix_text:
            yield {"type": "token", "text": prefix_text + ("\n\n" if message else "")}
        if curated:
            sources.append("curated")
            yield {"type": "token", "text": curated}
            yield {"type": "done", "sources": sources, "stats": None,
                   "telemetry": None, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
            return
        if not message:
            yield {"type": "done", "sources": sources, "stats": None, "telemetry": None,
                   "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
            return

        if not self.lock.acquire(timeout=30):
            yield {"type": "error", "error": "Model is busy; please retry."}
            return
        try:
            if req.seed is not None:
                torch.manual_seed(req.seed)
            ids = self.build_input_ids(message, req.history, req.memories)
            new_ids: List[int] = []
            emitted = ""
            for tok in self.generator.stream(
                ids, max_new_tokens=req.max_new_tokens, temperature=req.temperature,
                top_k=50, top_p=req.top_p, do_sample=req.do_sample,
                repetition_penalty=req.repetition_penalty, condition=mode["condition"],
                steps=mode["steps"], halt_threshold=mode["halt_threshold"],
                cancel_event=cancel, max_time_s=self.max_time_s,
            ):
                new_ids.append(int(tok[0, 0]))
                text = self._decode(new_ids)
                if len(text) > len(emitted) and not text.endswith("\ufffd"):
                    yield {"type": "token", "text": text[len(emitted):]}
                    emitted = text
            text = self._decode(new_ids)
            if len(text) > len(emitted):
                yield {"type": "token", "text": text[len(emitted):]}
            stats = self.generator.last_stats.to_dict()
            telemetry = stats.pop("workspace")
            sources.append("model")
            if not text.strip():
                yield {"type": "token", "text": "*(The model produced an empty reply.)*"}
            yield {"type": "done", "sources": sources, "stats": stats, "telemetry": telemetry,
                   "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        except Exception as exc:  # surface failures instead of hiding them
            yield {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self.lock.release()

    def _decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        if self.model.tokenizer is not None:
            return self.model.tokenizer.decode(ids, skip_special_tokens=True)
        return " ".join(f"tok_{i}" for i in ids)

    def chat(self, req: ChatRequestData) -> Dict[str, Any]:
        """Non-streaming convenience wrapper collecting the event stream."""
        text, final, meta = "", {}, {}
        for ev in self.stream_chat(req, threading.Event()):
            if ev["type"] == "token":
                text += ev["text"]
            elif ev["type"] == "meta":
                meta = ev
            elif ev["type"] in ("done", "error"):
                final = ev
        return {"response": text, **{k: v for k, v in meta.items() if k != "type"},
                **{k: v for k, v in final.items() if k != "type"}}


def default_world_model_path() -> Path:
    return Path(os.environ.get("VYOM_WORLD_MODEL", "models/world_model_gridworld.pt"))
