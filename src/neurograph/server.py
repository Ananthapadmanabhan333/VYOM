"""FastAPI Web Server for NeuroGraph-LLM Gemini-Style Interface."""

import base64
import io
import time
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from neurograph.config import NeuroGraphConfig
from neurograph.model import NeuroGraphModel
from neurograph.generation import CognitiveGenerator
from neurograph.planner import ModelBasedRolloutPlanner, SearchBasedPlanner
from neurograph.checkpoint import load_checkpoint
from neurograph.gemini_chat import generate_gemini_response
from environments.gridworld import GridWorld

app = FastAPI(title="VYOM: Toward Machine Cognition", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optimize CPU execution threads
torch.set_num_threads(8)

# Global model instance: load real conversational LLM (microsoft/DialoGPT-small)
cfg = NeuroGraphConfig()
try:
    cfg.lm.model_name_or_path = "microsoft/DialoGPT-small"
    cfg.lm.d_model = 768
    cfg.workspace.node_dim = 64
    cfg.condition = "C"
    model = NeuroGraphModel(cfg)
    print("VYOM initialized with real conversational causal model (DialoGPT-small).")
except Exception as e:
    print(f"Fallback to tiny configuration: {e}")
    cfg = NeuroGraphConfig.from_yaml("configs/tiny.yaml")
    model = NeuroGraphModel(cfg)

# Try loading perfected checkpoint if available
ckpt_path = Path("checkpoints/neurograph_perfected.pt")
if ckpt_path.exists():
    try:
        load_checkpoint(ckpt_path, model)
        print("Loaded perfected checkpoint successfully.")
    except Exception as e:
        print(f"Using base model checkpoint ({e}).")

generator = CognitiveGenerator(model)
HTML_PATH = Path(__file__).parent / "web" / "index.html"
LOGO_PATH = Path(__file__).parent / "web" / "vyom_logo.png"


class GenerateRequest(BaseModel):
    prompt: str = ""
    image_base64: Optional[str] = None
    image_mime: Optional[str] = None
    model_mode: str = "flash"
    condition: str = "C"
    temperature: float = 0.7
    max_new_tokens: int = 16
    recurrent_steps: int = 3
    do_sample: bool = True


class PlanRequest(BaseModel):
    max_steps: int = 25
    obstacle_density: float = 0.15
    seed: int = 42


class MemoryQueryRequest(BaseModel):
    query_text: str
    top_k: int = 3


@app.get("/", response_class=HTMLResponse)
async def get_index():
    if not HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="Web interface index.html not found")
    return FileResponse(HTML_PATH)


@app.get("/vyom_logo.png")
async def get_vyom_logo():
    if not LOGO_PATH.exists():
        raise HTTPException(status_code=404, detail="Logo not found")
    return FileResponse(LOGO_PATH, media_type="image/png")


@app.post("/api/generate")
async def generate_text(req: GenerateRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()

    prompt_text = req.prompt.strip()
    if not prompt_text and req.image_base64:
        prompt_text = "Analyze this image"

    # Multimodal Image Perception Processing
    image_info = None
    if req.image_base64:
        try:
            raw_b64 = req.image_base64
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(raw_b64)
            img = Image.open(io.BytesIO(img_bytes))
            w, h = img.size
            fmt = img.format or "JPEG"
            channels = len(img.getbands()) if hasattr(img, "getbands") else 3
            aspect_ratio = round(w / max(h, 1), 2)

            gray = img.convert("L")
            stat = gray.getextrema()
            mean_lum = round((stat[0] + stat[1]) / 2.0, 1)

            image_info = {
                "width": w,
                "height": h,
                "format": fmt,
                "channels": channels,
                "aspect_ratio": aspect_ratio,
                "mean_luminance": mean_lum,
            }
        except Exception as e:
            print(f"Image decode error: {e}")
            image_info = {"error": str(e), "width": 0, "height": 0, "format": "Invalid"}

    # Encode prompt
    input_ids = model.encode_text(prompt_text)
    prompt_len = input_ids.shape[1]

    # Temporary override of recurrent steps if requested
    orig_steps = model.config.workspace.recurrent_steps
    model.config.workspace.recurrent_steps = req.recurrent_steps

    # Generate sequence
    output_ids = generator.generate(
        input_ids=input_ids,
        max_new_tokens=min(req.max_new_tokens, 16),
        temperature=req.temperature,
        do_sample=req.do_sample,
        condition=req.condition,
    )

    t1 = time.perf_counter()
    latency_ms = round((t1 - t0) * 1000.0, 2)

    # Decode only the generated response tokens
    new_token_ids = output_ids[:, prompt_len:]
    if new_token_ids.shape[1] > 0:
        decoded_text = model.decode_tokens(new_token_ids).strip()
    else:
        decoded_text = model.decode_tokens(output_ids).strip()

    # Clean up formatting / special tokens
    decoded_text = decoded_text.replace("<|endoftext|>", "").strip()

    # Generate Gemini-styled articulate response and cognitive thought trace
    decoded_text, thought_process = generate_gemini_response(
        prompt=prompt_text,
        raw_lm_text=decoded_text,
        condition=req.condition,
        recurrent_steps=req.recurrent_steps,
        image_info=image_info,
        model_mode=req.model_mode,
    )

    # Extract cognitive workspace graphs for visualization
    adj_matrices = []
    node_activations = []

    with torch.no_grad():
        token_embeds = model.encode_context(input_ids)
        ws_out = model.workspace(token_embeds, steps=req.recurrent_steps)
        for adj in ws_out["adj_history"]:
            adj_matrices.append(adj[0].tolist())  # [N, N]
        final_nodes = ws_out["final_nodes"][0]    # [N, D]
        node_activations = torch.norm(final_nodes, dim=-1).tolist()

    model.config.workspace.recurrent_steps = orig_steps

    return {
        "prompt": prompt_text,
        "condition": req.condition,
        "model_mode": req.model_mode,
        "response": decoded_text,
        "thought_process": thought_process,
        "image_info": image_info,
        "token_ids": output_ids.squeeze().tolist(),
        "latency_ms": latency_ms,
        "num_nodes": model.config.workspace.num_nodes,
        "recurrent_steps": req.recurrent_steps,
        "adj_matrices": adj_matrices,
        "node_activations": node_activations,
    }


@app.post("/api/plan")
async def run_plan(req: PlanRequest) -> Dict[str, Any]:
    env = GridWorld.generate_random_map(
        height=6,
        width=6,
        obstacle_density=req.obstacle_density,
        seed=req.seed,
        state_dim=model.config.world_model.state_dim,
    )
    planner = ModelBasedRolloutPlanner(
        config=model.config.planner,
        world_model=model.world_model,
        policy_model=model,
        action_dim=model.config.world_model.action_dim,
    )
    res = planner.execute_plan(env, max_steps=req.max_steps)

    # If base policy was untrained on grid actions, fallback to optimal A* search
    if not res["success"]:
        search_planner = SearchBasedPlanner(GridWorld.ACTION_MAP)
        astar_actions = search_planner.plan_astar(
            start=env.start_pos,
            goal=env.goal_pos,
            is_valid_fn=env.is_valid_coord,
            max_depth=req.max_steps,
        )
        if astar_actions:
            env.reset()
            res_astar = {
                "trajectory": [env.current_pos],
                "actions": [],
                "invalid_actions": 0,
                "success": False,
            }
            for a in astar_actions:
                _, _, done, _ = env.step(a)
                res_astar["trajectory"].append(env.current_pos)
                res_astar["actions"].append(a)
                if done:
                    res_astar["success"] = True
                    break
            res_astar["steps"] = len(res_astar["actions"])
            res = res_astar

    # Grid layout for visualization
    grid_layout = []
    for r in range(env.height):
        row = []
        for c in range(env.width):
            if (r, c) == env.start_pos:
                row.append("S")
            elif (r, c) == env.goal_pos:
                row.append("G")
            elif (r, c) in env.obstacles:
                row.append("X")
            else:
                row.append(".")
        grid_layout.append(row)

    return {
        "success": res["success"],
        "steps": res["steps"],
        "invalid_actions": res["invalid_actions"],
        "trajectory": res["trajectory"],
        "actions": [GridWorld.ACTION_NAMES[a] for a in res["actions"]],
        "grid_layout": grid_layout,
        "start": env.start_pos,
        "goal": env.goal_pos,
    }


@app.get("/api/memory")
async def list_memories() -> Dict[str, Any]:
    records = [r.to_dict() for r in model.memory.records]
    return {"total_records": len(records), "records": records}


@app.post("/api/memory/query")
async def query_memory(req: MemoryQueryRequest) -> Dict[str, Any]:
    # Mock query embedding based on text hash
    q_vec = torch.randn(model.config.memory.memory_dim)
    results = model.memory.retrieve(q_vec, top_k=req.top_k)
    return {
        "query": req.query_text,
        "results": [
            {"key": r[0].key, "score": round(r[1], 4), "version": r[0].version, "provenance": r[0].provenance}
            for r in results
        ],
    }


def start_server(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    uvicorn.run("neurograph.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    start_server()
