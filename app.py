from __future__ import annotations

from io import BytesIO
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from a_star_visualization import MazeAStarVisualizer

matplotlib.use("Agg")

app = FastAPI(title="A* Algorithm Web Visualizer")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def _render_final_frame_png(size: int, wall_prob: float, seed: Optional[int]) -> bytes:
    visualizer = MazeAStarVisualizer(size=size, wall_probability=wall_prob, seed=seed)
    frames = list(visualizer.a_star_steps())
    final_frame = frames[-1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("A* Final Path")
    ax.imshow(final_frame, cmap=visualizer._create_colormap(), vmin=0, vmax=7)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "size": 50,
            "wall_prob": 0.28,
            "seed": "",
            "preview_url": "/preview?size=50&wall_prob=0.28",
        },
    )


@app.post("/", response_class=HTMLResponse)
def generate(
    request: Request,
    size: int = Form(50),
    wall_prob: float = Form(0.28),
    seed: str = Form(""),
):
    parsed_seed: Optional[int] = int(seed) if seed.strip() else None
    preview_url = f"/preview?size={size}&wall_prob={wall_prob}"
    if parsed_seed is not None:
        preview_url += f"&seed={parsed_seed}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "size": size,
            "wall_prob": wall_prob,
            "seed": seed,
            "preview_url": preview_url,
        },
    )


@app.get("/preview")
def preview(size: int = 50, wall_prob: float = 0.28, seed: Optional[int] = None):
    image_bytes = _render_final_frame_png(size=size, wall_prob=wall_prob, seed=seed)
    return StreamingResponse(BytesIO(image_bytes), media_type="image/png")
