from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import math
from typing import Optional
from urllib.parse import urlencode

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from matplotlib import colors as mcolors
from PIL import Image

from a_star_visualization import MazeAStarVisualizer

VIEW_MODES = {"animation", "result"}
DEFAULT_VIEW_MODE = "animation"
ANIMATION_SPEED_MIN = 0.25
ANIMATION_SPEED_MAX = 4.0
DEFAULT_ANIMATION_SPEED = 1.0
BASE_FRAME_DURATION_MS = 40
GIF_MIN_FRAME_DURATION_MS = 20

GridPos = tuple[int, int]

app = FastAPI(title="A* Algorithm Web Visualizer")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@dataclass(eq=False)
class CustomMapSpec:
    width: int
    height: int
    walls: np.ndarray
    serialized: str


def _normalize_view_mode(view_mode: str) -> str:
    if view_mode in VIEW_MODES:
        return view_mode
    return DEFAULT_VIEW_MODE


def _normalize_animation_speed(animation_speed: float) -> float:
    return max(ANIMATION_SPEED_MIN, min(ANIMATION_SPEED_MAX, animation_speed))


def _animation_timing(animation_speed: float) -> tuple[int, int]:
    normalized_speed = _normalize_animation_speed(animation_speed)
    max_direct_speed = BASE_FRAME_DURATION_MS / GIF_MIN_FRAME_DURATION_MS

    if normalized_speed <= max_direct_speed:
        frame_duration_ms = max(
            GIF_MIN_FRAME_DURATION_MS,
            int(round(BASE_FRAME_DURATION_MS / normalized_speed)),
        )
        return frame_duration_ms, 1

    frame_stride = max(1, math.ceil(normalized_speed / max_direct_speed))
    frame_duration_ms = max(
        GIF_MIN_FRAME_DURATION_MS,
        int(round((BASE_FRAME_DURATION_MS * frame_stride) / normalized_speed)),
    )
    return frame_duration_ms, frame_stride


def _parse_seed(seed: str) -> Optional[int]:
    seed_value = seed.strip()
    if not seed_value:
        return None
    try:
        return int(seed_value)
    except ValueError as exc:
        raise ValueError("Seed must be an integer.") from exc


def _parse_optional_coordinate(raw_value: str, field_name: str) -> Optional[int]:
    value = raw_value.strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be 0 or greater.")
    return parsed


def _normalize_start_goal(
    start_row: Optional[int],
    start_col: Optional[int],
    goal_row: Optional[int],
    goal_col: Optional[int],
) -> Optional[tuple[GridPos, GridPos]]:
    coordinates = (start_row, start_col, goal_row, goal_col)
    if any(value is None for value in coordinates):
        return None

    start = (start_row, start_col)
    goal = (goal_row, goal_col)
    if start == goal:
        raise ValueError("Start and goal positions must be different.")
    return start, goal


def _parse_start_goal_from_form(
    start_row: str,
    start_col: str,
    goal_row: str,
    goal_col: str,
) -> Optional[tuple[GridPos, GridPos]]:
    parsed_start_row = _parse_optional_coordinate(start_row, "Start row")
    parsed_start_col = _parse_optional_coordinate(start_col, "Start col")
    parsed_goal_row = _parse_optional_coordinate(goal_row, "Goal row")
    parsed_goal_col = _parse_optional_coordinate(goal_col, "Goal col")
    return _normalize_start_goal(
        start_row=parsed_start_row,
        start_col=parsed_start_col,
        goal_row=parsed_goal_row,
        goal_col=parsed_goal_col,
    )


def _parse_start_goal_from_query(
    start_row: Optional[int],
    start_col: Optional[int],
    goal_row: Optional[int],
    goal_col: Optional[int],
) -> Optional[tuple[GridPos, GridPos]]:
    for field_name, value in (
        ("start_row", start_row),
        ("start_col", start_col),
        ("goal_row", goal_row),
        ("goal_col", goal_col),
    ):
        if value is not None and value < 0:
            raise ValueError(f"`{field_name}` must be 0 or greater.")

    return _normalize_start_goal(
        start_row=start_row,
        start_col=start_col,
        goal_row=goal_row,
        goal_col=goal_col,
    )


def _parse_dimension(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"`{name}` must be an integer.")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"`{name}` must be an integer.")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError(f"`{name}` must be an integer.")

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` must be an integer.") from exc

    if parsed <= 0:
        raise ValueError(f"`{name}` must be 1 or greater.")
    return parsed


def _parse_wall_value(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("Walls must use only 0 (free) or 1 (wall).")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("Walls must use only 0 (free) or 1 (wall).")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("Walls must use only 0 (free) or 1 (wall).")

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Walls must use only 0 (free) or 1 (wall).") from exc

    if parsed not in (0, 1):
        raise ValueError("Walls must use only 0 (free) or 1 (wall).")
    return parsed


def _parse_custom_map(custom_map_raw: str) -> Optional[CustomMapSpec]:
    custom_map_text = custom_map_raw.strip()
    if not custom_map_text:
        return None

    try:
        payload = json.loads(custom_map_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Custom map must be valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Custom map must be a JSON object.")

    raw_width = payload.get("n", payload.get("width"))
    raw_height = payload.get("k", payload.get("height"))

    if raw_width is None:
        raise ValueError("Custom map must include `n` (width).")
    if raw_height is None:
        raise ValueError("Custom map must include `k` (height).")

    width = _parse_dimension(raw_width, "n")
    height = _parse_dimension(raw_height, "k")

    if "walls" not in payload:
        raise ValueError("Custom map must include `walls`.")

    raw_walls = payload["walls"]
    if not isinstance(raw_walls, list):
        raise ValueError("`walls` must be an array.")

    contains_row_arrays = any(isinstance(row, list) for row in raw_walls)
    if contains_row_arrays and not all(isinstance(row, list) for row in raw_walls):
        raise ValueError("`walls` must be either a 1D or 2D array consistently.")

    if contains_row_arrays:
        if len(raw_walls) != height:
            raise ValueError(f"`walls` row count must match k({height}).")

        wall_rows: list[list[int]] = []
        for row in raw_walls:
            if len(row) != width:
                raise ValueError(f"Each `walls` row length must match n({width}).")
            wall_rows.append([_parse_wall_value(cell) for cell in row])

        wall_grid = np.array(wall_rows, dtype=np.int8)
    else:
        expected_length = width * height
        if len(raw_walls) != expected_length:
            raise ValueError(f"1D `walls` length must be n*k({expected_length}).")

        flat_values = [_parse_wall_value(cell) for cell in raw_walls]
        wall_grid = np.array(flat_values, dtype=np.int8).reshape((height, width))

    serialized = json.dumps(
        {"n": width, "k": height, "walls": wall_grid.tolist()},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return CustomMapSpec(width=width, height=height, walls=wall_grid, serialized=serialized)


def _build_preview_url(
    size: int,
    wall_prob: float,
    seed: Optional[int],
    view_mode: str,
    animation_speed: float,
    custom_map: str = "",
    start_goal: Optional[tuple[GridPos, GridPos]] = None,
) -> str:
    params: dict[str, int | float | str] = {
        "size": size,
        "wall_prob": wall_prob,
        "view_mode": _normalize_view_mode(view_mode),
        "animation_speed": _normalize_animation_speed(animation_speed),
    }

    if seed is not None:
        params["seed"] = seed
    if custom_map:
        params["custom_map"] = custom_map
    if start_goal is not None:
        (start_row, start_col), (goal_row, goal_col) = start_goal
        params["start_row"] = start_row
        params["start_col"] = start_col
        params["goal_row"] = goal_row
        params["goal_col"] = goal_col

    return f"/preview?{urlencode(params)}"


def _validate_start_goal_for_request(
    start_goal: Optional[tuple[GridPos, GridPos]],
    size: int,
    custom_map: Optional[CustomMapSpec],
) -> None:
    if start_goal is None:
        return

    if custom_map is None:
        width = int(size)
        height = int(size)
    else:
        width = custom_map.width
        height = custom_map.height

    (start_row, start_col), (goal_row, goal_col) = start_goal
    for label, row, col in (
        ("start", start_row, start_col),
        ("goal", goal_row, goal_col),
    ):
        if row >= height or col >= width:
            raise ValueError(f"{label} position {(row, col)} is out of bounds for grid size {height}x{width}.")

    if custom_map is not None:
        if custom_map.walls[start_row, start_col] == 1:
            raise ValueError("Start must be on a free cell (0) in the current custom map.")
        if custom_map.walls[goal_row, goal_col] == 1:
            raise ValueError("Goal must be on a free cell (0) in the current custom map.")


def _frame_scale(width: int, height: int) -> int:
    return max(2, min(16, 640 // max(1, width, height)))


def _build_visualizer(
    size: int,
    wall_prob: float,
    seed: Optional[int],
    custom_map: Optional[CustomMapSpec],
    start_goal: Optional[tuple[GridPos, GridPos]],
) -> MazeAStarVisualizer:
    start_goal_kwargs: dict[str, GridPos] = {}
    if start_goal is not None:
        start_goal_kwargs["start"], start_goal_kwargs["goal"] = start_goal

    if custom_map is None:
        return MazeAStarVisualizer(
            size=size,
            wall_probability=wall_prob,
            seed=seed,
            **start_goal_kwargs,
        )

    return MazeAStarVisualizer(
        width=custom_map.width,
        height=custom_map.height,
        wall_grid=custom_map.walls,
        seed=seed,
        **start_goal_kwargs,
    )


def _render_animation_gif(
    size: int,
    wall_prob: float,
    seed: Optional[int],
    frame_duration_ms: int,
    frame_stride: int = 1,
    custom_map: Optional[CustomMapSpec] = None,
    start_goal: Optional[tuple[GridPos, GridPos]] = None,
) -> bytes:
    visualizer = _build_visualizer(
        size=size,
        wall_prob=wall_prob,
        seed=seed,
        custom_map=custom_map,
        start_goal=start_goal,
    )

    frames = list(visualizer.a_star_steps())
    sampled_frames = frames[::frame_stride]
    if (len(frames) - 1) % frame_stride != 0:
        sampled_frames.append(frames[-1])

    palette = np.array(
        [
            tuple(int(channel * 255) for channel in mcolors.to_rgb(color))
            for color in visualizer._create_colormap().colors
        ],
        dtype=np.uint8,
    )
    scale = _frame_scale(width=visualizer.width, height=visualizer.height)

    images: list[Image.Image] = []
    for frame in sampled_frames:
        rgb_frame = palette[frame]
        frame_image = Image.fromarray(rgb_frame, mode="RGB")
        if scale > 1:
            frame_image = frame_image.resize(
                (frame_image.width * scale, frame_image.height * scale),
                Image.Resampling.NEAREST,
            )
        images.append(frame_image)

    buf = BytesIO()
    images[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )
    buf.seek(0)
    return buf.getvalue()


def _render_final_frame_png(
    size: int,
    wall_prob: float,
    seed: Optional[int],
    custom_map: Optional[CustomMapSpec] = None,
    start_goal: Optional[tuple[GridPos, GridPos]] = None,
) -> bytes:
    visualizer = _build_visualizer(
        size=size,
        wall_prob=wall_prob,
        seed=seed,
        custom_map=custom_map,
        start_goal=start_goal,
    )

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


def _page_context(
    request: Request,
    size: int,
    wall_prob: float,
    seed: str,
    start_row: str,
    start_col: str,
    goal_row: str,
    goal_col: str,
    view_mode: str,
    animation_speed: float,
    preview_url: str,
    custom_map: str = "",
    error_message: str = "",
) -> dict[str, object]:
    return {
        "request": request,
        "size": size,
        "wall_prob": wall_prob,
        "seed": seed,
        "start_row": start_row,
        "start_col": start_col,
        "goal_row": goal_row,
        "goal_col": goal_col,
        "view_mode": view_mode,
        "animation_speed": animation_speed,
        "preview_url": preview_url,
        "custom_map": custom_map,
        "error_message": error_message,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    preview_url = _build_preview_url(
        size=50,
        wall_prob=0.28,
        seed=None,
        view_mode=DEFAULT_VIEW_MODE,
        animation_speed=DEFAULT_ANIMATION_SPEED,
    )

    return templates.TemplateResponse(
        "index.html",
        _page_context(
            request=request,
            size=50,
            wall_prob=0.28,
            seed="",
            start_row="",
            start_col="",
            goal_row="",
            goal_col="",
            view_mode=DEFAULT_VIEW_MODE,
            animation_speed=DEFAULT_ANIMATION_SPEED,
            preview_url=preview_url,
        ),
    )


@app.post("/", response_class=HTMLResponse)
def generate(
    request: Request,
    size: int = Form(50),
    wall_prob: float = Form(0.28),
    seed: str = Form(""),
    start_row: str = Form(""),
    start_col: str = Form(""),
    goal_row: str = Form(""),
    goal_col: str = Form(""),
    view_mode: str = Form(DEFAULT_VIEW_MODE),
    animation_speed: float = Form(DEFAULT_ANIMATION_SPEED),
    custom_map: str = Form(""),
):
    parsed_view_mode = _normalize_view_mode(view_mode)
    parsed_animation_speed = _normalize_animation_speed(animation_speed)

    error_message = ""
    parsed_seed: Optional[int] = None
    parsed_custom_map: Optional[CustomMapSpec] = None
    parsed_start_goal: Optional[tuple[GridPos, GridPos]] = None

    try:
        parsed_seed = _parse_seed(seed)
    except ValueError as exc:
        error_message = str(exc)

    if not error_message:
        try:
            parsed_custom_map = _parse_custom_map(custom_map)
        except ValueError as exc:
            error_message = str(exc)

    if not error_message:
        try:
            parsed_start_goal = _parse_start_goal_from_form(
                start_row=start_row,
                start_col=start_col,
                goal_row=goal_row,
                goal_col=goal_col,
            )
        except ValueError as exc:
            error_message = str(exc)

    if not error_message:
        try:
            _validate_start_goal_for_request(
                start_goal=parsed_start_goal,
                size=size,
                custom_map=parsed_custom_map,
            )
        except ValueError as exc:
            error_message = str(exc)

    preview_url = _build_preview_url(
        size=size,
        wall_prob=wall_prob,
        seed=parsed_seed,
        view_mode=parsed_view_mode,
        animation_speed=parsed_animation_speed,
        custom_map=parsed_custom_map.serialized if parsed_custom_map else "",
        start_goal=parsed_start_goal,
    )

    return templates.TemplateResponse(
        "index.html",
        _page_context(
            request=request,
            size=size,
            wall_prob=wall_prob,
            seed=seed,
            start_row=start_row,
            start_col=start_col,
            goal_row=goal_row,
            goal_col=goal_col,
            view_mode=parsed_view_mode,
            animation_speed=parsed_animation_speed,
            preview_url=preview_url,
            custom_map=custom_map,
            error_message=error_message,
        ),
    )


@app.get("/preview")
def preview(
    size: int = 50,
    wall_prob: float = 0.28,
    seed: Optional[int] = None,
    start_row: Optional[int] = None,
    start_col: Optional[int] = None,
    goal_row: Optional[int] = None,
    goal_col: Optional[int] = None,
    view_mode: str = DEFAULT_VIEW_MODE,
    animation_speed: float = DEFAULT_ANIMATION_SPEED,
    custom_map: str = "",
):
    try:
        parsed_custom_map = _parse_custom_map(custom_map)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        parsed_start_goal = _parse_start_goal_from_query(
            start_row=start_row,
            start_col=start_col,
            goal_row=goal_row,
            goal_col=goal_col,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    parsed_view_mode = _normalize_view_mode(view_mode)

    try:
        if parsed_view_mode == "animation":
            frame_duration_ms, frame_stride = _animation_timing(animation_speed)
            image_bytes = _render_animation_gif(
                size=size,
                wall_prob=wall_prob,
                seed=seed,
                frame_duration_ms=frame_duration_ms,
                frame_stride=frame_stride,
                custom_map=parsed_custom_map,
                start_goal=parsed_start_goal,
            )
            media_type = "image/gif"
        else:
            image_bytes = _render_final_frame_png(
                size=size,
                wall_prob=wall_prob,
                seed=seed,
                custom_map=parsed_custom_map,
                start_goal=parsed_start_goal,
            )
            media_type = "image/png"
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StreamingResponse(
        BytesIO(image_bytes),
        media_type=media_type,
        headers={"Cache-Control": "no-store"},
    )
