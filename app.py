from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from io import BytesIO
import json
import math
from typing import Optional
from urllib.parse import urlencode
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

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
CUSTOM_MAP_MAX_DIM = 2000
LARGE_MAP_DIM_THRESHOLD = 500
MAX_PREVIEW_CACHE_ENTRIES = 16
RESULT_IMAGE_TARGET_PX = 1600
RESULT_IMAGE_MAX_SCALE = 24
LARGE_MAP_RESULT_NOTICE = "500 이상 맵에서는 애니메이션 대신 최종 결과만 표시합니다."

GridPos = tuple[int, int]

app = FastAPI(title="A* Algorithm Web Visualizer")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
app.state.preview_cache = {}


@dataclass(eq=False)
class CustomMapSpec:
    width: int
    height: int
    walls: np.ndarray
    serialized: str


@dataclass(eq=False)
class PreviewRequestSpec:
    size: int
    wall_prob: float
    seed: Optional[int]
    view_mode: str
    animation_speed: float
    custom_map: Optional[CustomMapSpec]
    start_goal: Optional[tuple[GridPos, GridPos]]


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


def _parse_dimension(value: object, name: str, max_value: Optional[int] = None) -> int:
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
    if max_value is not None and parsed > max_value:
        raise ValueError(f"`{name}` must be {max_value} or less.")
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


def _serialize_custom_map(width: int, height: int, wall_grid: np.ndarray) -> str:
    total_cells = width * height
    if total_cells >= 2500:
        walls_payload: list[object] = wall_grid.reshape(-1).tolist()
    else:
        walls_payload = wall_grid.tolist()

    return json.dumps(
        {"n": width, "k": height, "walls": walls_payload},
        ensure_ascii=False,
        separators=(",", ":"),
    )


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

    width = _parse_dimension(raw_width, "n", max_value=CUSTOM_MAP_MAX_DIM)
    height = _parse_dimension(raw_height, "k", max_value=CUSTOM_MAP_MAX_DIM)

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

    serialized = _serialize_custom_map(width=width, height=height, wall_grid=wall_grid)
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


def _cache_preview_request(preview_request: PreviewRequestSpec) -> str:
    preview_token = uuid4().hex
    preview_cache: MutableMapping[str, PreviewRequestSpec] = app.state.preview_cache
    preview_cache[preview_token] = preview_request
    while len(preview_cache) > MAX_PREVIEW_CACHE_ENTRIES:
        oldest_token = next(iter(preview_cache))
        del preview_cache[oldest_token]
    return preview_token


def _build_cached_preview_url(preview_token: str) -> str:
    return f"/preview?preview_token={preview_token}"


def _custom_map_for_preview_cache(custom_map: Optional[CustomMapSpec]) -> Optional[CustomMapSpec]:
    if custom_map is None:
        return None
    return CustomMapSpec(
        width=custom_map.width,
        height=custom_map.height,
        walls=custom_map.walls,
        serialized="",
    )


def _effective_dimensions(size: int, custom_map: Optional[CustomMapSpec]) -> tuple[int, int]:
    if custom_map is None:
        normalized_size = int(size)
        return normalized_size, normalized_size
    return custom_map.width, custom_map.height


def _resolve_view_mode_for_dimensions(
    requested_view_mode: str,
    width: int,
    height: int,
) -> tuple[str, bool]:
    normalized_view_mode = _normalize_view_mode(requested_view_mode)
    if normalized_view_mode == "animation" and max(width, height) >= LARGE_MAP_DIM_THRESHOLD:
        return "result", True
    return normalized_view_mode, False


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


def _frame_scale(width: int, height: int) -> int:
    return max(2, min(16, 640 // max(1, width, height)))


def _result_frame_scale(width: int, height: int) -> int:
    return max(1, min(RESULT_IMAGE_MAX_SCALE, RESULT_IMAGE_TARGET_PX // max(1, width, height)))


def _palette_array(visualizer: MazeAStarVisualizer) -> np.ndarray:
    return np.array(
        [
            tuple(int(channel * 255) for channel in mcolors.to_rgb(color))
            for color in visualizer._create_colormap().colors
        ],
        dtype=np.uint8,
    )


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

    palette = _palette_array(visualizer)
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

    final_frame = visualizer.solve_final_grid()
    palette = _palette_array(visualizer)
    scale = _result_frame_scale(width=visualizer.width, height=visualizer.height)
    rgb_frame = palette[final_frame]
    final_image = Image.fromarray(rgb_frame, mode="RGB")
    if scale > 1:
        final_image = final_image.resize(
            (final_image.width * scale, final_image.height * scale),
            Image.Resampling.NEAREST,
        )

    buf = BytesIO()
    final_image.save(buf, format="PNG", optimize=False, compress_level=1)
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
    notice_message: str = "",
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
        "notice_message": notice_message,
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
    notice_message = ""
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

    effective_width, effective_height = _effective_dimensions(size=size, custom_map=parsed_custom_map)
    applied_view_mode, coerced_to_result = _resolve_view_mode_for_dimensions(
        requested_view_mode=parsed_view_mode,
        width=effective_width,
        height=effective_height,
    )
    if coerced_to_result and not error_message:
        notice_message = LARGE_MAP_RESULT_NOTICE

    if error_message and parsed_custom_map is None:
        preview_url = _build_preview_url(
            size=size,
            wall_prob=wall_prob,
            seed=parsed_seed,
            view_mode=applied_view_mode,
            animation_speed=parsed_animation_speed,
            custom_map=parsed_custom_map.serialized if parsed_custom_map else "",
            start_goal=parsed_start_goal,
        )
    else:
        preview_token = _cache_preview_request(
            PreviewRequestSpec(
                size=size,
                wall_prob=wall_prob,
                seed=parsed_seed,
                view_mode=applied_view_mode,
                animation_speed=parsed_animation_speed,
                custom_map=_custom_map_for_preview_cache(parsed_custom_map),
                start_goal=parsed_start_goal,
            )
        )
        preview_url = _build_cached_preview_url(preview_token)

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
            view_mode=applied_view_mode,
            animation_speed=parsed_animation_speed,
            preview_url=preview_url,
            custom_map=custom_map,
            error_message=error_message,
            notice_message=notice_message,
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
    preview_token: str = "",
):
    if preview_token:
        preview_request: Optional[PreviewRequestSpec] = app.state.preview_cache.get(preview_token)
        if preview_request is None:
            raise HTTPException(status_code=404, detail="Preview token not found or expired.")

        size = preview_request.size
        wall_prob = preview_request.wall_prob
        seed = preview_request.seed
        parsed_custom_map = preview_request.custom_map
        parsed_start_goal = preview_request.start_goal
        parsed_view_mode = preview_request.view_mode
        animation_speed = preview_request.animation_speed
        coerced_to_result = False
    else:
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

        try:
            _validate_start_goal_for_request(
                start_goal=parsed_start_goal,
                size=size,
                custom_map=parsed_custom_map,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        effective_width, effective_height = _effective_dimensions(size=size, custom_map=parsed_custom_map)
        parsed_view_mode, coerced_to_result = _resolve_view_mode_for_dimensions(
            requested_view_mode=view_mode,
            width=effective_width,
            height=effective_height,
        )

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

    response_headers = {"Cache-Control": "no-store"}
    if coerced_to_result:
        response_headers["X-Preview-Mode-Applied"] = "result"

    return StreamingResponse(
        BytesIO(image_bytes),
        media_type=media_type,
        headers=response_headers,
    )
