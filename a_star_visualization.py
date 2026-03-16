from __future__ import annotations

import argparse
from collections import deque
import heapq
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

GridPos = Tuple[int, int]
Priority = Tuple[float, float, float]


@dataclass(order=True)
class PrioritizedNode:
    priority: Priority
    position: GridPos


class MazeAStarVisualizer:
    """Generate a maze and visualize every A* step with color-coded states."""

    FREE = 0
    WALL = 1
    OPEN = 2
    CLOSED = 3
    PATH = 4
    START = 5
    GOAL = 6
    CURRENT = 7
    ORTHOGONAL_COST = 1.0
    DIAGONAL_COST = 2**0.5

    def __init__(
        self,
        size: int = 50,
        wall_probability: float = 0.28,
        seed: Optional[int] = None,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        wall_grid: Optional[np.ndarray] = None,
        start: Optional[GridPos] = None,
        goal: Optional[GridPos] = None,
    ):
        self.rng = random.Random(seed)
        self.wall_probability = wall_probability
        if (start is None) != (goal is None):
            raise ValueError("Both `start` and `goal` must be provided together.")

        if wall_grid is not None:
            normalized_grid = self._normalize_wall_grid(wall_grid)
            self.height, self.width = normalized_grid.shape
            if width is not None and int(width) != self.width:
                raise ValueError("`width` does not match custom wall grid width.")
            if height is not None and int(height) != self.height:
                raise ValueError("`height` does not match custom wall grid height.")
            if start is not None and goal is not None:
                self.start, self.goal, self.base_maze = self._prepare_manual_start_goal_for_fixed_maze(
                    start=start,
                    goal=goal,
                    maze=normalized_grid,
                )
            else:
                self.base_maze = normalized_grid
                self.start, self.goal = self._pick_start_goal_for_fixed_maze(self.base_maze, min_distance=10)
        else:
            self.width = int(width) if width is not None else int(size)
            self.height = int(height) if height is not None else int(size)
            if self.width < 3 or self.height < 3:
                raise ValueError("Grid width/height must both be at least 3.")
            if start is not None and goal is not None:
                self.start, self.goal = self._validate_manual_start_goal(start=start, goal=goal, maze=None)
            else:
                self.start, self.goal = self._pick_start_goal(min_distance=100)
            self.base_maze = self._create_solvable_maze()

        self.goal_direction = self._direction_vector(self.start, self.goal)

    def _normalize_wall_grid(self, wall_grid: np.ndarray) -> np.ndarray:
        if wall_grid.ndim != 2:
            raise ValueError("Custom wall grid must be a 2D array.")
        normalized = wall_grid.astype(np.int8, copy=True)
        if normalized.shape[0] <= 0 or normalized.shape[1] <= 0:
            raise ValueError("Custom wall grid cannot be empty.")
        if not np.isin(normalized, [self.FREE, self.WALL]).all():
            raise ValueError("Custom wall grid values must be 0 (free) or 1 (wall).")
        return normalized

    def _normalize_position(self, position: GridPos, name: str) -> GridPos:
        try:
            row_raw, col_raw = position
        except Exception as exc:
            raise ValueError(f"`{name}` must be a (row, col) pair.") from exc

        if isinstance(row_raw, bool) or isinstance(col_raw, bool):
            raise ValueError(f"`{name}` row/col must be integers.")
        if isinstance(row_raw, float) and not row_raw.is_integer():
            raise ValueError(f"`{name}` row/col must be integers.")
        if isinstance(col_raw, float) and not col_raw.is_integer():
            raise ValueError(f"`{name}` row/col must be integers.")

        try:
            row = int(row_raw)
            col = int(col_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"`{name}` row/col must be integers.") from exc

        if row < 0 or col < 0 or row >= self.height or col >= self.width:
            raise ValueError(
                f"`{name}` position {(row, col)} is out of bounds for grid size {self.height}x{self.width}."
            )
        return (row, col)

    def _validate_manual_start_goal(
        self,
        start: GridPos,
        goal: GridPos,
        maze: Optional[np.ndarray],
    ) -> Tuple[GridPos, GridPos]:
        normalized_start = self._normalize_position(start, "start")
        normalized_goal = self._normalize_position(goal, "goal")

        if normalized_start == normalized_goal:
            raise ValueError("Start and goal must be different positions.")

        if maze is not None:
            if maze[normalized_start] == self.WALL:
                raise ValueError("`start` must be on a free cell (0) in the current map.")
            if maze[normalized_goal] == self.WALL:
                raise ValueError("`goal` must be on a free cell (0) in the current map.")
            if not self._has_path(maze, normalized_start, normalized_goal):
                raise ValueError("No path exists between the provided start and goal positions.")

        return normalized_start, normalized_goal

    def _prepare_manual_start_goal_for_fixed_maze(
        self,
        start: GridPos,
        goal: GridPos,
        maze: np.ndarray,
    ) -> Tuple[GridPos, GridPos, np.ndarray]:
        normalized_start, normalized_goal = self._validate_manual_start_goal(start=start, goal=goal, maze=None)
        adjusted_maze = maze.copy()
        adjusted_maze[normalized_start] = self.FREE
        adjusted_maze[normalized_goal] = self.FREE

        if not self._has_path(adjusted_maze, normalized_start, normalized_goal):
            raise ValueError("No path exists between the provided start and goal positions.")

        return normalized_start, normalized_goal, adjusted_maze

    def _pick_start_goal(self, min_distance: int) -> Tuple[GridPos, GridPos]:
        candidates = [(row, col) for row in range(1, self.height - 1) for col in range(1, self.width - 1)]
        if len(candidates) < 2:
            raise ValueError("Not enough free cells to place start and goal.")

        max_distance = self.heuristic((1, 1), (self.height - 2, self.width - 2))
        required_distance = min(min_distance, max_distance)

        for _ in range(1000):
            start = self.rng.choice(candidates)
            goal = self.rng.choice(candidates)
            if start == goal:
                continue
            if self.heuristic(start, goal) >= required_distance:
                return start, goal

        raise RuntimeError("Could not choose start/goal with required distance. Try increasing grid size.")

    def _pick_start_goal_for_fixed_maze(self, maze: np.ndarray, min_distance: int) -> Tuple[GridPos, GridPos]:
        free_cells = [tuple(int(idx) for idx in pos) for pos in np.argwhere(maze == self.FREE)]
        if len(free_cells) < 2:
            raise ValueError("Custom map must include at least two free cells for start and goal.")

        max_distance = self.heuristic((0, 0), (self.height - 1, self.width - 1))
        required_distance = min(min_distance, max_distance)

        for _ in range(3000):
            start = self.rng.choice(free_cells)
            goal = self.rng.choice(free_cells)
            if start == goal:
                continue
            if self.heuristic(start, goal) < required_distance:
                continue
            if self._has_path(maze, start, goal):
                return start, goal

        for _ in range(3000):
            start = self.rng.choice(free_cells)
            goal = self.rng.choice(free_cells)
            if start == goal:
                continue
            if self._has_path(maze, start, goal):
                return start, goal

        raise RuntimeError("Custom map does not contain any reachable start/goal pair.")

    def _create_random_maze(self) -> np.ndarray:
        maze = np.zeros((self.height, self.width), dtype=np.int8)
        for row in range(self.height):
            for col in range(self.width):
                if row in (0, self.height - 1) or col in (0, self.width - 1):
                    maze[row, col] = self.WALL
                elif self.rng.random() < self.wall_probability:
                    maze[row, col] = self.WALL

        maze[self.start] = self.FREE
        maze[self.goal] = self.FREE
        return maze

    def _create_solvable_maze(self, max_attempts: int = 200) -> np.ndarray:
        for _ in range(max_attempts):
            maze = self._create_random_maze()
            if self._has_path(maze, self.start, self.goal):
                return maze
        raise RuntimeError("Could not generate a solvable maze. Try lowering wall density.")

    def _has_path(self, maze: np.ndarray, start: GridPos, goal: GridPos) -> bool:
        queue: deque[GridPos] = deque([start])
        visited = {start}

        while queue:
            current = queue.popleft()
            if current == goal:
                return True

            for neighbor, _ in self._neighbors(current, maze=maze):
                if neighbor in visited or maze[neighbor] == self.WALL:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        return False

    def _neighbors(self, position: GridPos, maze: Optional[np.ndarray] = None) -> Iterable[Tuple[GridPos, float]]:
        row, col = position
        for d_row, d_col, move_cost in (
            (1, 0, self.ORTHOGONAL_COST),
            (-1, 0, self.ORTHOGONAL_COST),
            (0, 1, self.ORTHOGONAL_COST),
            (0, -1, self.ORTHOGONAL_COST),
            (1, 1, self.DIAGONAL_COST),
            (1, -1, self.DIAGONAL_COST),
            (-1, 1, self.DIAGONAL_COST),
            (-1, -1, self.DIAGONAL_COST),
        ):
            n_row, n_col = row + d_row, col + d_col
            if not (0 <= n_row < self.height and 0 <= n_col < self.width):
                continue

            if maze is not None and d_row != 0 and d_col != 0:
                # Block corner-cutting through walls while allowing true diagonal movement.
                if maze[row + d_row, col] == self.WALL or maze[row, col + d_col] == self.WALL:
                    continue

            yield (n_row, n_col), move_cost

    @staticmethod
    def heuristic(a: GridPos, b: GridPos) -> float:
        d_row = abs(a[0] - b[0])
        d_col = abs(a[1] - b[1])
        min_delta = min(d_row, d_col)
        max_delta = max(d_row, d_col)
        return min_delta * MazeAStarVisualizer.DIAGONAL_COST + (max_delta - min_delta)

    @staticmethod
    def _direction_vector(start: GridPos, goal: GridPos) -> Tuple[float, float]:
        d_row = goal[0] - start[0]
        d_col = goal[1] - start[1]
        norm = (d_row**2 + d_col**2) ** 0.5
        if norm == 0:
            return (0.0, 0.0)
        return (d_row / norm, d_col / norm)

    def _direction_penalty(self, current: GridPos, neighbor: GridPos) -> float:
        step_row = neighbor[0] - current[0]
        step_col = neighbor[1] - current[1]
        step_norm = (step_row**2 + step_col**2) ** 0.5
        if step_norm == 0:
            return 1.0
        alignment = (step_row / step_norm) * self.goal_direction[0] + (step_col / step_norm) * self.goal_direction[1]
        # Lower penalty means the step is better aligned with the start->goal direction vector.
        return 1.0 - alignment

    def _ordered_neighbors(self, position: GridPos) -> List[Tuple[GridPos, float]]:
        neighbors = list(self._neighbors(position, maze=self.base_maze))
        neighbors.sort(key=lambda item: self._direction_penalty(position, item[0]))
        return neighbors

    def a_star_steps(self) -> Iterator[np.ndarray]:
        """Yield each search step so users can observe the running process, not just the final result."""
        grid = self.base_maze.copy()
        grid[self.start] = self.START
        grid[self.goal] = self.GOAL

        start_h = self.heuristic(self.start, self.goal)
        open_heap: List[PrioritizedNode] = [PrioritizedNode((start_h, 0.0, start_h), self.start)]
        g_score = {self.start: 0.0}
        came_from: dict[GridPos, GridPos] = {}
        open_set = {self.start}
        closed_set = set()

        yield grid.copy()

        while open_heap:
            current = heapq.heappop(open_heap).position
            if current not in open_set:
                continue

            open_set.remove(current)
            step_grid = grid.copy()
            if current not in (self.start, self.goal):
                step_grid[current] = self.CURRENT
            yield self._mark_start_goal(step_grid)

            if current != self.start:
                closed_set.add(current)
                grid[current] = self.CLOSED

            if current == self.goal:
                for step in self._reconstruct_path(came_from, current):
                    if step not in (self.start, self.goal):
                        grid[step] = self.PATH
                        yield self._mark_start_goal(grid.copy())
                return

            for neighbor, move_cost in self._ordered_neighbors(current):
                if self.base_maze[neighbor] == self.WALL or neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + move_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score = self.heuristic(neighbor, self.goal)
                    f_score = tentative_g + h_score
                    direction_penalty = self._direction_penalty(current, neighbor)
                    heapq.heappush(open_heap, PrioritizedNode((f_score, direction_penalty, h_score), neighbor))

                    if neighbor not in (self.start, self.goal):
                        grid[neighbor] = self.OPEN
                    open_set.add(neighbor)

            yield self._mark_start_goal(grid.copy())

    def solve_final_grid(self) -> np.ndarray:
        """Run A* to completion and return only the final rendered grid."""
        grid = self.base_maze.copy()
        grid[self.start] = self.START
        grid[self.goal] = self.GOAL

        start_h = self.heuristic(self.start, self.goal)
        open_heap: List[PrioritizedNode] = [PrioritizedNode((start_h, 0.0, start_h), self.start)]
        g_score = {self.start: 0.0}
        came_from: dict[GridPos, GridPos] = {}
        open_set = {self.start}
        closed_set = set()

        while open_heap:
            current = heapq.heappop(open_heap).position
            if current not in open_set:
                continue

            open_set.remove(current)

            if current != self.start:
                closed_set.add(current)
                grid[current] = self.CLOSED

            if current == self.goal:
                for step in self._reconstruct_path(came_from, current):
                    if step not in (self.start, self.goal):
                        grid[step] = self.PATH
                return self._mark_start_goal(grid)

            for neighbor, move_cost in self._ordered_neighbors(current):
                if self.base_maze[neighbor] == self.WALL or neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + move_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score = self.heuristic(neighbor, self.goal)
                    f_score = tentative_g + h_score
                    direction_penalty = self._direction_penalty(current, neighbor)
                    heapq.heappush(open_heap, PrioritizedNode((f_score, direction_penalty, h_score), neighbor))

                    if neighbor not in (self.start, self.goal):
                        grid[neighbor] = self.OPEN
                    open_set.add(neighbor)

        return self._mark_start_goal(grid)

    def _mark_start_goal(self, grid: np.ndarray) -> np.ndarray:
        grid[self.start] = self.START
        grid[self.goal] = self.GOAL
        return grid

    def _reconstruct_path(self, came_from: dict[GridPos, GridPos], current: GridPos) -> List[GridPos]:
        path = []
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))

    def _create_colormap(self) -> ListedColormap:
        return ListedColormap(
            [
                "#ffffff",  # FREE
                "#1f2937",  # WALL
                "#93c5fd",  # OPEN
                "#cbd5e1",  # CLOSED
                "#22c55e",  # PATH
                "#f59e0b",  # START
                "#ef4444",  # GOAL
                "#a855f7",  # CURRENT
            ]
        )

    def visualize(self, interval: float = 0.02, output: Optional[str] = None, show: bool = True) -> None:
        """Show live animation and optionally export process animation (gif/mp4) or final image (png)."""
        frames = list(self.a_star_steps())

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("A* Algorithm Process Visualization")
        ax.set_xticks([])
        ax.set_yticks([])

        image = ax.imshow(frames[0], cmap=self._create_colormap(), vmin=0, vmax=7)

        def update(frame: np.ndarray):
            image.set_data(frame)
            return (image,)

        interval_ms = max(1, int(interval * 1000))
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=interval_ms,
            blit=True,
            repeat=False,
        )

        if output:
            suffix = Path(output).suffix.lower()
            if suffix in {".gif", ".mp4"}:
                ani.save(output)
            else:
                fig.savefig(output, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize A* pathfinding process on a random maze.")
    parser.add_argument("--size", type=int, default=50, help="Grid size (default: 50)")
    parser.add_argument("--width", type=int, default=None, help="Grid width (overrides --size)")
    parser.add_argument("--height", type=int, default=None, help="Grid height (overrides --size)")
    parser.add_argument("--wall-prob", type=float, default=0.28, help="Wall density 0.0~1.0")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--interval", type=float, default=0.02, help="Animation delay in seconds")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (.gif/.mp4 for process animation, others for final frame image)",
    )
    parser.add_argument("--no-show", action="store_true", help="Run without opening GUI window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualizer = MazeAStarVisualizer(
        size=args.size,
        width=args.width,
        height=args.height,
        wall_probability=args.wall_prob,
        seed=args.seed,
    )
    visualizer.visualize(interval=args.interval, output=args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
