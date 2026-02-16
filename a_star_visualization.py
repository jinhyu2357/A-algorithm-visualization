#!/usr/bin/env python3
"""A* pathfinding visualization on a 50x50 maze grid."""

from __future__ import annotations

import argparse
import heapq
import random
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

GridPos = Tuple[int, int]


@dataclass(order=True)
class PrioritizedNode:
    priority: float
    position: GridPos


class MazeAStarVisualizer:
    """Generate a maze and visualize A* execution with color-coded states."""

    FREE = 0
    WALL = 1
    OPEN = 2
    CLOSED = 3
    PATH = 4
    START = 5
    GOAL = 6

    def __init__(self, size: int = 50, wall_probability: float = 0.28, seed: Optional[int] = None):
        self.size = size
        self.wall_probability = wall_probability
        self.rng = random.Random(seed)
        self.start: GridPos = (1, 1)
        self.goal: GridPos = (size - 2, size - 2)

        self.base_maze = self._create_solvable_maze()

    def _create_random_maze(self) -> np.ndarray:
        maze = np.zeros((self.size, self.size), dtype=np.int8)

        for row in range(self.size):
            for col in range(self.size):
                if row in (0, self.size - 1) or col in (0, self.size - 1):
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
        queue: List[GridPos] = [start]
        visited = {start}

        while queue:
            current = queue.pop(0)
            if current == goal:
                return True

            for neighbor in self._neighbors(current):
                if neighbor in visited:
                    continue
                if maze[neighbor] == self.WALL:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        return False

    def _neighbors(self, position: GridPos) -> Iterable[GridPos]:
        row, col = position
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n_row, n_col = row + d_row, col + d_col
            if 0 <= n_row < self.size and 0 <= n_col < self.size:
                yield (n_row, n_col)

    @staticmethod
    def heuristic(a: GridPos, b: GridPos) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star_steps(self) -> Iterator[np.ndarray]:
        grid = self.base_maze.copy()
        grid[self.start] = self.START
        grid[self.goal] = self.GOAL

        open_heap: List[PrioritizedNode] = [PrioritizedNode(0, self.start)]
        g_score = {self.start: 0}
        came_from: dict[GridPos, GridPos] = {}
        open_set = {self.start}
        closed_set = set()

        yield grid.copy()

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
                        yield grid.copy()
                return

            for neighbor in self._neighbors(current):
                if self.base_maze[neighbor] == self.WALL or neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_heap, PrioritizedNode(f_score, neighbor))

                    if neighbor != self.goal and neighbor != self.start:
                        grid[neighbor] = self.OPEN
                    open_set.add(neighbor)

            grid[self.start] = self.START
            grid[self.goal] = self.GOAL
            yield grid.copy()

    def _reconstruct_path(self, came_from: dict[GridPos, GridPos], current: GridPos) -> List[GridPos]:
        path = []
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))

    def visualize(self, interval: float = 0.02, output: Optional[str] = None, show: bool = True) -> None:
        cmap = ListedColormap(
            [
                "#ffffff",  # FREE
                "#1f2937",  # WALL
                "#93c5fd",  # OPEN
                "#cbd5e1",  # CLOSED
                "#22c55e",  # PATH
                "#f59e0b",  # START
                "#ef4444",  # GOAL
            ]
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("A* Algorithm Visualization (50x50 maze)")
        ax.set_xticks([])
        ax.set_yticks([])

        image = ax.imshow(self.base_maze, cmap=cmap, vmin=0, vmax=6)
        for frame in self.a_star_steps():
            image.set_data(frame)
            if show:
                plt.pause(interval)

        if output:
            fig.savefig(output, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize A* pathfinding on a random maze.")
    parser.add_argument("--size", type=int, default=50, help="Grid size (default: 50)")
    parser.add_argument("--wall-prob", type=float, default=0.28, help="Wall density 0.0~1.0")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--interval", type=float, default=0.02, help="Animation delay in seconds")
    parser.add_argument("--output", type=str, default=None, help="Optional image output path")
    parser.add_argument("--no-show", action="store_true", help="Run without opening GUI window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualizer = MazeAStarVisualizer(size=args.size, wall_probability=args.wall_prob, seed=args.seed)
    visualizer.visualize(interval=args.interval, output=args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
