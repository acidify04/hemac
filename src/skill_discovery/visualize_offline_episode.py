"""Interactively replay one HeMAC offline-data episode saved as a PT file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "src/skill_discovery/offline_data"
DEFAULT_FPS = 5.0

FALLBACK_CHANNEL_NAMES = {
    "observer": {
        "global_map": (
            "coverage",
            "boundary",
            "obstacle",
            "warning",
            "drones",
            "goal",
        ),
        "local_map": (
            "coverage",
            "boundary",
            "obstacle",
            "warning",
            "drones",
            "goal",
        ),
    },
    "drone": {
        "global_map": (
            "coverage",
            "boundary",
            "obstacle",
            "warning",
            "other_drones",
            "observer",
            "goal",
        ),
        "local_map": (
            "coverage",
            "boundary",
            "obstacle",
            "warning",
            "other_drones",
            "observer",
            "goal",
        ),
    },
    "global_state": {
        "central_map": (
            "coverage",
            "boundary",
            "obstacle",
            "warning",
            "all_drones",
            "observer",
            "goal",
        ),
    },
}

CHANNEL_COLORS = {
    "obstacle": np.asarray((0.95, 0.12, 0.10), dtype=np.float32),
    "warning": np.asarray((1.00, 0.43, 0.16), dtype=np.float32),
    "drones": np.asarray((0.10, 0.88, 1.00), dtype=np.float32),
    "other_drones": np.asarray((0.10, 0.88, 1.00), dtype=np.float32),
    "all_drones": np.asarray((0.10, 0.88, 1.00), dtype=np.float32),
    "observer": np.asarray((1.00, 0.82, 0.18), dtype=np.float32),
    "goal": np.asarray((1.00, 0.25, 0.72), dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    """Parse episode selection and playback options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episode",
        nargs="?",
        type=Path,
        help="PT episode file or a directory containing episode files.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--agent",
        default="observer_0",
        help="Initially selected agent, for example observer_0 or drone_1.",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--auto",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start in automatic playback mode.",
    )
    parser.add_argument(
        "--save-gif",
        type=Path,
        help="Render the full episode to this GIF instead of opening a window.",
    )
    return parser.parse_args()


def find_episode_path(path: Path | None, data_dir: Path) -> Path:
    """Resolve a file argument or select the most recently modified episode."""
    search_root = data_dir if path is None else path
    search_root = search_root.expanduser().resolve()
    if search_root.is_file():
        if search_root.suffix != ".pt":
            raise ValueError(f"Episode must be a .pt file: {search_root}")
        return search_root
    if not search_root.is_dir():
        raise FileNotFoundError(f"Episode path does not exist: {search_root}")

    candidates = list(search_root.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt episodes found under {search_root}")
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime_ns)


def load_episode(path: Path) -> dict[str, Any]:
    """Load and minimally validate a joint-trajectory episode."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "metadata",
        "global_state",
        "observer",
        "drone",
        "individual_rewards",
        "team_reward",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(
            f"{path} is not a supported joint episode; missing {sorted(missing)}."
        )
    if "central_map" not in payload["global_state"]:
        raise ValueError(f"{path} has no global_state.central_map tensor.")
    return payload


def as_numpy(value: Any) -> np.ndarray:
    """Return a CPU NumPy view for tensor-like data."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def blend(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    """Alpha-blend one semantic channel into an RGB map in place."""
    amount = np.clip(mask, 0.0, 1.0)[..., None] * float(alpha)
    rgb *= 1.0 - amount
    rgb += amount * color


def render_semantic_map(
    channels: Any,
    channel_names: tuple[str, ...] | list[str],
    obstacle_scale: float = 1.0,
) -> np.ndarray:
    """Convert a CHW semantic map into an RGB visualization."""
    array = np.asarray(as_numpy(channels), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Semantic map must be CHW, got {array.shape}.")
    names = tuple(channel_names)
    if len(names) != array.shape[0]:
        names = tuple(f"channel_{index}" for index in range(array.shape[0]))

    height, width = array.shape[1:]
    rgb = np.empty((height, width, 3), dtype=np.float32)
    rgb[:] = (0.025, 0.035, 0.045)

    if "boundary" in names:
        search_mask = np.clip(array[names.index("boundary")], 0.0, 1.0)
        inside = np.asarray((0.075, 0.095, 0.075), dtype=np.float32)
        rgb[:] = rgb * (1.0 - search_mask[..., None]) + inside * search_mask[..., None]
    if "coverage" in names:
        blend(
            rgb,
            array[names.index("coverage")],
            np.asarray((0.12, 0.88, 0.38), dtype=np.float32),
            0.78,
        )

    for name in names:
        color = CHANNEL_COLORS.get(name)
        if color is None:
            continue
        channel = array[names.index(name)]
        if name == "obstacle":
            normalized = channel / max(
                float(obstacle_scale),
                np.finfo(np.float32).tiny,
            )
            channel = np.sqrt(np.clip(normalized, 0.0, 1.0))
            # Keep decayed-but-still-stored obstacle beliefs visible. The title
            # reports the raw confidence so this floor is not mistaken for 1.0.
            channel = np.where(
                array[names.index(name)] > 0.0,
                np.maximum(channel, 0.3),
                0.0,
            )
        alpha = 0.55 if name == "warning" else 0.95
        blend(rgb, channel, color, alpha)
    return np.clip(rgb, 0.0, 1.0)


class EpisodeViewer:
    """Matplotlib controller for interactive offline-episode playback."""

    def __init__(
        self,
        payload: dict[str, Any],
        episode_path: Path,
        initial_agent: str,
        fps: float,
        autoplay: bool,
    ) -> None:
        self.payload = payload
        self.episode_path = episode_path
        self.metadata = payload["metadata"]
        self.agent_order = list(self.metadata.get("agent_order", ()))
        if not self.agent_order:
            self.agent_order = list(payload["observer"].get("agent_ids", ()))
            self.agent_order.extend(payload["drone"].get("agent_ids", ()))
        if not self.agent_order:
            raise ValueError("Episode metadata contains no agent IDs.")
        if initial_agent not in self.agent_order:
            available = ", ".join(self.agent_order)
            raise ValueError(f"Unknown agent {initial_agent!r}; choose one of {available}.")

        self.channel_names = self.metadata.get(
            "channel_names",
            FALLBACK_CHANNEL_NAMES,
        )
        self.selected_agent = initial_agent
        self.step = 0
        self.transition_count = int(payload["team_reward"].shape[0])
        self.max_step = self.transition_count
        self.playing = bool(autoplay)
        self.interval_ms = max(int(1000.0 / max(float(fps), 0.1)), 1)
        self.obstacle_scales = self.build_obstacle_scales()

        self.figure = plt.figure(figsize=(15.5, 9.2), facecolor="#091016")
        grid = self.figure.add_gridspec(
            2,
            3,
            height_ratios=(3.0, 1.75),
            hspace=0.26,
            wspace=0.16,
        )
        self.central_axis = self.figure.add_subplot(grid[0, 0])
        self.global_axis = self.figure.add_subplot(grid[0, 1])
        self.local_axis = self.figure.add_subplot(grid[0, 2])
        self.reward_axis = self.figure.add_subplot(grid[1, :2])
        self.action_axis = self.figure.add_subplot(grid[1, 2])
        self.axes = (
            self.central_axis,
            self.global_axis,
            self.local_axis,
            self.reward_axis,
            self.action_axis,
        )
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.timer = self.figure.canvas.new_timer(interval=self.interval_ms)
        self.timer.add_callback(self.advance)
        if self.playing:
            self.timer.start()
        self.draw()

    def role_and_index(self) -> tuple[str, int]:
        """Return dataset role and within-role index for the selected agent."""
        role = "observer" if self.selected_agent.startswith("observer_") else "drone"
        agent_ids = list(self.payload[role].get("agent_ids", ()))
        if not agent_ids:
            agent_ids = [
                agent_id
                for agent_id in self.agent_order
                if agent_id.startswith(f"{role}_")
            ]
        return role, agent_ids.index(self.selected_agent)

    def channel_names_for(self, group: str, map_name: str) -> tuple[str, ...]:
        """Read channel names with a stable fallback for older files."""
        names = self.channel_names.get(group, {}).get(map_name)
        if names is None:
            names = FALLBACK_CHANNEL_NAMES[group][map_name]
        return tuple(names)

    def build_obstacle_scales(self) -> dict[tuple[str, str], float]:
        """Find episode-level obstacle maxima for confidence-preserving display."""
        scales = {}
        map_sources = {
            ("global_state", "central_map"): self.payload["global_state"]["central_map"],
            ("observer", "global_map"): self.payload["observer"]["observations"]["global_map"],
            ("observer", "local_map"): self.payload["observer"]["observations"]["local_map"],
            ("drone", "global_map"): self.payload["drone"]["observations"]["global_map"],
            ("drone", "local_map"): self.payload["drone"]["observations"]["local_map"],
        }
        for key, maps in map_sources.items():
            names = self.channel_names_for(*key)
            if "obstacle" not in names:
                scales[key] = 1.0
                continue
            obstacle = as_numpy(maps)[..., names.index("obstacle"), :, :]
            maximum = float(np.max(obstacle, initial=0.0))
            scales[key] = maximum if maximum > 0.0 else 1.0
        return scales

    def transition_value(self, name: str, default: bool = False) -> bool:
        """Read a transition flag, using final outcome at the T+1 state."""
        if self.step < self.transition_count and name in self.payload:
            return bool(as_numpy(self.payload[name])[self.step].reshape(-1)[0])
        outcome = self.payload.get("outcome", {})
        return bool(outcome.get(name, default))

    def draw_map(
        self,
        axis,
        channels: Any,
        names: tuple[str, ...],
        title: str,
        obstacle_scale: float,
    ) -> None:
        """Draw one semantic map with consistent styling."""
        channel_array = as_numpy(channels)
        rgb = render_semantic_map(channel_array, names, obstacle_scale)
        axis.imshow(rgb, interpolation="nearest", origin="upper")
        obstacle_summary = ""
        if "obstacle" in names:
            obstacle = channel_array[names.index("obstacle")]
            obstacle_summary = (
                f"  |  obstacle max={float(np.max(obstacle, initial=0.0)):.2e}, "
                f"cells={int(np.count_nonzero(obstacle))}"
            )
        axis.set_title(
            title + obstacle_summary,
            color="#e9f1f7",
            fontsize=10,
            pad=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#405363")

    def draw_rewards(self) -> None:
        """Draw team and per-agent transition rewards up to the cursor."""
        axis = self.reward_axis
        team = as_numpy(self.payload["team_reward"]).reshape(-1)
        individual = as_numpy(self.payload["individual_rewards"])
        x_values = np.arange(self.transition_count)
        axis.plot(x_values, team, color="#f5f0d0", linewidth=1.7, label="team")
        palette = ("#f4c95d", "#47d7ef", "#76e39b", "#7ea6ff")
        for index, agent_id in enumerate(self.agent_order):
            axis.plot(
                x_values,
                individual[:, index],
                color=palette[index % len(palette)],
                linewidth=1.0,
                alpha=0.8,
                label=agent_id,
            )
        axis.axvline(
            min(self.step, max(self.transition_count - 1, 0)),
            color="#ff5f87",
            linewidth=1.2,
        )
        axis.set_xlim(0, max(self.transition_count - 1, 1))
        axis.set_yscale("symlog", linthresh=0.05, linscale=1.0)
        axis.set_title(
            "Reward timeline (symmetric log scale)",
            color="#e9f1f7",
            fontsize=12,
        )
        axis.set_xlabel("cycle", color="#9eb1bf")
        axis.grid(color="#263743", alpha=0.5, linewidth=0.6)
        axis.tick_params(colors="#9eb1bf")
        axis.legend(
            loc="upper left",
            ncol=min(len(self.agent_order) + 1, 5),
            fontsize=8,
            frameon=False,
            labelcolor="#dbe7ef",
        )
        if self.step < self.transition_count:
            values = [f"team={team[self.step]:.3f}"]
            values.extend(
                f"{agent_id}={individual[self.step, index]:.3f}"
                for index, agent_id in enumerate(self.agent_order)
            )
            axis.text(
                0.99,
                0.04,
                "  ".join(values),
                transform=axis.transAxes,
                color="#dbe7ef",
                fontsize=8,
                family="monospace",
                ha="right",
                va="bottom",
                bbox={"facecolor": "#0b141b", "alpha": 0.78, "edgecolor": "none"},
            )

    def draw_actions(self, role: str, role_index: int) -> None:
        """Draw current action and the observation's five-action history."""
        axis = self.action_axis
        history = as_numpy(
            self.payload[role]["observations"]["action_history"]
        )[self.step, role_index]
        history_x = np.arange(-history.shape[0], 0)
        colors = ("#ff6b6b", "#4ecdc4", "#ffe66d")
        labels = ("action x", "action y", "action z")
        for component in range(min(history.shape[1], 3)):
            axis.plot(
                history_x,
                history[:, component],
                marker="o",
                markersize=3,
                linewidth=1.4,
                color=colors[component],
                label=labels[component],
            )

        action_text = "terminal observation"
        reward_text = "reward: n/a"
        mask_text = "acted: False"
        if self.step < self.transition_count:
            action = as_numpy(self.payload[role]["actions"])[self.step, role_index]
            agent_index = self.agent_order.index(self.selected_agent)
            reward = float(as_numpy(self.payload["individual_rewards"])[self.step, agent_index])
            acted = bool(as_numpy(self.payload["agent_mask"])[self.step, agent_index])
            action_text = "action: [" + ", ".join(f"{value:6.2f}" for value in action) + "]"
            reward_text = f"individual reward: {reward:8.3f}"
            mask_text = f"acted: {acted}"

        axis.set_title(f"{self.selected_agent} action", color="#e9f1f7", fontsize=12)
        axis.set_xlabel("previous action index", color="#9eb1bf")
        axis.grid(color="#263743", alpha=0.5, linewidth=0.6)
        axis.tick_params(colors="#9eb1bf")
        axis.legend(frameon=False, fontsize=8, labelcolor="#dbe7ef")
        axis.text(
            0.02,
            0.04,
            f"{action_text}\n{reward_text}\n{mask_text}",
            transform=axis.transAxes,
            color="#dbe7ef",
            fontsize=9,
            family="monospace",
            va="bottom",
        )

    def draw(self) -> None:
        """Redraw the dashboard at the current observation index."""
        for axis in self.axes:
            axis.clear()
            axis.set_facecolor("#0d171f")

        role, role_index = self.role_and_index()
        central_map = self.payload["global_state"]["central_map"][self.step]
        observations = self.payload[role]["observations"]
        global_map = observations["global_map"][self.step, role_index]
        local_map = observations["local_map"][self.step, role_index]
        self.draw_map(
            self.central_axis,
            central_map,
            self.channel_names_for("global_state", "central_map"),
            "Shared central map",
            self.obstacle_scales[("global_state", "central_map")],
        )
        self.draw_map(
            self.global_axis,
            global_map,
            self.channel_names_for(role, "global_map"),
            f"{self.selected_agent} global map",
            self.obstacle_scales[(role, "global_map")],
        )
        self.draw_map(
            self.local_axis,
            local_map,
            self.channel_names_for(role, "local_map"),
            f"{self.selected_agent} local map",
            self.obstacle_scales[(role, "local_map")],
        )
        self.draw_rewards()
        self.draw_actions(role, role_index)

        outcome = self.payload.get("outcome", {})
        category = outcome.get(
            "category",
            self.metadata.get("outcome_category", "unknown"),
        )
        goal_found = self.transition_value("goal_found")
        drone_goal_found = self.transition_value("drone_goal_found")
        success = self.transition_value("success")
        mode = "AUTO" if self.playing else "STEP"
        self.figure.suptitle(
            f"{self.episode_path.name}  |  category: {category}  |  "
            f"cycle {self.step}/{self.max_step}  |  {mode}\n"
            f"goal found: {goal_found}  |  drone goal found: {drone_goal_found}  "
            f"|  success: {success}",
            color="#f2f7fa",
            fontsize=13,
            y=0.985,
        )
        self.figure.text(
            0.5,
            0.012,
            "Space: auto/step   Left/Right: cycle   1-4: agent   "
            "Home/End: first/last   Q/Esc: quit",
            color="#8fa7b7",
            ha="center",
            fontsize=9,
        )
        self.figure.canvas.draw_idle()

    def set_step(self, step: int) -> None:
        """Move to a bounded observation index and redraw."""
        self.step = min(max(int(step), 0), self.max_step)
        self.draw()

    def advance(self) -> None:
        """Advance automatic playback, stopping at the final observation."""
        if not self.playing:
            return
        if self.step >= self.max_step:
            self.playing = False
            self.timer.stop()
            self.draw()
            return
        self.set_step(self.step + 1)

    def toggle_playback(self) -> None:
        """Switch between automatic and manual playback."""
        self.playing = not self.playing
        if self.playing:
            if self.step >= self.max_step:
                self.step = 0
            self.timer.start()
        else:
            self.timer.stop()
        self.draw()

    def on_key(self, event) -> None:
        """Handle interactive playback and selected-agent controls."""
        key = (event.key or "").lower()
        if key in {"escape", "q"}:
            plt.close(self.figure)
        elif key in {" ", "space"}:
            self.toggle_playback()
        elif key in {"right", "d"}:
            self.set_step(self.step + 1)
        elif key in {"left", "a"}:
            self.set_step(self.step - 1)
        elif key == "home":
            self.set_step(0)
        elif key == "end":
            self.set_step(self.max_step)
        elif key in {str(index) for index in range(1, len(self.agent_order) + 1)}:
            self.selected_agent = self.agent_order[int(key) - 1]
            self.draw()

    def save_gif(self, output_path: Path, fps: float) -> None:
        """Render every stored observation into an animated GIF."""
        self.playing = False
        self.timer.stop()
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def update(frame: int):
            self.step = frame
            self.draw()
            return ()

        movie = animation.FuncAnimation(
            self.figure,
            update,
            frames=self.max_step + 1,
            interval=max(int(1000.0 / max(fps, 0.1)), 1),
            blit=False,
        )
        movie.save(output_path, writer=animation.PillowWriter(fps=max(fps, 0.1)))
        print(f"Saved GIF: {output_path}")


def main() -> None:
    """Open an interactive viewer or export one episode to GIF."""
    args = parse_args()
    episode_path = find_episode_path(args.episode, args.data_dir)
    payload = load_episode(episode_path)
    viewer = EpisodeViewer(
        payload,
        episode_path,
        initial_agent=args.agent,
        fps=args.fps,
        autoplay=args.auto,
    )
    print(
        f"Loaded {episode_path} | transitions={viewer.transition_count} | "
        f"agents={viewer.agent_order}"
    )
    if args.save_gif is not None:
        viewer.save_gif(args.save_gif, args.fps)
        plt.close(viewer.figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
