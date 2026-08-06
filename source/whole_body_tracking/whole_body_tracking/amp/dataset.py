"""Expert transition loading for the recovery AMP discriminator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .features import build_amp_state


_REQUIRED_ARRAYS = (
    "schema_version",
    "fps",
    "body_names",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass(frozen=True)
class AmpClipInfo:
    path: str
    frames: int
    fps: float
    stride: int
    transitions: int


class AmpExpertDataset:
    """Precompute fixed-length AMP state pairs from one file or a directory."""

    def __init__(
        self,
        motion_path: str | Path,
        body_names: list[str],
        anchor_body_name: str,
        step_dt: float,
        device: str | torch.device,
        required_clip_name: str | None = None,
    ):
        self.device = torch.device(device)
        paths = self._resolve_paths(Path(motion_path))
        states: list[torch.Tensor] = []
        next_states: list[torch.Tensor] = []
        clip_infos: list[AmpClipInfo] = []

        if anchor_body_name in body_names:
            raise ValueError("AMP body_names must exclude the anchor body.")
        if not body_names:
            raise ValueError("AMP body_names must contain at least one body.")

        required_states: torch.Tensor | None = None
        required_next_states: torch.Tensor | None = None
        random_states: list[torch.Tensor] = []
        random_next_states: list[torch.Tensor] = []
        for path in paths:
            state, next_state, info = self._load_clip(
                path, body_names, anchor_body_name, step_dt, self.device
            )
            states.append(state)
            next_states.append(next_state)
            clip_infos.append(info)
            if required_clip_name is not None and path.name == required_clip_name:
                if required_states is not None:
                    raise ValueError(
                        f"Required AMP expert clip name is ambiguous: {required_clip_name}"
                    )
                required_states = state
                required_next_states = next_state
            else:
                random_states.append(state)
                random_next_states.append(next_state)

        if required_clip_name is not None and required_states is None:
            raise FileNotFoundError(
                f"Required AMP expert clip was not found in {motion_path}: {required_clip_name}"
            )
        if required_clip_name is not None and not random_states:
            raise ValueError("AMP expert sampling requires at least one non-required clip.")

        self.states = torch.cat(states, dim=0).contiguous()
        self.next_states = torch.cat(next_states, dim=0).contiguous()
        self.clip_infos = tuple(clip_infos)
        self.state_dim = self.states.shape[1]
        self.required_clip_name = required_clip_name
        self.required_states = required_states
        self.required_next_states = required_next_states
        self.random_states = torch.cat(random_states, dim=0).contiguous() if random_states else self.states
        self.random_next_states = (
            torch.cat(random_next_states, dim=0).contiguous() if random_next_states else self.next_states
        )

    @staticmethod
    def _resolve_paths(motion_path: Path) -> list[Path]:
        if motion_path.is_file():
            if motion_path.suffix != ".npz":
                raise ValueError(f"AMP expert file must be .npz: {motion_path}")
            return [motion_path]
        if not motion_path.is_dir():
            raise FileNotFoundError(f"AMP expert path does not exist: {motion_path}")
        paths = sorted(motion_path.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No .npz AMP expert files found in: {motion_path}")
        return paths

    @staticmethod
    def _load_clip(
        path: Path,
        selected_body_names: list[str],
        anchor_body_name: str,
        step_dt: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, AmpClipInfo]:
        with np.load(path, allow_pickle=False) as data:
            missing_arrays = [key for key in _REQUIRED_ARRAYS if key not in data]
            if missing_arrays:
                raise ValueError(f"{path} is missing AMP arrays: {missing_arrays}")
            schema_version = int(np.asarray(data["schema_version"]).item())
            if schema_version != 2:
                raise ValueError(f"{path} has unsupported schema_version={schema_version}; expected 2")
            fps = float(np.asarray(data["fps"]).item())
            if fps <= 0.0 or step_dt <= 0.0:
                raise ValueError(f"fps and step_dt must be positive, got fps={fps}, step_dt={step_dt}")
            stride = max(1, round(fps * step_dt))
            represented_dt = stride / fps
            if abs(represented_dt - step_dt) > max(1.0e-6, 0.05 * step_dt):
                raise ValueError(
                    f"{path}: expert fps={fps:g} cannot represent environment step_dt={step_dt:g} "
                    f"within 5% (nearest stride={stride}, dt={represented_dt:g})"
                )

            available_names = np.asarray(data["body_names"]).astype(str).tolist()
            required_names = [anchor_body_name, *selected_body_names]
            missing_names = [name for name in required_names if name not in available_names]
            if missing_names:
                raise ValueError(f"{path} is missing AMP bodies: {missing_names}")
            anchor_index = available_names.index(anchor_body_name)
            body_indices = [available_names.index(name) for name in selected_body_names]

            arrays: dict[str, torch.Tensor] = {}
            for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                value = np.asarray(data[key])
                if value.ndim != 3 or value.shape[0] <= stride:
                    raise ValueError(f"{path}: invalid {key} shape {value.shape} for stride {stride}")
                if not np.isfinite(value).all():
                    raise ValueError(f"{path}: {key} contains NaN or infinity")
                arrays[key] = torch.as_tensor(value[:, body_indices], device=device, dtype=torch.float32)
            anchor_pos = torch.as_tensor(
                np.asarray(data["body_pos_w"])[:, anchor_index], device=device, dtype=torch.float32
            )
            anchor_quat = torch.as_tensor(
                np.asarray(data["body_quat_w"])[:, anchor_index], device=device, dtype=torch.float32
            )

        all_states = build_amp_state(
            arrays["body_pos_w"],
            arrays["body_quat_w"],
            arrays["body_lin_vel_w"],
            arrays["body_ang_vel_w"],
            anchor_pos,
            anchor_quat,
        )
        transitions = all_states.shape[0] - stride
        info = AmpClipInfo(str(path), all_states.shape[0], fps, stride, transitions)
        return all_states[:-stride], all_states[stride:], info

    def __len__(self) -> int:
        return self.states.shape[0]

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one expert batch, including every required-clip transition once.

        The remaining entries are sampled with replacement from all other
        clips.  The final permutation prevents the required transitions from
        occupying fixed locations in an effective or micro batch.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.required_states is None:
            indices = torch.randint(len(self), (batch_size,), device=self.device)
            return self.states[indices], self.next_states[indices]

        required_count = self.required_transition_count
        if batch_size < required_count:
            raise ValueError(
                f"AMP expert batch_size={batch_size} cannot contain all {required_count} "
                f"transitions from required clip {self.required_clip_name}."
            )
        random_count = batch_size - required_count
        if random_count:
            indices = torch.randint(
                self.random_states.shape[0], (random_count,), device=self.device
            )
            states = torch.cat((self.required_states, self.random_states[indices]), dim=0)
            next_states = torch.cat(
                (self.required_next_states, self.random_next_states[indices]), dim=0
            )
        else:
            states = self.required_states
            next_states = self.required_next_states
        permutation = torch.randperm(batch_size, device=self.device)
        return states[permutation], next_states[permutation]

    @property
    def required_transition_count(self) -> int:
        if self.required_states is None:
            return 0
        return int(self.required_states.shape[0])

    @property
    def random_transition_count(self) -> int:
        return int(self.random_states.shape[0])
