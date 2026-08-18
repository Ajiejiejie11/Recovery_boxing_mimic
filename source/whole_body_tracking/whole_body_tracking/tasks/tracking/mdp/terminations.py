from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes, _recovery_feet_stable


def motion_complete(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """End an episode only after the sampled reference reaches its source motion boundary."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.motion_complete


def tracking_motion_complete(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.motion_complete & command.tracking_env_mask


def tracking_phase_timeout(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Time out tracking and task without consuming their budget during recovery.

    A freshly sampled boxing reference or velocity command gets a fresh budget.
    Tracking and task are mutually exclusive, so their summed step counters equal
    the total non-recovery time. Recovery has its own terminated six-second
    deadline in the state machine below.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    non_recovery_steps = command.tracking_steps + command.task_steps
    return ~command.recovery_active & (non_recovery_steps >= env.max_episode_length)


def update_recovery_state_and_check_termination(
    env: ManagerBasedRLEnv,
    command_name: str,
    fall_height_threshold: float,
    fall_upright_threshold: float,
    stand_height_threshold: float,
    stand_upright_threshold: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_contact_time: float,
    max_planar_speed: float,
) -> torch.Tensor:
    """Update the recovery gate, then return immediate/deadline terminations.

    Entry uses a permissive OR predicate (low torso or severe tilt); exit uses a
    stricter AND predicate plus stable two-foot contact.  The resulting
    hysteresis prevents phase chatter near one threshold.  Height and contacts
    are training-only state used by the environment/critic, never actor inputs.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    height = command.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
    gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, command.robot.data.GRAVITY_VEC_W)
    uprightness = -gravity_b[:, 2]
    fallen = (height < fall_height_threshold) | (uprightness < fall_upright_threshold)
    feet_stable = _recovery_feet_stable(
        env, asset_cfg, sensor_cfg, min_contact_time, max_planar_speed
    )
    standing = (
        (height >= stand_height_threshold)
        & (uprightness >= stand_upright_threshold)
        & feet_stable
    )
    return command.update_recovery_state(fallen, standing, height, uprightness, feet_stable)


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def tracking_bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    bad_height = bad_anchor_pos_z_only(env, command_name, threshold)
    downward_failure = command.robot_anchor_pos_w[:, 2] < command.anchor_pos_w[:, 2]
    # In delayed slots, a downward collapse must reach the absolute physical
    # fall predicate instead of being reset early by reference-relative height
    # error. Ordinary slots and upward deviations retain the legacy behavior.
    delayed_downward = command.delay_env_mask & downward_failure
    return bad_height & command.tracking_env_mask & ~delayed_downward


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # print("tar :", command.anchor_pos_w[:, -1])
    # print(command.robot_anchor_pos_w[:, -1])
    # print("error ###########", torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]))
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def tracking_bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    return bad_anchor_ori(env, asset_cfg, command_name, threshold) & env.command_manager.get_term(command_name).tracking_env_mask


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)
