from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _tracking_mask(command: MotionCommand) -> torch.Tensor:
    return command.tracking_env_mask.float()


def _recovery_mask(command: MotionCommand) -> torch.Tensor:
    return command.recovery_active.float()


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2) * _tracking_mask(command)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2) * _tracking_mask(command)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2) * _tracking_mask(command)

def motion_relative_body_position_z_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.square(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.exp(-error.mean(-1) / std**2) * _tracking_mask(command)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2) * _tracking_mask(command)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2) * _tracking_mask(command)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2) * _tracking_mask(command)

def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def joint_pos_track_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.mean(
        torch.square(command.robot_joint_pos - command.joint_pos), dim=-1
    )
    return torch.exp(-error / std**2)

def foot_slip_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    command_name: str | None = None,
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    reward = torch.sum(reward, dim=1)
    if command_name is not None:
        command: MotionCommand = env.command_manager.get_term(command_name)
        reward = reward * _tracking_mask(command)
    return reward


def tracking_undesired_contacts(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float, command_name: str
) -> torch.Tensor:
    """Usual non-foot/non-hand contact penalty, disabled while using limbs to rise."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    is_contact = torch.max(torch.norm(forces, dim=-1), dim=1)[0] > threshold
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.sum(is_contact, dim=-1) * _tracking_mask(command)


def self_collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    force_threshold: float,
    saturation_force: float,
) -> torch.Tensor:
    """Penalize filtered robot-on-robot contacts, bounded independently of pair count.

    Each configured contact sensor has exactly one source body and filters only
    non-adjacent robot bodies.  Taking the strongest contact instead of summing
    all pairs keeps this shared regularizer in ``[0, 1]`` and prevents its scale
    from changing when more collision pairs are monitored.
    """
    if saturation_force <= force_threshold:
        raise ValueError("saturation_force must be greater than force_threshold")

    penalty = torch.zeros(env.num_envs, device=env.device)
    force_range = saturation_force - force_threshold
    for sensor_name in sensor_names:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
        force_matrix = contact_sensor.data.force_matrix_w
        if force_matrix is None:
            raise RuntimeError(f"Contact sensor '{sensor_name}' must configure filter_prim_paths_expr")
        pair_force = torch.linalg.norm(force_matrix, dim=-1)
        pair_penalty = ((pair_force - force_threshold) / force_range).clamp(0.0, 1.0)
        penalty = torch.maximum(penalty, pair_penalty.amax(dim=(1, 2)))
    return penalty


def recovery_upright_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Signed torso-uprightness reward for the recovery-only task.

    ``uprightness`` is one when the torso is upright, zero when its vertical
    axis is horizontal, and negative when inverted. Keeping the negative half
    gives the policy a gradient away from head-down/inverted local optima.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    uprightness = command.recovery_torso_uprightness.clamp(-1.0, 1.0)
    return uprightness * _recovery_mask(command)


def recovery_height_reward(env: ManagerBasedRLEnv, command_name: str, target_height: float) -> torch.Tensor:
    """Dense torso-height reward, normalized to the standing threshold."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return (command.recovery_torso_height / target_height).clamp(0.0, 1.0) * _recovery_mask(command)


def _recovery_feet_stable(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_contact_time: float,
    max_planar_speed: float,
) -> torch.Tensor:
    """Return true when both feet have sustained contact without sliding."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    planar_speed = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    return torch.all((contact_time >= min_contact_time) & (planar_speed <= max_planar_speed), dim=-1)


def _smoothstep(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Smoothly map ``[lower, upper]`` to ``[0, 1]`` with zero endpoint slopes."""
    if upper <= lower:
        raise ValueError(f"smoothstep upper bound ({upper}) must exceed lower bound ({lower})")
    phase = ((value - lower) / (upper - lower)).clamp(0.0, 1.0)
    return phase.square() * (3.0 - 2.0 * phase)


def _late_recovery_gate(
    command: MotionCommand,
    min_height: float,
    full_height: float,
    min_uprightness: float,
    full_uprightness: float,
) -> torch.Tensor:
    """Enable terminal-shaping terms only after meaningful get-up progress."""
    height_gate = _smoothstep(command.recovery_torso_height, min_height, full_height)
    upright_gate = _smoothstep(command.recovery_torso_uprightness, min_uprightness, full_uprightness)
    return height_gate * upright_gate * _recovery_mask(command)


def recovery_feet_stable_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_height: float,
    full_height: float,
    min_uprightness: float,
    full_uprightness: float,
) -> torch.Tensor:
    """Reward stable feet near standing without rewarding stable feet while lying."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    gate = _late_recovery_gate(command, min_height, full_height, min_uprightness, full_uprightness)
    return command.recovery_feet_stable.float() * gate


def recovery_full_body_reference_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
    min_height: float,
    full_height: float,
    min_uprightness: float,
    full_uprightness: float,
) -> torch.Tensor:
    """Track the recovery stand command with every joint near standing.

    The exponential similarity is in ``(0, 1]`` and is smoothly suppressed
    before the late recovery stage, leaving the early support phase unconstrained.
    """
    if std <= 0.0:
        raise ValueError("recovery full-body reference std must be positive")
    command: MotionCommand = env.command_manager.get_term(command_name)
    joint_error = (
        command.robot_joint_pos[:, asset_cfg.joint_ids] - command.joint_pos[:, asset_cfg.joint_ids]
    )
    # Use the shortest angular distance for yaw-like joints.
    joint_error = torch.atan2(torch.sin(joint_error), torch.cos(joint_error))
    mean_square_error = joint_error.square().mean(dim=-1)
    similarity = torch.exp(-mean_square_error / std**2)
    gate = _late_recovery_gate(command, min_height, full_height, min_uprightness, full_uprightness)
    return similarity * gate


def recovery_torso_reference_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    min_height: float,
    full_height: float,
    min_uprightness: float,
    full_uprightness: float,
) -> torch.Tensor:
    """Track the default-stand torso orientation only in late recovery.

    The aligned target preserves the reference roll/pitch while matching the
    robot's current global yaw, so this term cannot make the robot turn toward
    the source clip or constrain its horizontal position and velocity.
    """
    if std <= 0.0:
        raise ValueError("recovery torso reference std must be positive")
    command: MotionCommand = env.command_manager.get_term(command_name)
    target_quat_w = command.body_quat_relative_w[:, command.motion_anchor_body_index]
    orientation_error = quat_error_magnitude(target_quat_w, command.robot_anchor_quat_w).square()
    similarity = torch.exp(-orientation_error / std**2)
    gate = _late_recovery_gate(command, min_height, full_height, min_uprightness, full_uprightness)
    return similarity * gate
