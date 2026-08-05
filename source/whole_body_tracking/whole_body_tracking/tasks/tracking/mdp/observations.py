from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from whole_body_tracking.amp.features import build_amp_state
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def recovery_state_privileged(
    env: ManagerBasedEnv,
    command_name: str,
) -> torch.Tensor:
    """Training-only recovery state for the shared critic (four scalars).

    The actor deliberately does not receive these values.  It distinguishes a
    fall from deployable proprioception and projected gravity, while the critic
    may use exact phase, window progress, torso-link height, and stable-feet
    state to reduce value aliasing between the two reward regimes.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    torso_height = command.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.stack(
        (
            command.recovery_active.float(),
            command.recovery_progress,
            torso_height,
            command.recovery_feet_stable.float(),
        ),
        dim=-1,
    )


def robot_amp_state(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Whole-body state used only by the external recovery AMP sidecar.

    The term is a separate observation group, so it never changes the Actor or
    critic input dimensions.  Body selection and ordering come from the motion
    command and are reused by the expert loader in the runner.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = [
        index
        for index, body_name in enumerate(command.cfg.body_names)
        if body_name != command.cfg.anchor_body_name
    ]
    if not body_indices:
        raise ValueError("AMP requires at least one command body other than the anchor body.")
    return build_amp_state(
        command.robot_body_pos_w[:, body_indices],
        command.robot_body_quat_w[:, body_indices],
        command.robot_body_lin_vel_w[:, body_indices],
        command.robot_body_ang_vel_w[:, body_indices],
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
    )
