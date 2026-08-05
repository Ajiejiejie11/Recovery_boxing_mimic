"""Shared online/offline AMP state construction.

Quaternions use Isaac Lab's ``(w, x, y, z)`` convention.  Each selected body
contributes 15 values: anchor-relative position, anchor-relative 6-D rotation,
body-local linear velocity, and body-local angular velocity.  Removing global
translation and heading makes the discriminator judge motion style rather than
where the recovery happens in the scene.
"""

from __future__ import annotations

import torch


AMP_FEATURES_PER_BODY = 15


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized-or-nearly-normalized ``wxyz`` quaternions to matrices."""
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion [..., 4], got {tuple(quaternion.shape)}")
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    w, x, y, z = quaternion.unbind(dim=-1)
    two = 2.0
    return torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1.0 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def build_amp_state(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    anchor_pos_w: torch.Tensor,
    anchor_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Build one AMP state for each batch item.

    Args:
        body_*: Tensors shaped ``[batch, bodies, ...]`` in world coordinates.
        anchor_*: Anchor pose tensors shaped ``[batch, ...]`` in world coordinates.

    Returns:
        A tensor shaped ``[batch, bodies * 15]``.
    """
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"Expected body positions [B, N, 3], got {tuple(body_pos_w.shape)}")
    expected_prefix = body_pos_w.shape[:2]
    expected_shapes = {
        "body_quat_w": expected_prefix + (4,),
        "body_lin_vel_w": expected_prefix + (3,),
        "body_ang_vel_w": expected_prefix + (3,),
        "anchor_pos_w": (body_pos_w.shape[0], 3),
        "anchor_quat_w": (body_pos_w.shape[0], 4),
    }
    actual = {
        "body_quat_w": body_quat_w.shape,
        "body_lin_vel_w": body_lin_vel_w.shape,
        "body_ang_vel_w": body_ang_vel_w.shape,
        "anchor_pos_w": anchor_pos_w.shape,
        "anchor_quat_w": anchor_quat_w.shape,
    }
    for name, shape in expected_shapes.items():
        if tuple(actual[name]) != tuple(shape):
            raise ValueError(f"Expected {name} shape {shape}, got {tuple(actual[name])}")

    body_rotation_w = quaternion_to_matrix(body_quat_w)
    anchor_rotation_w = quaternion_to_matrix(anchor_quat_w)
    world_to_anchor = anchor_rotation_w.transpose(-1, -2)
    world_to_body = body_rotation_w.transpose(-1, -2)

    relative_position = torch.matmul(
        world_to_anchor[:, None], (body_pos_w - anchor_pos_w[:, None]).unsqueeze(-1)
    ).squeeze(-1)
    relative_rotation = torch.matmul(world_to_anchor[:, None], body_rotation_w)
    # Store the first two columns contiguously: [column_0, column_1].
    rotation_6d = relative_rotation[..., :, :2].transpose(-1, -2).reshape(*expected_prefix, 6)
    local_linear_velocity = torch.matmul(world_to_body, body_lin_vel_w.unsqueeze(-1)).squeeze(-1)
    local_angular_velocity = torch.matmul(world_to_body, body_ang_vel_w.unsqueeze(-1)).squeeze(-1)

    features = torch.cat(
        (relative_position, rotation_6d, local_linear_velocity, local_angular_velocity), dim=-1
    )
    return features.reshape(body_pos_w.shape[0], -1)
