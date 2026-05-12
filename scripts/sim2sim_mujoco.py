"""Run a magicbot-z1-mimic (BeyondMimic-derived) Z1 policy.onnx in MuJoCo (sim-to-sim).

The script reproduces the same observation layout exported from Isaac Lab:

    obs (124,) = [ command (46) | motion_anchor_ori_b (6) | base_ang_vel (3) |
                   joint_pos_rel (23) | joint_vel (23) | actions (23) ]

It then turns the policy output into joint torques via an emulated PD controller
that matches `Z1FlatPPORunnerCfg`'s implicit actuator gains, applies them through
MuJoCo's `<motor>` actuators, and steps a viewer.

Usage:
    python scripts/sim2sim_mujoco.py \\
        --xml source/whole_body_tracking/whole_body_tracking/assets/magicbot-z1_description/mjcf/MagicBotZ1_23dof.xml \\
        --policy logs/rsl_rl/z1_flat/2000iters/exported/policy.onnx \\
        --motion source/whole_body_tracking/whole_body_tracking/datasets/simple_dance.npz
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnx
import onnxruntime as ort


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def parse_csv_list(s: str, dtype=str):
    return [dtype(x) for x in s.split(",") if x != ""]


def load_policy_metadata(onnx_path: str) -> dict:
    m = onnx.load(onnx_path)
    md = {p.key: p.value for p in m.metadata_props}
    return md


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two scalar-first quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Scalar-first quaternion → 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v from world frame into the body frame defined by q (scalar-first)."""
    return quat_to_rotmat(quat_conjugate(q)) @ v


def quat_relative(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Quaternion that rotates frame `from` to frame `to`: q_rel = conj(q_from) * q_to."""
    return quat_mul(quat_conjugate(q_from), q_to)


def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """First two columns of a rotation matrix, flattened (Isaac Lab 6D convention)."""
    return R[:, :2].reshape(-1)


def yaw_from_quat(q: np.ndarray) -> float:
    """Extract the Z-axis yaw (rad) from a scalar-first quaternion (w, x, y, z)."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_quat(yaw: float) -> np.ndarray:
    """Scalar-first quaternion representing a pure yaw rotation around Z."""
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)


# ----------------------------------------------------------------------------
# Motion loader
# ----------------------------------------------------------------------------


class Motion:
    def __init__(self, npz_path: str, anchor_body_index: int):
        data = np.load(npz_path)
        self.fps = int(np.atleast_1d(data["fps"])[0])
        self.joint_pos = data["joint_pos"].astype(np.float32)  # (T, 23) - Isaac joint order
        self.joint_vel = data["joint_vel"].astype(np.float32)  # (T, 23)
        self.body_pos_w = data["body_pos_w"].astype(np.float32)  # (T, N_body, 3) - Isaac body order
        self.body_quat_w = data["body_quat_w"].astype(np.float32)  # (T, N_body, 4)
        if "body_lin_vel_w" in data:
            self.body_lin_vel_w = data["body_lin_vel_w"].astype(np.float32)
            self.body_ang_vel_w = data["body_ang_vel_w"].astype(np.float32)
        else:
            self.body_lin_vel_w = None
            self.body_ang_vel_w = None
        self.T = self.joint_pos.shape[0]
        self.anchor_idx = anchor_body_index  # torso_link index in body axis (Isaac order)
        # Index 0 of body axis is the floating base (pelvis).
        self.pelvis_idx = 0

        if not (0 <= self.anchor_idx < self.body_pos_w.shape[1]):
            raise ValueError(
                f"anchor_body_index={self.anchor_idx} out of range "
                f"(npz has {self.body_pos_w.shape[1]} bodies)"
            )

    def pelvis_pos_at(self, t: int) -> np.ndarray:
        return self.body_pos_w[t % self.T, self.pelvis_idx]

    def pelvis_quat_at(self, t: int) -> np.ndarray:
        return self.body_quat_w[t % self.T, self.pelvis_idx]

    def pelvis_lin_vel_at(self, t: int) -> np.ndarray:
        return self.body_lin_vel_w[t % self.T, self.pelvis_idx] if self.body_lin_vel_w is not None else np.zeros(3)

    def pelvis_ang_vel_at(self, t: int) -> np.ndarray:
        return self.body_ang_vel_w[t % self.T, self.pelvis_idx] if self.body_ang_vel_w is not None else np.zeros(3)

    def __len__(self) -> int:
        return self.T

    def joint_pos_at(self, t: int) -> np.ndarray:
        return self.joint_pos[t % self.T]

    def joint_vel_at(self, t: int) -> np.ndarray:
        return self.joint_vel[t % self.T]

    def anchor_quat_at(self, t: int) -> np.ndarray:
        return self.body_quat_w[t % self.T, self.anchor_idx]

    def anchor_pos_at(self, t: int) -> np.ndarray:
        return self.body_pos_w[t % self.T, self.anchor_idx]


# ----------------------------------------------------------------------------
# Sim-to-sim runner
# ----------------------------------------------------------------------------


class MuJoCoZ1Sim2Sim:
    def __init__(self, args):
        # ---- load MuJoCo model -------------------------------------------------
        self.model = mujoco.MjModel.from_xml_path(args.xml)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # ---- load policy + metadata --------------------------------------------
        md = load_policy_metadata(args.policy)
        self.policy_joint_names = parse_csv_list(md["joint_names"])
        self.action_scale = np.array(parse_csv_list(md["action_scale"], float), dtype=np.float32)
        self.kp = np.array(parse_csv_list(md["joint_stiffness"], float), dtype=np.float32)
        self.kd = np.array(parse_csv_list(md["joint_damping"], float), dtype=np.float32)
        self.default_q = np.array(parse_csv_list(md["default_joint_pos"], float), dtype=np.float32)
        self.anchor_body_name = md["anchor_body_name"].strip()
        assert len(self.policy_joint_names) == 23, "policy must use 23 joints"

        self.sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
        in_shape = self.sess.get_inputs()[0].shape
        print(f"[INFO] ONNX input: {self.sess.get_inputs()[0].name} {in_shape}")

        # ---- index maps --------------------------------------------------------
        # MuJoCo joint id for each policy joint
        self.mj_joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.policy_joint_names],
            dtype=np.int32,
        )
        if np.any(self.mj_joint_ids < 0):
            missing = [n for n, i in zip(self.policy_joint_names, self.mj_joint_ids) if i < 0]
            raise RuntimeError(f"Joints missing in MJCF: {missing}")
        # qpos/qvel addresses for each policy joint (hinge joints: 1 dof each)
        self.qpos_addrs = np.array([self.model.jnt_qposadr[j] for j in self.mj_joint_ids], dtype=np.int32)
        self.qvel_addrs = np.array([self.model.jnt_dofadr[j] for j in self.mj_joint_ids], dtype=np.int32)

        # MuJoCo actuator id for each policy joint
        self.mj_actuator_ids = np.zeros(23, dtype=np.int32)
        for i, jid in enumerate(self.mj_joint_ids):
            actuator_idx = -1
            for a in range(self.model.nu):
                if self.model.actuator_trnid[a, 0] == jid:
                    actuator_idx = a
                    break
            if actuator_idx < 0:
                raise RuntimeError(f"No <motor> actuator drives joint {self.policy_joint_names[i]}")
            self.mj_actuator_ids[i] = actuator_idx

        # Anchor body id (torso_link)
        self.anchor_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.anchor_body_name)
        if self.anchor_bid < 0:
            raise RuntimeError(f"anchor body '{self.anchor_body_name}' not in MJCF")
        # Pelvis (= floating-base body) for base ang vel
        self.pelvis_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        # Floor geom (used for penetration tracking through the contact solver).
        self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # ---- timing ------------------------------------------------------------
        self.policy_dt = args.policy_dt
        self.sim_dt = args.sim_dt or self.model.opt.timestep
        if abs(self.model.opt.timestep - self.sim_dt) > 1e-9:
            self.model.opt.timestep = self.sim_dt
        self.decimation = max(1, int(round(self.policy_dt / self.sim_dt)))
        print(f"[INFO] policy_dt={self.policy_dt}s, sim_dt={self.sim_dt}s, decimation={self.decimation}")

        # ---- motion ------------------------------------------------------------
        self.motion = Motion(args.motion, anchor_body_index=args.motion_anchor_idx)
        self.motion_dt = 1.0 / self.motion.fps
        self.motion_steps_per_policy = max(1, int(round(self.policy_dt / self.motion_dt)))
        print(
            f"[INFO] motion fps={self.motion.fps}, T={self.motion.T}, anchor_idx_in_npz={self.motion.anchor_idx}, "
            f"motion_steps_per_policy={self.motion_steps_per_policy}"
        )

        # ---- buffers -----------------------------------------------------------
        self.last_action = np.zeros(23, dtype=np.float32)
        self.motion_t = 0
        self.render = args.render
        self.real_time = args.real_time
        self.zero_motion = args.zero_motion
        self.init_at_motion = args.init_at_motion
        self.debug_steps = args.debug_steps
        self.torque_clip = args.torque_clip
        self.action_clip = args.action_clip
        self.hold_default = args.hold_default

    # ------------------------------------------------------------------ obs ---
    def build_obs(self) -> np.ndarray:
        # joint state in Isaac (policy) order
        q = self.data.qpos[self.qpos_addrs].astype(np.float32)
        qd = self.data.qvel[self.qvel_addrs].astype(np.float32)

        # base ang vel in pelvis BODY frame.
        # MuJoCo free-joint convention: qvel[3:6] of the floating base IS already in body frame.
        base_ang_vel_b = self.data.qvel[3:6].astype(np.float32)

        # motion_anchor_ori_b: 6D rep of R_robot_anchor^T @ R_motion_anchor
        # (i.e. motion-anchor orientation expressed in robot-anchor frame)
        robot_anchor_quat = self.data.xquat[self.anchor_bid].astype(np.float32)
        if self.zero_motion:
            motion_anchor_quat = robot_anchor_quat.copy()  # identity relative
        else:
            motion_anchor_quat = self.motion.anchor_quat_at(self.motion_t).astype(np.float32)
        q_rel = quat_relative(robot_anchor_quat, motion_anchor_quat)
        R_rel = quat_to_rotmat(q_rel)
        motion_anchor_ori_b = rotmat_to_6d(R_rel).astype(np.float32)

        # command = concat(motion.joint_pos, motion.joint_vel)
        if self.zero_motion:
            m_jp = self.default_q.copy()
            m_jv = np.zeros(23, dtype=np.float32)
        else:
            m_jp = self.motion.joint_pos_at(self.motion_t).astype(np.float32)
            m_jv = self.motion.joint_vel_at(self.motion_t).astype(np.float32)
        command = np.concatenate([m_jp, m_jv], axis=0)

        # joint_pos_rel = joint_pos - default_joint_pos
        joint_pos_rel = q - self.default_q

        obs = np.concatenate(
            [command, motion_anchor_ori_b, base_ang_vel_b, joint_pos_rel, qd, self.last_action],
            axis=0,
        ).astype(np.float32)
        assert obs.shape == (124,), f"got obs.shape={obs.shape}"
        return obs

    # ------------------------------------------------------------------ act ---
    def set_action(self, action: np.ndarray):
        """Latch the new policy action (sets per-joint target_q, used by PD at sim rate)."""
        if self.action_clip is not None:
            action = np.clip(action, -self.action_clip, self.action_clip)
        self.last_action = action.astype(np.float32)
        self.target_q = self.default_q + self.action_scale * self.last_action

    def apply_pd(self) -> np.ndarray:
        """Compute PD torques at the current sim state and push to data.ctrl.

        Mirrors Isaac Lab's ImplicitActuator which runs inside PhysX at every sim step
        (not only at policy decimation).
        """
        q = self.data.qpos[self.qpos_addrs].astype(np.float32)
        qd = self.data.qvel[self.qvel_addrs].astype(np.float32)
        tau = self.kp * (self.target_q - q) - self.kd * qd
        if self.torque_clip is not None:
            tau = np.clip(tau, -self.torque_clip, self.torque_clip)
        self.data.ctrl[:] = 0.0  # head_actuator stays at 0
        for i, aid in enumerate(self.mj_actuator_ids):
            self.data.ctrl[aid] = tau[i]
        return tau

    # ------------------------------------------------------------------ dbg ---
    def debug_print(self, step: int, obs: np.ndarray, action: np.ndarray, tau: np.ndarray):
        np.set_printoptions(precision=3, suppress=True, linewidth=160)
        cmd_jp = obs[0:23]
        cmd_jv = obs[23:46]
        anchor6d = obs[46:52]
        bav = obs[52:55]
        jp_rel = obs[55:78]
        jv = obs[78:101]
        last_act = obs[101:124]
        q_real = self.data.qpos[self.qpos_addrs]
        qd_real = self.data.qvel[self.qvel_addrs]
        print(f"\n========== step {step}  motion_t={self.motion_t} ==========")
        print(f"pelvis pos     : {self.data.qpos[:3]}")
        print(f"pelvis quat    : {self.data.qpos[3:7]}")
        print(f"pelvis lin vel : {self.data.qvel[:3]}")
        print(f"pelvis ang vel : {self.data.qvel[3:6]}  (body frame)")
        print(f"q (Isaac order): {q_real}")
        print(f"qd(Isaac order): {qd_real}")
        print(f"--- obs ---")
        print(f"cmd joint_pos  : {cmd_jp}")
        print(f"cmd joint_vel  : {cmd_jv}")
        print(f"anchor_ori_6d  : {anchor6d}")
        print(f"base_ang_vel_b : {bav}")
        print(f"joint_pos_rel  : {jp_rel}")
        print(f"joint_vel      : {jv}")
        print(f"last_action    : {last_act}")
        print(f"--- io ---")
        print(f"action (new)   : {action}")
        print(f"action |max|   : {np.abs(action).max():.3f}  joint={self.policy_joint_names[int(np.argmax(np.abs(action)))]}")
        print(f"target_q       : {self.target_q}")
        print(f"torque         : {tau}")
        print(f"torque |max|   : {np.abs(tau).max():.2f}  joint={self.policy_joint_names[int(np.argmax(np.abs(tau)))]}")

    # ------------------------------------------------------------------ run ---
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        if self.init_at_motion and not self.zero_motion:
            # full match to motion frame 0: joints + pelvis pose
            jp0 = self.motion.joint_pos_at(0)
            for i, qa in enumerate(self.qpos_addrs):
                self.data.qpos[qa] = jp0[i]
            self.data.qpos[0:3] = self.motion.pelvis_pos_at(0)
            self.data.qpos[3:7] = self.motion.pelvis_quat_at(0)
            self.data.qpos[2] = max(self.data.qpos[2], 0.30)  # avoid ground penetration
        else:
            # default joint pose; pelvis x/y/z from XML, but inherit yaw from motion[0]
            for i, qa in enumerate(self.qpos_addrs):
                self.data.qpos[qa] = self.default_q[i]
            if not self.zero_motion:
                yaw0 = yaw_from_quat(self.motion.pelvis_quat_at(0))
                self.data.qpos[3:7] = yaw_quat(yaw0)
                print(f"[INIT] inherited yaw from motion[0] = {np.rad2deg(yaw0):+.2f} deg")
        mujoco.mj_forward(self.model, self.data)
        self.last_action[:] = 0.0
        self.motion_t = 0
        self.target_q = self.default_q.copy()

    def run(self, max_seconds: float | None):
        self.reset()
        step_count = 0
        start_time = time.time()
        next_real_time = start_time
        max_pen_depth = 0.0  # deepest penetration of any robot geom into the floor (m)

        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        try:
            while True:
                # ---- policy step -----------------------------------------------
                if self.hold_default:
                    obs = np.zeros(124, dtype=np.float32)
                    action = np.zeros(23, dtype=np.float32)
                else:
                    obs = self.build_obs()
                    action = self.sess.run(None, {"obs": obs.reshape(1, -1)})[0].reshape(-1)
                self.set_action(action)
                tau = self.apply_pd()

                if step_count < self.debug_steps:
                    self.debug_print(step_count, obs, action, tau)

                # ---- physics decimation: PD runs at every sim step -------------
                for _ in range(self.decimation):
                    self.apply_pd()
                    mujoco.mj_step(self.model, self.data)
                    if self.viewer is not None:
                        self.viewer.sync()
                    # track penetration via active contacts with the floor
                    for ci in range(self.data.ncon):
                        c = self.data.contact[ci]
                        if self.floor_gid in (int(c.geom1), int(c.geom2)) and c.dist < 0:
                            max_pen_depth = max(max_pen_depth, float(-c.dist))

                # ---- advance motion --------------------------------------------
                self.motion_t = (self.motion_t + self.motion_steps_per_policy) % self.motion.T
                step_count += 1

                # ---- timing ----------------------------------------------------
                if self.real_time:
                    next_real_time += self.policy_dt
                    sleep = next_real_time - time.time()
                    if sleep > 0:
                        time.sleep(sleep)

                if max_seconds is not None and (time.time() - start_time) > max_seconds:
                    break
                if self.viewer is not None and not self.viewer.is_running():
                    break
        finally:
            if self.viewer is not None:
                self.viewer.close()

        elapsed = time.time() - start_time
        print(
            f"[DONE] ran {step_count} policy steps in {elapsed:.2f}s "
            f"({step_count / elapsed:.1f} steps/s, sim_time={step_count * self.policy_dt:.2f}s)"
        )
        print(
            f"[DONE] final pelvis pos={self.data.qpos[:3]}  "
            f"quat={self.data.qpos[3:7]}  ang_vel={self.data.qvel[3:6]}"
        )
        z = float(self.data.qpos[2])
        if z < 0.3:
            print(f"[WARN] pelvis z={z:.3f}m  -> robot has fallen.")
        elif z > 1.2:
            print(f"[WARN] pelvis z={z:.3f}m  -> robot has flown/jumped.")
        else:
            print(f"[OK] pelvis z={z:.3f}m  -> robot upright-ish.")
        print(f"[CONTACT] max penetration into floor over the run = {max_pen_depth*1000:.2f}mm")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Sim-to-sim playback of a magicbot-z1-mimic Z1 policy in MuJoCo")
    parser.add_argument(
        "--xml",
        type=str,
        default="source/whole_body_tracking/whole_body_tracking/assets/magicbot-z1_description/mjcf/MagicBotZ1_23dof.xml",
        help="Path to the MuJoCo MJCF model.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="logs/rsl_rl/z1_flat/2000iters/exported/policy.onnx",
        help="Path to the exported policy.onnx (must contain Isaac-Lab metadata).",
    )
    parser.add_argument(
        "--motion",
        type=str,
        default="source/whole_body_tracking/whole_body_tracking/datasets/simple_dance.npz",
        help="Path to the motion .npz used as the reference command.",
    )
    parser.add_argument(
        "--motion_anchor_idx",
        type=int,
        default=3,
        help=(
            "Index of the anchor body (torso_link) along the body axis of the .npz. "
            "Defaults to 3 which matches the Isaac-Lab L/R-paired BFS body order for Z1."
        ),
    )
    parser.add_argument("--sim_dt", type=float, default=None, help="MuJoCo timestep (default: from XML).")
    parser.add_argument("--policy_dt", type=float, default=0.02, help="Policy control dt (default 50Hz).")
    parser.add_argument("--max_seconds", type=float, default=None, help="Optional time budget (wall-clock).")
    parser.add_argument("--no_render", dest="render", action="store_false", help="Disable the MuJoCo viewer.")
    parser.add_argument("--no_real_time", dest="real_time", action="store_false", help="Run as fast as possible.")
    parser.add_argument(
        "--zero_motion",
        action="store_true",
        help="Override motion command to default-pose stand-still. Useful to test if the policy can balance.",
    )
    parser.add_argument(
        "--init_at_motion",
        action="store_true",
        help="Initialize robot joints at motion[t=0] instead of the default pose.",
    )
    parser.add_argument(
        "--debug_steps",
        type=int,
        default=0,
        help="Print full obs / action / torque vectors for the first N policy steps.",
    )
    parser.add_argument(
        "--torque_clip",
        type=float,
        default=None,
        help="Clip joint torques to +/- this value before sending to MuJoCo (safety).",
    )
    parser.add_argument(
        "--action_clip",
        type=float,
        default=None,
        help="Clip policy action to +/- this value (default off; Isaac uses ~100 internally).",
    )
    parser.add_argument(
        "--hold_default",
        action="store_true",
        help="Bypass the policy: just PD-hold the default pose. Sanity-check that PD is stable.",
    )
    parser.set_defaults(render=True, real_time=True)
    args = parser.parse_args()

    sim = MuJoCoZ1Sim2Sim(args)
    sim.run(args.max_seconds)


if __name__ == "__main__":
    main()
