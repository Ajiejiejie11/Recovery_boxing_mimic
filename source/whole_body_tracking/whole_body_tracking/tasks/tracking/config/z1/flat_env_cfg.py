from isaaclab.utils import configclass

from whole_body_tracking.robots.z1 import Z1_ACTION_SCALE, Z1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.z1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class Z1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = Z1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = Z1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

        # Stage-2 randomization for robustness and sim-to-real transfer.
        self.commands.motion.debug_vis = False
        self.commands.motion.pose_range = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.01, 0.01),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
        }
        self.commands.motion.velocity_range = {
            "x": (-0.10, 0.10),
            "y": (-0.10, 0.10),
            "z": (-0.05, 0.05),
            "roll": (-0.10, 0.10),
            "pitch": (-0.10, 0.10),
            "yaw": (-0.10, 0.10),
        }
        self.commands.motion.joint_position_range = (-0.03, 0.03)
        self.observations.policy.enable_corruption = True

        self.events.physics_material.params["static_friction_range"] = (0.6, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.0)
        self.events.hand_physics_material.params["static_friction_range"] = (0.6, 1.2)
        self.events.hand_physics_material.params["dynamic_friction_range"] = (0.5, 1.0)
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (-0.01, 0.01)
        self.events.scale_link_mass.params["mass_distribution_params"] = (0.95, 1.05)
        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.7, 1.3)
        self.events.actuator_gains.params["damping_distribution_params"] = (0.7, 1.3)
        self.events.base_com.params["com_range"] = {
            "x": (-0.3, 0.3),
            "y": (-0.2, 0.2),
            "z": (-0.1, 0.1),
        }
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        }

        # The longest source clip is about 14 s; allow continuous traversal across all of its bins.
        self.episode_length_s = 20.0

        # Hand/ankle tracking error is a soft failure. Hard failure is reserved for
        # torso height/orientation loss (falling) and other true terminations.
        self.terminations.ee_ankle_pos = None
        self.terminations.ee_hand_pos = None


@configclass
class Z1FlatWoStateEstimationEnvCfg(Z1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class Z1FlatLowFreqEnvCfg(Z1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
