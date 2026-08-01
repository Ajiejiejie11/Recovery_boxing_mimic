from pathlib import Path

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.z1 import Z1_ACTION_SCALE, Z1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.z1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


DATASET_DIR = Path(__file__).resolve().parents[4] / "datasets"


@configclass
class Z1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = Z1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Filtered one-to-many sensors isolate robot-on-robot contact from the
        # ground contacts required during recovery.  Hand-to-hand and
        # hand-to-torso pairs are deliberately absent because the recovery
        # target is a crossed-hands guard pose.
        leg_collision_targets = [
            "{ENV_REGEX_NS}/Robot/pelvis",
            "{ENV_REGEX_NS}/Robot/left_hip_roll_link",
            "{ENV_REGEX_NS}/Robot/left_knee_link",
            "{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
            "{ENV_REGEX_NS}/Robot/right_hip_roll_link",
            "{ENV_REGEX_NS}/Robot/right_knee_link",
            "{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        ]
        self.scene.self_collision_left_wrist = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link",
            filter_prim_paths_expr=leg_collision_targets,
        )
        self.scene.self_collision_right_wrist = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
            filter_prim_paths_expr=leg_collision_targets,
        )
        self.scene.self_collision_left_elbow = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_elbow_link",
            filter_prim_paths_expr=leg_collision_targets,
        )
        self.scene.self_collision_right_elbow = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_elbow_link",
            filter_prim_paths_expr=leg_collision_targets,
        )
        self.scene.self_collision_left_knee = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_knee_link",
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/Robot/right_knee_link",
                "{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
            ],
        )
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
        # Global reset allocation: 40% recovery and 60% reference tracking.
        # Within the tracking subset, coverage/hard/soft remain
        # 0.625/0.125/0.25, corresponding to 37.5%/7.5%/15% globally.
        self.commands.motion.recovery_fraction = 0.40
        self.commands.motion.recovery_target_file = str(
            DATASET_DIR / "recovery_targets/boxing_walk_001_get_ready_370_530.npz"
        )
        self.commands.motion.recovery_target_frame = 64
        self.commands.motion.recovery_reset_file = str(
            DATASET_DIR / "fall_recovety/prepare_stand_slice_06/train_npz"
        )
        # The source clips were already sliced at 0.6 m.  Their converted torso
        # heights top out at 0.6194 m, so 0.62 includes the full curated reset
        # distribution while the 0.75 m torso-link success threshold stays
        # safely above it.
        self.commands.motion.recovery_reset_max_height = 0.62
        self.commands.motion.recovery_reset_max_uprightness = 1.0
        self.commands.motion.recovery_duration_s = 6.0
        self.commands.motion.coverage_sampling_fraction = 0.625
        self.commands.motion.hard_failure_replay_fraction = 0.125
        self.commands.motion.tracking_error_replay_fraction = 0.25

        # Recovery task group: uprightness is signed, so inverted poses are
        # penalized rather than entering a zero-gradient region. Feet and the
        # two reference bridges are smoothly enabled only near standing to
        # preserve valid get-up motions.
        self.rewards.recovery_upright = RewTerm(
            func=mdp.recovery_upright_reward, weight=1.5, params={"command_name": "motion"}
        )
        self.rewards.recovery_height = RewTerm(
            func=mdp.recovery_height_reward,
            weight=3.5,
            params={"command_name": "motion", "target_height": 0.75},
        )
        late_recovery_gate = {
            "min_height": 0.55,
            "full_height": 0.70,
            "min_uprightness": 0.30,
            "full_uprightness": 0.80,
        }
        self.rewards.recovery_feet_stable = RewTerm(
            func=mdp.recovery_feet_stable_reward,
            weight=0.10,
            params={"command_name": "motion", **late_recovery_gate},
        )
        self.rewards.recovery_lower_body_reference = RewTerm(
            func=mdp.recovery_lower_body_reference_reward,
            weight=0.15,
            params={
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_hip_.*_joint",
                        ".*_knee_joint",
                        ".*_ankle_.*_joint",
                        "waist_yaw_joint",
                    ],
                ),
                "std": 0.50,
                **late_recovery_gate,
            },
        )
        self.rewards.recovery_torso_reference = RewTerm(
            func=mdp.recovery_torso_reference_reward,
            weight=0.15,
            params={"command_name": "motion", "std": 0.40, **late_recovery_gate},
        )

        # Entry: torso <0.50 m OR tilt >70 deg. Exit: torso >=0.75 m,
        # uprightness >=0.85, and both feet stable. The command term owns the
        # six-second phase-local counter and returns only failure/immediate done.
        self.terminations.recovery_state = DoneTerm(
            func=mdp.update_recovery_state_and_check_termination,
            params={
                "command_name": "motion",
                # Keep this below the lowest valid boxing crouch (~0.546 m,
                # before reset noise) so a legitimate pose does not open the
                # recovery gate.
                "fall_height_threshold": 0.50,
                "fall_upright_threshold": 0.342,
                "stand_height_threshold": 0.75,
                "stand_upright_threshold": 0.85,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "min_contact_time": 0.15,
                "max_planar_speed": 0.20,
            },
        )
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
            "x": (-0.15, 0.15),
            "y": (-0.15, 0.15),
            "z": (-0.05, 0.05),
            "roll": (-0.15, 0.15),
            "pitch": (-0.15, 0.15),
            "yaw": (-0.20, 0.20),
        }

        # The longest source clip is about 14 s; allow 20 s of tracking. This
        # budget pauses in recovery and restarts with the fresh post-recovery
        # boxing reference.
        self.episode_length_s = 20.0

        # Hand/ankle tracking error is a soft failure. Hard failure is reserved for
        # torso height/orientation loss (falling) and other true terminations.
        self.terminations.ee_ankle_pos = None
        self.terminations.ee_hand_pos = None

        # Hands and forearms are legitimate supports during a get-up.  Keep this
        # contact regularizer for reference tracking only; joint/action/torque
        # regularization remains shared by both tasks.
        self.rewards.undesired_contacts.func = mdp.tracking_undesired_contacts
        self.rewards.undesired_contacts.params["command_name"] = "motion"
        # The strongest monitored self-contact is mapped to [0, 1]. At weight
        # -0.5 its worst per-second contribution is one eighth of the 4.0 dense
        # recovery reward and one tenth of the 5.0 tracking reward.
        self.rewards.self_collision = RewTerm(
            func=mdp.self_collision_penalty,
            weight=-0.5,
            params={
                "sensor_names": [
                    "self_collision_left_wrist",
                    "self_collision_right_wrist",
                    "self_collision_left_elbow",
                    "self_collision_right_elbow",
                    "self_collision_left_knee",
                ],
                "force_threshold": 5.0,
                "saturation_force": 40.0,
            },
        )


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
