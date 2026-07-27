import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

ARMATURE_90 = 0.028637594
ARMATURE_60 = 0.01503

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_90 = ARMATURE_90 * NATURAL_FREQ**2
STIFFNESS_60 = ARMATURE_60 * NATURAL_FREQ**2

DAMPING_90 = 2.0 * DAMPING_RATIO * ARMATURE_90 * NATURAL_FREQ
DAMPING_60 = 2.0 * DAMPING_RATIO * ARMATURE_60 * NATURAL_FREQ
DAMPING_HIP_PITCH_KNEE = 5.19741
DAMPING_ANKLE = 2.5549

# Isaac Lab otherwise writes converted URDF assets to the shared
# /tmp/IsaacLab directory, which may be owned by another server user.
Z1_USD_DIR = os.environ.get(
    "WBT_USD_DIR", os.path.join("/tmp", f"magicbot-mimic-{os.getuid()}", "z1_urdf")
)

Z1_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/magicbot-z1_description/urdf/MagicBotZ1_23dof.urdf",
        usd_dir=Z1_USD_DIR,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            "left_hip_pitch_joint": -0.3,
            "left_hip_roll_joint": 0,
            "left_hip_yaw_joint": 0,
            "left_knee_joint": 0.65,
            "left_ankle_pitch_joint": -0.3,
            "left_ankle_roll_joint": -0.,

            "right_hip_pitch_joint": -0.3,
            "right_hip_roll_joint": 0,
            "right_hip_yaw_joint": 0,
            "right_knee_joint": 0.65,
            "right_ankle_pitch_joint": -0.3,
            "right_ankle_roll_joint": -0.,

            "waist_yaw_joint": 0.0,

            "left_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.15,
            "left_shoulder_yaw_joint": 0.,
            "left_elbow_joint": 1.,
            "left_wrist_yaw_joint": 0.,

            "right_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.15,
            "right_shoulder_yaw_joint": 0.,
            "right_elbow_joint": 1.,
            "right_wrist_yaw_joint": 0.,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 120,
                ".*_hip_roll_joint": 120,
                ".*_hip_yaw_joint": 120,
                ".*_knee_joint": 120,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 35,
                ".*_hip_roll_joint": 35,
                ".*_hip_yaw_joint": 35,
                ".*_knee_joint": 35,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_90,
                ".*_hip_roll_joint": STIFFNESS_90,
                ".*_hip_yaw_joint": STIFFNESS_90,
                ".*_knee_joint": STIFFNESS_90,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_HIP_PITCH_KNEE,
                ".*_hip_roll_joint": DAMPING_90,
                ".*_hip_yaw_joint": DAMPING_90,
                ".*_knee_joint": DAMPING_HIP_PITCH_KNEE,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_90 * 2,
                ".*_hip_roll_joint": ARMATURE_90 * 2,
                ".*_hip_yaw_joint": ARMATURE_90 * 2,
                ".*_knee_joint": ARMATURE_90 * 2,
            },
            # friction = {
            #     ".*_hip_pitch_joint": 0.05,
            #     ".*_hip_roll_joint": 0.05,
            #     ".*_hip_yaw_joint": 0.05,
            #     ".*_knee_joint": 0.05,
            # },
            # min_delay = {
            #     ".*_hip_pitch_joint": 0,
            #     ".*_hip_roll_joint": 0,
            #     ".*_hip_yaw_joint": 0,
            #     ".*_knee_joint": 0,
            # },
            # max_delay = {
            #     ".*_hip_pitch_joint": 1,
            #     ".*_hip_roll_joint": 1,
            #     ".*_hip_yaw_joint": 1,
            #     ".*_knee_joint": 1,
            # },
        ),
        "ankle_pitch": ImplicitActuatorCfg(
            effort_limit_sim=80.0,
            velocity_limit_sim=18.0,
            joint_names_expr=[".*_ankle_pitch_joint"],
            stiffness=2.0 * STIFFNESS_60,
            damping=DAMPING_ANKLE,
            armature=2.0 * ARMATURE_60,
            # friction = 0.2,
            # min_delay = 0,
            # max_delay = 1,
        ),

        "ankle_roll": ImplicitActuatorCfg(
            effort_limit_sim=20.0,
            velocity_limit_sim=18.0,
            joint_names_expr=[".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_60,
            damping=DAMPING_ANKLE,
            armature=2.0 * ARMATURE_60,
            # friction = 0.2,
            # min_delay = 0,
            # max_delay = 1,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=120,
            velocity_limit_sim=35.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_90,
            damping=DAMPING_90,
            armature=ARMATURE_90 * 2,
            # friction = 0.05,
            # min_delay = 0,
            # max_delay = 1,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 50.0,
                ".*_shoulder_roll_joint": 50.0,
                ".*_shoulder_yaw_joint": 50.0,
                ".*_elbow_joint": 50.0,
                ".*_wrist_yaw_joint": 50.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 18.0,
                ".*_shoulder_roll_joint": 18.0,
                ".*_shoulder_yaw_joint": 18.0,
                ".*_elbow_joint": 18.0,
                ".*_wrist_yaw_joint": 18.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_60,
                ".*_shoulder_roll_joint": STIFFNESS_60,
                ".*_shoulder_yaw_joint": STIFFNESS_60,
                ".*_elbow_joint": STIFFNESS_60,
                ".*_wrist_yaw_joint": STIFFNESS_60,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_60,
                ".*_shoulder_roll_joint": DAMPING_60,
                ".*_shoulder_yaw_joint": DAMPING_60,
                ".*_elbow_joint": DAMPING_60,
                ".*_wrist_yaw_joint": DAMPING_60,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_60 * 1.5,
                ".*_shoulder_roll_joint": ARMATURE_60 * 1.5,
                ".*_shoulder_yaw_joint": ARMATURE_60 * 1.5,
                ".*_elbow_joint": ARMATURE_60 * 1.5,
                ".*_wrist_yaw_joint": ARMATURE_60 * 1.5,
            },
            # friction = {
            #     ".*_shoulder_pitch_joint": 0.02,
            #     ".*_shoulder_roll_joint": 0.02,
            #     ".*_shoulder_yaw_joint": 0.02,
            #     ".*_elbow_joint": 0.02,
            #     ".*_wrist_yaw_joint": 0.02,
            # },
            # min_delay = {
            #     ".*_shoulder_pitch_joint": 0,
            #     ".*_shoulder_roll_joint": 0,
            #     ".*_shoulder_yaw_joint": 0,
            #     ".*_elbow_joint": 0,
            #     ".*_wrist_yaw_joint": 0,
            # },
            # max_delay = {
            #     ".*_shoulder_pitch_joint": 1,
            #     ".*_shoulder_roll_joint": 1,
            #     ".*_shoulder_yaw_joint": 1,
            #     ".*_elbow_joint": 1,
            #     ".*_wrist_yaw_joint": 1,
            # },
        ),
    },
)

Z1_ACTION_SCALE = {}
for a in Z1_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            Z1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
