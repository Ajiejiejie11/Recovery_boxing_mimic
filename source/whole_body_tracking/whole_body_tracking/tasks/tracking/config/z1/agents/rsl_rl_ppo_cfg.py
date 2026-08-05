from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


PROJECT_ROOT = Path(__file__).resolve().parents[8]


@configclass
class RecoveryAmpCfg:
    """Independent recovery-style reward using AMP_mjlab-style interpolation."""

    enabled: bool = True
    expert_motion_path: str = str(PROJECT_ROOT / "motion_data/data_npz/npz")
    amp_reward_coef: float = 0.1
    amp_task_reward_lerp: float = 0.85
    hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    learning_rate: float = 1.0e-4
    batch_size: int = 2048
    updates_per_iteration: int = 4
    replay_capacity: int = 50_000
    min_replay_size: int = 4096
    gradient_penalty: float = 10.0
    max_grad_norm: float = 10.0


@configclass
class Z1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "z1_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Z1FlatRecoveryAmpPPORunnerCfg(Z1FlatPPORunnerCfg):
    experiment_name = "z1_flat_recovery_amp"
    recovery_amp: RecoveryAmpCfg = RecoveryAmpCfg()


LOW_FREQ_SCALE = 0.5


@configclass
class Z1FlatLowFreqPPORunnerCfg(Z1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)
