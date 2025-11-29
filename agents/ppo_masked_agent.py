import os
import sys
from typing import Callable, Optional

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from blockblast_game.game_env import BlockGameEnv
from agents.agent_visualizer import visualize_agent  # pragma: no cover

# Ensure project root is on the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Directories
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def make_env(rank: int, seed: int = 0) -> Callable[[], BlockGameEnv]:
    """Factory for SubprocVecEnv."""

    def _init():
        env = BlockGameEnv()
        env = Monitor(env)
        try:
            env.reset(seed=seed + rank)
        except TypeError:
            env.seed(seed + rank)
        return env

    return _init


def _standard_masked(env, **kwargs) -> MaskablePPO:
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=LOGS_DIR,
        learning_rate=5e-5,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.05,
        max_grad_norm=0.5,
        **kwargs,
    )


def train_masked_ppo(
    *,
    num_envs: int = 4,
    total_timesteps: int = 1_000_000,
    save_path: Optional[str] = None,
    continue_training: bool = False,
    pretrained_path: Optional[str] = None,
):
    save_dir = save_path or MODELS_DIR
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    if continue_training and pretrained_path and os.path.isfile(pretrained_path):
        print(f"[masked ppo] Continuing from {pretrained_path}")
        model = MaskablePPO.load(pretrained_path, env=env)
        model.tensorboard_log = LOGS_DIR
        reset_flag = False
    else:
        model = _standard_masked(env)
        reset_flag = True

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=save_dir,
        name_prefix="masked_ppo_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_env = SubprocVecEnv([make_env(0, seed=42)])
    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=100_000,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        reset_num_timesteps=reset_flag,
    )

    final_path = os.path.join(save_dir, "final_masked_ppo_model")
    model.save(final_path)
    print(f"[masked ppo] Training done: {final_path}.zip")
    return model


if __name__ == "__main__":
    # Configuration
    num_envs = 8
    total_timesteps = 1000000
    continue_training = False

    do_train = True
    do_visualize = True

    if do_train:
        pretrained = os.path.join(MODELS_DIR, "final_masked_ppo_model.zip")
        train_masked_ppo(
            num_envs=num_envs,
            total_timesteps=total_timesteps,
            continue_training=continue_training,
            pretrained_path=pretrained,
        )

    if do_visualize:
        render_env = BlockGameEnv(render_mode="human")
        render_env = Monitor(render_env)
        model_file = os.path.join(MODELS_DIR, "final_masked_ppo_model.zip")
        print(f"[masked ppo] Loading model from {model_file}")
        agent = MaskablePPO.load(model_file, env=render_env)
        visualize_agent(render_env, agent, episodes=10, delay=0.2, use_masks=True)
