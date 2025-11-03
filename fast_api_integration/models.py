# fast_api_integration/models.py

import os
from sb3_contrib import MaskablePPO
from agents.dqn_masked_agent import MaskableDQN
from blockblast_game.game_env import BlockGameEnv

# Directory where your .zip models are stored
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "agents", "models")

# This dict will hold loaded models
models = {}


def load_models() -> None:
    """Load all SB3 models into the `models` dict."""
    # Maskable PPO
    mp_path = os.path.join(MODELS_DIR, "final_masked_ppo_model.zip")
    if os.path.isfile(mp_path):
        try:
            models["MaskedPPO"] = MaskablePPO.load(mp_path)
        except Exception as e:
            print(f"Failed to load MaskablePPO: {e}")
    # Maskable DQN
    mdqn_path = os.path.join(MODELS_DIR, "final_masked_dqn_model.zip")
    if os.path.isfile(mdqn_path):
        try:
            dummy_env = BlockGameEnv(render_mode=None)
            models["MaskedDQN"] = MaskableDQN.load(mdqn_path, env=dummy_env)
        except Exception as e:
            print(f"Failed to load MaskableDQN: {e}")
