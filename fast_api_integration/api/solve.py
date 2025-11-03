from typing import List, Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import numpy as np
from blockblast_game.game_env import BlockGameEnv

# Import the already-loaded `models` dict
from fast_api_integration.models import models

router = APIRouter()


# --- Pydantic schemas --------------------------------
class SolveRequest(BaseModel):
    grid: List[List[int]]
    shapes: List[List[List[int]]]  # three 5×5 masks
    score: List[float]  # e.g. [0.0]
    combo: List[int]  # e.g. [0]
    model: Literal["PPO", "DQN", "MaskedPPO", "MaskedDQN"] = "MaskedDQN"


class StepPrediction(BaseModel):
    step: int
    action: int
    shape_idx: int
    row: int
    col: int
    board: List[List[int]]
    reward: float
    score: float
    lines_cleared: int
    done: bool


class SolveResponse(BaseModel):
    steps: List[StepPrediction]


@router.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    import numpy as np  # ← make sure numpy is in scope

    # 1. Retrieve model ------------------------------------------------------
    if req.model not in models:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not loaded.")
    model = models[req.model]

    # 2. Reconstruct environment --------------------------------------------
    env = BlockGameEnv(render_mode=None)
    env.game_state.grid = [row[:] for row in req.grid]
    env.game_state.score = float(req.score[0])
    env.game_state.combos = [["COMBO 0"], int(req.combo[0])]

    # build Shape objects from the raw 5×5 masks
    custom_shapes = []
    for mask in req.shapes:
        shp = type("ShapeObj", (), {})()
        shp.form = mask
        shp.color = 1  # dummy colour
        custom_shapes.append(shp)
    env.game_state.current_shapes = custom_shapes

    obs = env._get_observation()
    steps: list[StepPrediction] = []

    # 3. Play each of the three shapes in order ------------------------------
    for shape_idx in range(3):
        # shape might already have been consumed by the environment
        if env.game_state.current_shapes[shape_idx] == 0:
            continue

        # all actions whose decoded shape-index == current loop index
        shape_actions = [
            a
            for a in range(env.action_space.n)
            if env._decode_action(a)[0] == shape_idx
        ]
        if not shape_actions:  # no legal placement → skip
            continue

        # ----- choose an action --------------------------------------------
        if hasattr(env, "action_masks"):
            base_mask = env.action_masks()
            restricted_mask = np.zeros_like(base_mask, dtype=bool)
            restricted_mask[shape_actions] = True
            action_mask = base_mask & restricted_mask
            if not action_mask.any():  # shouldn’t happen, but be safe
                action = shape_actions[0]
            else:
                action, _ = model.predict(
                    obs, action_masks=action_mask, deterministic=True
                )
        else:
            action, _ = model.predict(obs, deterministic=True)
            if action not in shape_actions:  # fall back to first legal move
                action = shape_actions[0]

        # ----- apply --------------------------------------------------------
        obs, reward, terminated, truncated, info = env.step(int(action))
        placed_shape_idx, row, col = env._decode_action(int(action))

        steps.append(
            StepPrediction(
                step=len(steps) + 1,
                action=int(action),
                shape_idx=placed_shape_idx,
                row=row,
                col=col,
                board=[r[:] for r in env.game_state.grid],
                reward=reward,
                score=float(env.game_state.score),
                lines_cleared=int(info.get("lines_cleared", 0)),
                done=bool(terminated or truncated),
            )
        )

        if terminated or truncated:
            break

    return SolveResponse(steps=steps)
