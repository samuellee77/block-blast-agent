import os
import copy
import math
import random
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from blockblast_game.game_state import BlockGameState

# ----------------------------------------------------------------------
#  Device: CPU (your current PyTorch build doesn't support RTX 5090 SM)
# ----------------------------------------------------------------------
DEVICE = th.device("cpu")


# ----------------------------------------------------------------------
#  Encoder + Policy/Value Network
# ----------------------------------------------------------------------
class BlockBlastEncoder(nn.Module):
    """
    Encode BlockBlast state as:
      - grid:   (8, 8)      -> CNN over 1 channel
      - shapes: (3, 5, 5)   -> CNN over 3 channels
      - score:  (1,)
      - combo:  (1,)
    """

    def __init__(self, feat_dim: int = 128):
        super().__init__()

        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.shapes_cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        grid_flat = 16 * 8 * 8     # 1024
        shapes_flat = 16 * 5 * 5   # 400

        self.fc = nn.Sequential(
            nn.Linear(grid_flat + shapes_flat + 2, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim),
            nn.ReLU(),
        )

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        # obs["grid"]   : (B, 8, 8)
        # obs["shapes"] : (B, 3, 5, 5)
        # obs["score"]  : (B, 1)
        # obs["combo"]  : (B, 1)
        grid = obs["grid"].float().unsqueeze(1)   # (B,1,8,8)
        shapes = obs["shapes"].float()            # (B,3,5,5)
        score = obs["score"].float()              # (B,1)
        combo = obs["combo"].float()              # (B,1)

        g = self.grid_cnn(grid)
        s = self.shapes_cnn(shapes)
        x = th.cat([g, s, score, combo], dim=1)
        return self.fc(x)


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style network:
      - Policy head: logits over 192 actions (3 shapes × 8 × 8 positions)
      - Value head: scalar in [-1, 1]
    """

    def __init__(self, feat_dim: int = 128, n_actions: int = 3 * 8 * 8):
        super().__init__()
        self.encoder = BlockBlastEncoder(feat_dim)
        self.policy_head = nn.Linear(feat_dim, n_actions)
        self.value_head = nn.Linear(feat_dim, 1)

    def forward(self, obs: Dict[str, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        h = self.encoder(obs)
        logits = self.policy_head(h)        # (B, 192)
        value = th.tanh(self.value_head(h)) # (B, 1) in [-1, 1]
        return logits, value


# ----------------------------------------------------------------------
#  State encoding helpers
# ----------------------------------------------------------------------
def encode_obs_from_state(state: BlockGameState) -> Dict[str, np.ndarray]:
    """
    Convert BlockGameState.get_normalized_state() into:
      - grid:   (8, 8)
      - shapes: (3, 5, 5)
      - score:  (1,)
      - combo:  (1,)
    """
    s = state.get_normalized_state()

    grid = np.array(s["grid"], dtype=np.float32)  # (8,8)

    shapes_arr = np.zeros((3, 5, 5), dtype=np.float32)
    for i in range(3):
        if i >= len(s["shapes"]):
            break
        shp = s["shapes"][i]
        if shp and len(shp) > 0 and len(shp[0]) > 0:
            h = min(5, len(shp))
            w = min(5, len(shp[0]))
            shapes_arr[i, :h, :w] = np.array([row[:w] for row in shp[:h]], dtype=np.float32)

    score = np.array([float(s["score"])], dtype=np.float32)
    combo = np.array([float(s["combo"])], dtype=np.float32)

    return {
        "grid": grid,
        "shapes": shapes_arr,
        "score": score,
        "combo": combo,
    }


def to_torch_obs(batch_obs: List[Dict[str, np.ndarray]]) -> Dict[str, th.Tensor]:
    out: Dict[str, th.Tensor] = {}
    for key in ["grid", "shapes", "score", "combo"]:
        arr = np.stack([o[key] for o in batch_obs], axis=0)
        out[key] = th.tensor(arr, dtype=th.float32, device=DEVICE)
    return out


def all_valid_actions(state: BlockGameState) -> List[Tuple[int, int, int, int]]:
    """
    Return list of (shape_idx, row, col, flat_action_idx) for all valid moves.
    flat_action_idx = shape_idx * 64 + row * 8 + col
    """
    valids = state.get_valid_actions()  # list of (shape_idx, row, col)
    actions = []
    for (si, r, c) in valids:
        flat = si * 64 + r * 8 + c
        actions.append((si, r, c, flat))
    return actions


# ----------------------------------------------------------------------
#  MCTS Node
# ----------------------------------------------------------------------
class MCTSNode:
    def __init__(
        self,
        state: BlockGameState,
        parent: Optional["MCTSNode"] = None,
        prior: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.children: Dict[int, "MCTSNode"] = {}   # flat_action -> child
        self.P: Dict[int, float] = {}              # prior prob for each action
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.prior = prior
        self.is_expanded = False
        self.is_terminal = state.game_over

    def expand(self, net: AlphaZeroNet, value_scale: float = 100.0):
        """
        Expand the node:
          - Run the network to get policy logits + value
          - Set priors P(a|s) over valid actions
          - Create children nodes (unvisited yet)
          - Return value estimate v in score units (but we keep it in [-1,1])
        """
        if self.is_terminal:
            # Terminal: value is just final score (normalized later)
            v = float(self.state.score) / value_scale
            return v

        obs_np = encode_obs_from_state(self.state)
        t_obs = {
            k: th.tensor(obs_np[k], dtype=th.float32, device=DEVICE).unsqueeze(0)
            for k in obs_np
        }

        with th.no_grad():
            logits, v = net(t_obs)
            logits = logits[0].cpu().numpy()
            v = v.item()  # in [-1,1]

        actions = all_valid_actions(self.state)
        if not actions:
            # No valid moves even though not flagged as game_over; treat as terminal
            self.is_terminal = True
            v = float(self.state.score) / value_scale
            return v

        # Mask invalid actions: we only keep priors for valid ones
        valid_indices = [flat for (_, _, _, flat) in actions]
        valid_logits = logits[valid_indices]
        valid_logits = valid_logits - valid_logits.max()
        exp_logits = np.exp(valid_logits)
        probs = exp_logits / exp_logits.sum()

        # Save priors, create children
        self.P = {}
        for (prob, (si, r, c, flat)) in zip(probs, actions):
            self.P[flat] = float(prob)
            next_state = copy.deepcopy(self.state)
            valid, _ = next_state.place_shape(si, r, c)
            if not valid:
                # Shouldn't really happen, but skip if so
                continue
            self.children[flat] = MCTSNode(next_state, parent=self, prior=float(prob))

        self.is_expanded = True
        return v

    def select_child(self, c_puct: float) -> Tuple[int, "MCTSNode"]:
        """
        Select child with maximum PUCT score:
          Q + c_puct * P * sqrt(sum_N) / (1 + N)
        """
        best_score = -1e9
        best_a = -1
        best_child = None

        total_N = sum(child.N for child in self.children.values()) + 1e-8

        for a, child in self.children.items():
            P_a = self.P.get(a, 0.0)
            Q = child.Q
            U = c_puct * P_a * math.sqrt(total_N) / (1 + child.N)
            score = Q + U
            if score > best_score:
                best_score = score
                best_a = a
                best_child = child

        return best_a, best_child

    def backup(self, v: float):
        """
        Backup value v up the tree.
        Single-player: we use the same v for every node in the path.
        """
        node: Optional[MCTSNode] = self
        while node is not None:
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            node = node.parent


# ----------------------------------------------------------------------
#  MCTS Search
# ----------------------------------------------------------------------
class MCTS:
    def __init__(
        self,
        net: AlphaZeroNet,
        num_simulations: int = 128,
        c_puct: float = 1.5,
        value_scale: float = 100.0,
        root_dirichlet_alpha: float = 0.3,
        root_exploration_fraction: float = 0.25,
    ):
        self.net = net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.value_scale = value_scale
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction

    def _add_dirichlet_noise(self, root: MCTSNode):
        """
        Add Dirichlet noise to the root priors to encourage exploration.
        P'(a) = (1 - ε) * P(a) + ε * Dir(α)
        """
        if not root.P:
            return

        actions = list(root.P.keys())
        n_actions = len(actions)
        noise = np.random.dirichlet([self.root_dirichlet_alpha] * n_actions)
        eps = self.root_exploration_fraction

        for a, n in zip(actions, noise):
            root.P[a] = float((1 - eps) * root.P[a] + eps * n)
            if a in root.children:
                root.children[a].prior = root.P[a]

    def run(self, state: BlockGameState, add_root_noise: bool = True) -> Tuple[MCTSNode, np.ndarray]:
        """
        Run MCTS from the given root state.
        Returns:
          - root node
          - π: visit-count based policy over 192 actions
        """
        root = MCTSNode(copy.deepcopy(state))
        _ = root.expand(self.net, value_scale=self.value_scale)

        # Add Dirichlet noise to root priors (training-time only)
        if add_root_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node = root

            # 1) SELECTION
            while node.is_expanded and not node.is_terminal and node.children:
                _, node = node.select_child(self.c_puct)

            # 2) EXPANSION + EVALUATION
            if node.is_terminal:
                # Use final score normalized to [-1,1] approx
                v = float(node.state.score) / self.value_scale
                v = math.tanh(v)
            else:
                v = node.expand(self.net, value_scale=self.value_scale)

            # 3) BACKUP
            node.backup(v)

        # Build visit-count policy π from root
        pi = np.zeros(3 * 8 * 8, dtype=np.float32)
        total_visits = sum(child.N for child in root.children.values())

        if total_visits == 0:
            # No visits? fall back to uniform over valid actions
            actions = all_valid_actions(root.state)
            if actions:
                p = 1.0 / len(actions)
                for (_, _, _, flat) in actions:
                    pi[flat] = p
            else:
                pi[:] = 1.0 / pi.size
        else:
            for a, child in root.children.items():
                pi[a] = child.N / total_visits

        return root, pi


# ----------------------------------------------------------------------
#  Replay Buffer for (state, π, z)
# ----------------------------------------------------------------------
class AZReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.states = deque(maxlen=capacity)
        self.policies = deque(maxlen=capacity)
        self.values = deque(maxlen=capacity)

    def push(self, obs: Dict[str, np.ndarray], pi: np.ndarray, z: float):
        self.states.append(obs)
        self.policies.append(pi.astype(np.float32))
        self.values.append(float(z))

    def __len__(self):
        return len(self.states)

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.states), size=batch_size, replace=False)
        states = [self.states[i] for i in idxs]
        pis = np.stack([self.policies[i] for i in idxs], axis=0)
        zs = np.array([self.values[i] for i in idxs], dtype=np.float32)
        return states, pis, zs


# ----------------------------------------------------------------------
#  Self-play episode with MCTS (fixed alignment + temperature schedule)
# ----------------------------------------------------------------------
def mcts_self_play_episode(
    mcts: MCTS,
    net: AlphaZeroNet,
    value_scale: float = 100.0,
    base_temperature: float = 1.0,
    late_temperature: float = 0.1,
    temp_switch_move: int = 20,
    max_moves: int = 500,
) -> Tuple[List[Dict[str, np.ndarray]], List[np.ndarray], float]:
    """
    Generate one self-play game using MCTS for move selection.

    IMPORTANT FIX:
      We now store (state_t, π_t) BEFORE applying the sampled action,
      so the replay buffer aligns states with the correct policies.

    Temperature schedule:
      - Moves < temp_switch_move  → T = base_temperature (more exploration)
      - Moves >= temp_switch_move → T = late_temperature (more greedy)

    Returns:
      states:   list of obs dicts
      policies: list of π vectors (len 192)
      final_score: final game score
    """
    state = BlockGameState()
    states: List[Dict[str, np.ndarray]] = []
    policies: List[np.ndarray] = []

    moves = 0
    while moves < max_moves and not state.game_over:
        # Run MCTS from the *current* state
        root, pi = mcts.run(state, add_root_noise=True)

        # --- store state & policy BEFORE the move (fixed alignment) ---
        obs = encode_obs_from_state(state)
        states.append(obs)
        policies.append(pi)

        # Choose temperature based on move index (0-based)
        if moves < temp_switch_move:
            T = base_temperature
        else:
            T = late_temperature

        visit_distribution = pi.copy()
        # Apply temperature: π_T ∝ π^(1/T)
        if T > 0:
            visit_distribution = visit_distribution ** (1.0 / T)
            if visit_distribution.sum() <= 0:
                # Fallback uniform over valid moves
                actions = all_valid_actions(state)
                if not actions:
                    break
                visit_distribution = np.zeros_like(pi)
                p = 1.0 / len(actions)
                for (_, _, _, flat) in actions:
                    visit_distribution[flat] = p
            else:
                visit_distribution /= visit_distribution.sum()
            action_idx = int(np.random.choice(len(visit_distribution), p=visit_distribution))
        else:
            # Pure argmax if T == 0
            action_idx = int(np.argmax(pi))

        # Decode flat action
        si = action_idx // 64
        pos = action_idx % 64
        r = pos // 8
        c = pos % 8

        valid, _ = state.place_shape(si, r, c)
        if not valid:
            # Shouldn't happen if MCTS respects valid actions, but be robust
            break

        moves += 1

    final_score = float(state.score)
    return states, policies, final_score


# ----------------------------------------------------------------------
#  Trainer
# ----------------------------------------------------------------------
class AlphaZeroMCTSTrainer:
    def __init__(
        self,
        lr: float = 1e-4,
        batch_size: int = 256,
        value_scale: float = 100.0,
        entropy_coef: float = 1e-3,
        policy_coef: float = 1.0,
        value_coef: float = 0.5,
        replay_capacity: int = 100_000,
    ):
        self.net = AlphaZeroNet().to(DEVICE)
        self.optim = Adam(self.net.parameters(), lr=lr)
        self.buffer = AZReplayBuffer(capacity=replay_capacity)
        self.batch_size = batch_size
        self.value_scale = value_scale
        self.entropy_coef = entropy_coef
        self.policy_coef = policy_coef
        self.value_coef = value_coef
        self.last_loss = None

    def add_episode(
        self,
        states: List[Dict[str, np.ndarray]],
        policies: List[np.ndarray],
        final_score: float,
    ):
        # Normalize final score to [-1,1] using tanh(score / value_scale)
        z = math.tanh(final_score / self.value_scale)
        for obs, pi in zip(states, policies):
            self.buffer.push(obs, pi, z)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        states, pis, zs = self.buffer.sample(self.batch_size)

        t_obs = to_torch_obs(states)
        t_pi = th.tensor(pis, dtype=th.float32, device=DEVICE)            # (B,192)
        t_z = th.tensor(zs, dtype=th.float32, device=DEVICE).unsqueeze(1) # (B,1)

        logits, values = self.net(t_obs)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        # Policy loss: cross-entropy with π
        policy_loss = -(t_pi * log_probs).sum(dim=1).mean()

        # Value loss: MSE between predicted value and z
        value_loss = F.mse_loss(values, t_z)

        # Entropy bonus
        entropy = -(probs * log_probs).sum(dim=1).mean()

        loss = (
            self.policy_coef * policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
        )

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optim.step()

        self.last_loss = loss.item()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        th.save(self.net.state_dict(), path)

    def load(self, path: str):
        state_dict = th.load(path, map_location=DEVICE)
        self.net.load_state_dict(state_dict)
        self.net.to(DEVICE)


def train_alphazero_mcts(
    total_episodes: int = 3_000,
    num_simulations: int = 128,
    value_scale: float = 100.0,
    base_temperature: float = 1.0,
    late_temperature: float = 0.1,
    temp_switch_move: int = 20,
    train_every_episodes: int = 10,
    train_steps_per_update: int = 20,
    save_dir: Optional[str] = None,
):
    """
    Main training entrypoint.

    Example:
        python -m agents.alphazero_mcts_agent
    """
    if save_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, "models")
    os.makedirs(save_dir, exist_ok=True)

    trainer = AlphaZeroMCTSTrainer(
        lr=1e-4,
        batch_size=256,
        value_scale=value_scale,
        entropy_coef=5e-3,
        policy_coef=1.0,
        value_coef=0.5,
        replay_capacity=100_000,
    )

    mcts = MCTS(
        net=trainer.net,
        num_simulations=num_simulations,
        c_puct=1.5,
        value_scale=value_scale,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
    )

    episode_scores: List[float] = []

    for ep in range(1, total_episodes + 1):
        states, policies, final_score = mcts_self_play_episode(
            mcts,
            trainer.net,
            value_scale=value_scale,
            base_temperature=base_temperature,
            late_temperature=late_temperature,
            temp_switch_move=temp_switch_move,
            max_moves=500,
        )

        trainer.add_episode(states, policies, final_score)
        episode_scores.append(final_score)

        if ep % train_every_episodes == 0 and len(trainer.buffer) >= trainer.batch_size:
            for _ in range(train_steps_per_update):
                trainer.train_step()

        if ep % 10 == 0:
            last_k = min(len(episode_scores), 10)
            avg_last = sum(episode_scores[-last_k:]) / last_k
            print(
                f"[AZ-MCTS] ep={ep}/{total_episodes}  "
                f"avg_score(last {last_k})={avg_last:.2f}  "
                f"buffer={len(trainer.buffer)}  "
                f"last_loss={trainer.last_loss if trainer.last_loss is not None else 'N/A'}"
            )

    model_path = os.path.join(save_dir, "alphazero_mcts_model.pt")
    trainer.save(model_path)
    print(f"[AZ-MCTS] Training done. Saved model to {model_path}")


# ----------------------------------------------------------------------
#  Policy wrapper for evaluation (NN-only, no search at inference)
# ----------------------------------------------------------------------
class AlphaZeroMCTSPolicyWrapper:
    """
    Wraps a trained AlphaZeroNet so it looks like a Stable-Baselines policy:

        .predict(obs, action_masks=None, deterministic=True)

    NOTE: this uses only the network policy (no tree search) at inference time.
    This is useful if you want a fast, SB3-style interface, but it will be
    weaker than "MCTS + NN".
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = th.device(device)
        self.net = AlphaZeroNet().to(self.device)
        state_dict = th.load(model_path, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    @th.no_grad()
    def predict(self, obs: Dict[str, np.ndarray], action_masks=None, deterministic: bool = True):
        t_obs = {
            k: th.tensor(obs[k], dtype=th.float32, device=self.device).unsqueeze(0)
            for k in obs
        }
        logits, _ = self.net(t_obs)
        logits = logits[0].cpu().numpy()

        if action_masks is not None:
            masked_logits = logits.copy()
            masked_logits[~action_masks] = -1e9
            logits = masked_logits

        if deterministic:
            action = int(np.argmax(logits))
        else:
            logits = logits - logits.max()
            probs = np.exp(logits)
            s = probs.sum()
            if s <= 0:
                if action_masks is not None and action_masks.any():
                    valid = np.nonzero(action_masks)[0]
                    action = int(np.random.choice(valid))
                else:
                    action = 0
            else:
                probs /= s
                action = int(np.random.choice(len(probs), p=probs))

        return action, None


# ----------------------------------------------------------------------
#  Simple MCTS eval wrapper (search at inference)
# ----------------------------------------------------------------------
class AlphaZeroMCTSSearchWrapper:
    """
    A simple wrapper that runs MCTS at inference time so you can see the full
    AlphaZero-style boost over random.

    Usage pattern (custom eval loop, *not* SB3):
        net_wrapper = AlphaZeroMCTSSearchWrapper(".../alphazero_mcts_model.pt")
        state = BlockGameState()
        while not state.game_over:
            action_flat = net_wrapper.select_action(state)
            si = action_flat // 64
            pos = action_flat % 64
            r = pos // 8
            c = pos % 8
            state.place_shape(si, r, c)
    """

    def __init__(
        self,
        model_path: str,
        num_simulations: int = 128,
        c_puct: float = 1.5,
        value_scale: float = 100.0,
        device: str = "cpu",
    ):
        self.device = th.device(device)
        self.net = AlphaZeroNet().to(self.device)
        state_dict = th.load(model_path, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()

        # MCTS used *only* for inference here: no Dirichlet noise.
        self.mcts = MCTS(
            net=self.net,
            num_simulations=num_simulations,
            c_puct=c_puct,
            value_scale=value_scale,
            root_dirichlet_alpha=0.0,
            root_exploration_fraction=0.0,
        )

    def select_action(self, state: BlockGameState) -> int:
        """
        Run MCTS from the given state and return the flat action index (0..191)
        corresponding to argmax over visit counts.
        """
        root, pi = self.mcts.run(state, add_root_noise=False)
        return int(np.argmax(pi))


if __name__ == "__main__":
    # You can tune total_episodes / num_simulations depending on runtime.
    train_alphazero_mcts(
        total_episodes=1000,
        num_simulations=128,
        value_scale=100.0,
        base_temperature=1.0,
        late_temperature=0.1,
        temp_switch_move=20,
        train_every_episodes=10,
        train_steps_per_update=20,
    )
