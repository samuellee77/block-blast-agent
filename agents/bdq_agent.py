import os, csv, random
from collections import deque

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from stable_baselines3.common.monitor import Monitor
from blockblast_game.game_env import BlockGameEnv

# ----------------------------------------------------------------------
#  Device: keep CPU for now (your GPU wheel doesn't support RTX 5090 SM)
# ----------------------------------------------------------------------
DEVICE = th.device("cpu")


# ----------------------------------------------------------------------
#  Feature extractor: small, fast CNN + MLP
# ----------------------------------------------------------------------
class BlockBlastEncoder(nn.Module):
    """
    Encode BlockBlast observations:
      - grid:   (8, 8)      -> CNN over 1 channel
      - shapes: (3, 5, 5)   -> CNN over 3 channels
      - score:  (1,)
      - combo:  (1,)
    """

    def __init__(self, feat_dim: int = 128):
        super().__init__()

        # Much smaller convs than before for speed
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

        # 16 * 8 * 8 = 1024, 16 * 5 * 5 = 400
        grid_flat = 16 * 8 * 8
        shapes_flat = 16 * 5 * 5

        self.fc = nn.Sequential(
            nn.Linear(grid_flat + shapes_flat + 2, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict) -> th.Tensor:
        grid = obs["grid"].float().unsqueeze(1)   # (B, 1, 8, 8)
        shapes = obs["shapes"].float()            # (B, 3, 5, 5)
        score = obs["score"].float()              # (B, 1)
        combo = obs["combo"].float()              # (B, 1)

        g = self.grid_cnn(grid)
        s = self.shapes_cnn(shapes)
        x = th.cat([g, s, score, combo], dim=1)
        return self.fc(x)


class DuelingHead(nn.Module):
    """
    Dueling head for one branch (shape / row / col).
    Outputs Q-values for that branch.
    """

    def __init__(self, feat_dim: int, n_actions: int):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, h: th.Tensor) -> th.Tensor:
        v = self.value(h)                     # (B, 1)
        a = self.advantage(h)                 # (B, n_actions)
        return v + (a - a.mean(dim=1, keepdim=True))


class BDQ(nn.Module):
    """
    Branching Dueling Q-network:
      - q_shape: 3 actions
      - q_row:   8 actions
      - q_col:   8 actions
    """

    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.encoder = BlockBlastEncoder(feat_dim)
        self.q_shape = DuelingHead(feat_dim, 3)
        self.q_row = DuelingHead(feat_dim, 8)
        self.q_col = DuelingHead(feat_dim, 8)

    def forward(self, obs: dict):
        h = self.encoder(obs)
        return self.q_shape(h), self.q_row(h), self.q_col(h)


# ----------------------------------------------------------------------
#  Replay buffer (simple, fast)
# ----------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_triplet, reward, next_state, done):
        # action_triplet = (shape_idx, row, col)
        self.buffer.append((state, action_triplet, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = map(list, zip(*batch))
        return s, a, np.array(r, dtype=np.float32), s2, np.array(d, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)


def to_torch_obs(batch_obs):
    """
    Convert list[dict] of numpy obs to dict of torch tensors on DEVICE.
    Keys: "grid", "shapes", "score", "combo"
    """
    out = {}
    for key in ["grid", "shapes", "score", "combo"]:
        arr = np.stack([o[key] for o in batch_obs], axis=0)
        out[key] = th.tensor(arr, dtype=th.float32, device=DEVICE)
    return out


# ----------------------------------------------------------------------
#  BDQ Agent with simple Double Q + masking at action selection
# ----------------------------------------------------------------------
class BDQAgent:
    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.995,
        batch_size: int = 256,
        tau: float = 0.005,
        eps_start: float = 0.2,
        eps_end: float = 0.05,
        eps_decay_steps: int = 200_000,
    ):
        self.net = BDQ().to(DEVICE)
        self.target = BDQ().to(DEVICE)
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()

        self.optim = Adam(self.net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.total_steps = 0

    def epsilon(self) -> float:
        """Linear decay of epsilon."""
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @th.no_grad()
    def select_action(self, env: BlockGameEnv, obs: dict):
        """
        Epsilon-greedy over valid actions.
        env: underlying BlockGameEnv (NOT Monitor)
        obs: dict observation from env._get_observation()
        """
        self.total_steps += 1
        eps = self.epsilon()

        # 1) valid flat actions, then decode to (shape,row,col)
        flat_valids = env.get_valid_actions()  # list[int]
        if not flat_valids:
            # Fallback: should be rare because generator avoids dead states
            a_flat = random.randrange(env.action_space.n)
            si, r, c = env._decode_action(a_flat)
            return si, r, c

        valid_triplets = [env._decode_action(a) for a in flat_valids]

        # 2) ε-greedy random among valid actions
        if random.random() < eps:
            return random.choice(valid_triplets)

        # 3) greedy action: compute Q_total only once, then loop over valids
        o = {
            k: th.tensor(obs[k], dtype=th.float32, device=DEVICE).unsqueeze(0)
            for k in obs
        }
        q_s, q_r, q_c = self.net(o)  # (1,3), (1,8), (1,8)
        q_s = q_s[0].cpu().numpy()
        q_r = q_r[0].cpu().numpy()
        q_c = q_c[0].cpu().numpy()

        best = None
        best_q = -1e9
        for si, r, c in valid_triplets:
            q = q_s[si] + q_r[r] + q_c[c]
            if q > best_q:
                best_q = q
                best = (si, r, c)

        return best

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        s = to_torch_obs(states)
        s2 = to_torch_obs(next_states)
        r = th.tensor(rewards, dtype=th.float32, device=DEVICE).unsqueeze(1)  # (B,1)
        d = th.tensor(dones, dtype=th.float32, device=DEVICE).unsqueeze(1)    # (B,1)

        # 1) Q(s, a)
        q_s, q_r, q_c = self.net(s)  # (B,3), (B,8), (B,8)
        a = th.tensor(actions, dtype=th.long, device=DEVICE)  # (B,3)
        si = a[:, 0].unsqueeze(1)  # (B,1)
        ri = a[:, 1].unsqueeze(1)
        ci = a[:, 2].unsqueeze(1)

        q_sa = (
            q_s.gather(1, si)
            + q_r.gather(1, ri)
            + q_c.gather(1, ci)
        )  # (B,1)

        # 2) Double Q target: use online net for argmax, target net for value
        with th.no_grad():
            q_s2, q_r2, q_c2 = self.net(s2)            # online
            a2_shape = q_s2.argmax(dim=1, keepdim=True)
            a2_row = q_r2.argmax(dim=1, keepdim=True)
            a2_col = q_c2.argmax(dim=1, keepdim=True)

            tq_s2, tq_r2, tq_c2 = self.target(s2)      # target
            q_next = (
                tq_s2.gather(1, a2_shape)
                + tq_r2.gather(1, a2_row)
                + tq_c2.gather(1, a2_col)
            )  # (B,1)

            target = r + (1.0 - d) * self.gamma * q_next

        loss = F.smooth_l1_loss(q_sa, target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optim.step()

        self.last_loss = loss.item()

        # 3) Polyak update of target network
        with th.no_grad():
            for p, tp in zip(self.net.parameters(), self.target.parameters()):
                tp.data.lerp_(p.data, self.tau)


# ----------------------------------------------------------------------
#  Training loop
# ----------------------------------------------------------------------
def train_bdq(
    total_steps: int = 1_000_000,
    log_every: int = 5_000,
    save_dir: str = None,
):
    if save_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, "models")
    os.makedirs(save_dir, exist_ok=True)

    env = Monitor(BlockGameEnv(render_mode=None))
    agent = BDQAgent()
    obs, _ = env.reset()

    ep_return, ep_len = 0.0, 0
    episode_scores = []
    losses = []
    for step in range(1, total_steps + 1):
        # Use underlying BlockGameEnv for valid action / decode
        si, r, c = agent.select_action(env.env, obs)
        a_flat = si * 64 + r * 8 + c

        next_obs, reward, done, truncated, info = env.step(a_flat)
        done_flag = float(done or truncated)

        # Store transition
        agent.buffer.push(obs, (si, r, c), reward, next_obs, done_flag)

        # Book-keeping
        obs = next_obs
        ep_return += reward
        ep_len += 1

        # Train
        agent.train_step()

        if done or truncated:
            print(f"[Episode ended] step={step} return={ep_return:.2f} length={ep_len}")
            # Save to logs
            episode_scores.append(ep_return)
            obs, _ = env.reset()
            ep_return, ep_len = 0.0, 0

        if step % log_every == 0:
            loss_val = getattr(agent, "last_loss", None)
            print(
                f"[BDQ] step={step:,}  "
                f"buffer={len(agent.buffer)}  "
                f"eps={agent.epsilon():.3f}  "
                f"loss={loss_val:.5f}  "
                f"latest_score={episode_scores[-1] if episode_scores else 0:.2f}"
            )

            # Save history
            if loss_val is not None:
                losses.append((step, loss_val))


    # Save only the online network (you can also save target if you want)
    model_path = os.path.join(save_dir, "bdq_model.pt")
    th.save(agent.net.state_dict(), model_path)
    print(f"[BDQ] Training done. Saved model to {model_path}")

    # Save losses
    with open(os.path.join(save_dir, "bdq_loss_log.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        writer.writerows(losses)

    # Save episode scores
    with open(os.path.join(save_dir, "bdq_score_log.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score"])
        for i, score in enumerate(episode_scores):
            writer.writerow([i, score])

    print(f"[BDQ] Saved loss log and score log to {save_dir}")

# ----------------------------------------------------------------------
#  BDQ evaluation wrapper (for simulate_playing.py)
# ----------------------------------------------------------------------
class BDQPolicyWrapper:
    """
    Lightweight wrapper around a trained BDQ network so it looks like a
    Stable-Baselines policy with a `.predict()` method returning a flat action.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = th.device(device)
        self.net = BDQ().to(self.device)
        state_dict = th.load(model_path, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    @th.no_grad()
    def predict(self, obs, action_masks=None, deterministic: bool = True):
        """
        obs: a single observation dict from BlockGameEnv._get_observation()
        action_masks: optional np.bool_ array of shape (192,)
        Returns: (action_int, None)
        """
        # Convert obs dict → torch batch of size 1
        o = {
            k: th.tensor(obs[k], dtype=th.float32, device=self.device).unsqueeze(0)
            for k in obs
        }

        q_s, q_r, q_c = self.net(o)  # (1,3), (1,8), (1,8)
        q_s = q_s[0].cpu().numpy()
        q_r = q_r[0].cpu().numpy()
        q_c = q_c[0].cpu().numpy()

        # Enumerate all (shape,row,col) combos and optionally respect action_masks
        best_action = None
        best_q = -1e9

        for shape_idx in range(3):
            for row in range(8):
                for col in range(8):
                    flat = shape_idx * 64 + row * 8 + col
                    if action_masks is not None and not action_masks[flat]:
                        continue
                    q = q_s[shape_idx] + q_r[row] + q_c[col]
                    if q > best_q:
                        best_q = q
                        best_action = flat

        # Fallback: if somehow no valid action was found
        if best_action is None:
            if action_masks is not None and action_masks.any():
                valid = np.nonzero(action_masks)[0]
                best_action = int(np.random.choice(valid))
            else:
                best_action = 0

        return int(best_action), None


if __name__ == "__main__":
    # You can lower this for a quick sanity test, e.g., 100_000
    train_bdq(total_steps=500_000)
