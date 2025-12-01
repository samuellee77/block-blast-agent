import os
import pandas as pd
import random
import statistics

from blockblast_game.game_state import BlockGameState
from agents.alphazero_mcts_agent import AlphaZeroMCTSSearchWrapper


def run_random_episode(max_moves: int = 500) -> float:
    """
    Pure random baseline on the raw BlockGameState.
    """
    state = BlockGameState()
    moves = 0

    while not state.game_over and moves < max_moves:
        # list of (shape_idx, row, col)
        valid_actions = state.get_valid_actions()
        if not valid_actions:
            break

        # Just use Python's random.choice here
        si, r, c = random.choice(valid_actions)

        valid, _ = state.place_shape(si, r, c)
        if not valid:
            # If Board logic is correct, this should be rare
            break

        moves += 1

    return float(state.score), moves


def run_mcts_episode(
    policy: AlphaZeroMCTSSearchWrapper,
    max_moves: int = 500,
) -> float:
    """
    Use AlphaZeroMCTSSearchWrapper to run full MCTS+NN search at each move.
    """
    state = BlockGameState()
    moves = 0

    while not state.game_over and moves < max_moves:
        # Ask MCTS which flat action (0..191) to play
        flat = policy.select_action(state)

        si = flat // 64
        pos = flat % 64
        r = pos // 8
        c = pos % 8

        valid, _ = state.place_shape(si, r, c)
        if not valid:
            # If this happens frequently, something is off in your action mapping
            break

        moves += 1

    return float(state.score), moves


def evaluate(
    model_path: str,
    num_episodes: int = 200,
    max_moves: int = 500,
    num_simulations: int = 128,
    c_puct: float = 1.5,
    value_scale: float = 100.0,
):
    # 1) Random baseline
    random_scores = []
    random_moves = []
    for ep in range(num_episodes):
        s, m = run_random_episode(max_moves=max_moves)
        random_scores.append(s)
        random_moves.append(m)
        if (ep + 1) % 20 == 0:
            print(f"[Random] ep={ep+1}/{num_episodes}  last_score={s:.2f}")

    # 2) AlphaZero-MCTS search agent
    mcts_agent = AlphaZeroMCTSSearchWrapper(
        model_path=model_path,
        num_simulations=num_simulations,
        c_puct=c_puct,
        value_scale=value_scale,
        device="cpu",
    )

    az_scores = []
    az_moves = []
    for ep in range(num_episodes):
        s, m = run_mcts_episode(mcts_agent, max_moves=max_moves)
        az_scores.append(s)
        az_moves.append(m)
        if (ep + 1) % 20 == 0:
            print(f"[AZ-MCTS Search] ep={ep+1}/{num_episodes}  last_score={s:.2f}")

    # 3) Print summary stats
    def summary(name, scores):
        if not scores:
            print(f"{name}: no scores recorded")
            return
        mean = statistics.mean(scores)
        stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        print(
            f"{name}: mean={mean:.2f}  std={stdev:.2f}  "
            f"min={min(scores):.2f}  max={max(scores):.2f}"
        )
    def save(method, scores, moves):
        df = pd.DataFrame({
            'method': method,
            "scores": scores,
            "moves": moves
        })
        return df
    random_df = save("random", random_scores, random_moves)
    random_df.to_csv("./random_results.csv", index=False)
    az_df = save("MCTS", az_scores, az_moves)
    az_df.to_csv("./mcts_results.csv", index=False)
    print("\n=== Evaluation Summary (BlockGameState + MCTS) ===")
    summary("Random Scores:", random_scores)
    summary("Random Moves:", random_moves)
    summary("AlphaZero-MCTS Search Scores:", az_scores)
    summary("AlphaZero-MCTS Search Moves:", az_moves)


if __name__ == "__main__":
    # Adjust this path if your models directory is elsewhere
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, os.pardir, "agents", "models")
    model_path = os.path.join(models_dir, "alphazero_mcts_model.pt")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"AlphaZero-MCTS model not found at {model_path}. "
            f"Train it first with `python -m agents.alphazero_mcts_agent`."
        )

    evaluate(
        model_path=model_path,
        num_episodes=500,
        max_moves=500,
        num_simulations=64,
        c_puct=1.5,
        value_scale=100.0,
    )
