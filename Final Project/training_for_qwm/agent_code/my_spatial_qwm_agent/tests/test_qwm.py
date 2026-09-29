"""
Unit and integration tests for my_spatial_qwm_agent:
  - DynActivation behavior and gradients
  - SpatialDuelingDQN fused representation extraction
  - LatentWorldModel transition and reward predictions
  - QWM Tree Search planning over actions
  - Partial weight loading from existing spatial DQN checkpoints
  - Phased DQN freezing during world model warmup
"""

from pathlib import Path
import os
import sys
import pytest
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent_code.my_spatial_qwm_agent.model import (
    ACTIONS,
    N_ACTIONS,
    DynActivation,
    dynActivation,
    dynActvation,
    SpatialDuelingDQN,
    LatentWorldModel,
    SpatialQWMAgent,
    load_matching_weights,
    new_model,
)


def test_dyn_activation():
    act = DynActivation()
    # Check alias
    assert dynActivation is DynActivation
    assert dynActvation is DynActivation

    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    out = act(x)
    # Initial alpha=1.0, beta=0.0 -> exact Mish
    expected = F.mish(x)
    assert torch.allclose(out, expected, atol=1e-6)

    # Check gradient flow to alpha and beta
    loss = out.sum()
    loss.backward()
    assert act.alpha.grad is not None
    assert act.beta.grad is not None


def test_spatial_dueling_dqn_fused():
    net = SpatialDuelingDQN()
    spatial = torch.randn(2, 10, 17, 17)
    features = torch.randn(2, 33)

    fused = net.encode_fused(spatial, features)
    assert fused.shape == (2, 256)

    q_from_fused = net.forward_from_fused(fused)
    assert q_from_fused.shape == (2, N_ACTIONS)

    q_full = net(spatial, features)
    assert torch.allclose(q_from_fused, q_full, atol=1e-5)


def test_latent_world_model():
    wm = LatentWorldModel()
    z = torch.randn(4, 256)
    actions = torch.tensor([0, 2, 4, 5])

    next_z, pred_r, pred_v, pred_legal = wm(z, actions)
    assert next_z.shape == (4, 256)
    assert pred_r.shape == (4, 1)
    assert pred_v.shape == (4, 1)
    assert pred_legal.shape == (4, N_ACTIONS)

    # Single step inference helper
    z_single = torch.randn(256)
    nz_s, r_s, v_s, leg_s = wm.predict_next(z_single, 1)
    assert nz_s.shape == (1, 256)
    assert isinstance(r_s, float)
    assert isinstance(v_s, float)
    assert leg_s.shape == (1, N_ACTIONS)

    # Direct predict_legal helper
    leg_direct = wm.predict_legal(z_single)
    assert leg_direct.shape == (1, N_ACTIONS)


def test_partial_checkpoint_loading():
    agent = new_model()
    # Check if existing spatial DQN Stage 4 checkpoint exists
    ckpt_path = ROOT_DIR / "agent_code/my_spatial_dqn_agent/runs/run_2026-09-06_18-24-53_spatial_dqn/best_model_s4_vs_rule_based.pt"
    if not ckpt_path.is_file():
        ckpt_path = ROOT_DIR / "agent_code/my_spatial_dqn_agent/my-saved-model-spatial-dqn.pt"

    if ckpt_path.is_file():
        # Store original world model weight
        orig_wm_weight = agent.world_model.trunk[0].weight.clone()
        agent.load(str(ckpt_path), partial=True)

        # Policy net conv layer was loaded from checkpoint
        assert agent.policy_net.conv[0].weight is not None
        # World model layer remained untouched
        assert torch.equal(agent.world_model.trunk[0].weight, orig_wm_weight)


def test_freezing_unfreezing():
    agent = new_model()
    assert not agent.is_dqn_frozen

    agent.freeze_dqn()
    assert agent.is_dqn_frozen
    for p in agent.policy_net.parameters():
        assert not p.requires_grad
    for p in agent.world_model.parameters():
        assert p.requires_grad

    agent.unfreeze_dqn()
    assert not agent.is_dqn_frozen
    for p in agent.policy_net.parameters():
        assert p.requires_grad


def test_qwm_tree_search_action_scores():
    agent = new_model()
    spatial = np.random.randn(10, 17, 17).astype(np.float32)
    features = np.random.randn(33).astype(np.float32)
    mask = np.array([True, True, False, True, True, False], dtype=bool)

    # Depth 1 search
    q1 = agent.tree_search_action_scores(spatial, features, depth=1, discount=0.1, valid_mask=mask)
    assert q1.shape == (N_ACTIONS,)
    assert np.all(np.isfinite(q1[mask]))
    assert q1[2] < -100.0  # masked action penalized
    assert q1[5] < -100.0

    # Depth 2 search
    q2 = agent.tree_search_action_scores(spatial, features, depth=2, discount=0.1, valid_mask=mask)
    assert q2.shape == (N_ACTIONS,)
    assert np.all(np.isfinite(q2[mask]))

    # Depth 4 search with return_details
    q4, details4 = agent.tree_search_action_scores(
        spatial, features, depth=4, beam_size=12, discount=0.1, valid_mask=mask, return_details=True
    )
    assert q4.shape == (N_ACTIONS,)
    assert len(details4) == N_ACTIONS
    for d in details4:
        assert len(d['actions']) == 4  # 4-step lookahead trajectory
        assert d['a0_name'] == ACTIONS[d['a0']]

    # Depth 8 search with return_details
    q8, details8 = agent.tree_search_action_scores(
        spatial, features, depth=8, beam_size=12, discount=0.1, valid_mask=mask, return_details=True
    )
    assert q8.shape == (N_ACTIONS,)
    assert len(details8) == N_ACTIONS
    for d in details8:
        assert len(d['actions']) == 8  # 8-step lookahead trajectory
        assert d['a0_name'] == ACTIONS[d['a0']]
    assert q8[2] < -100.0  # masked action penalized
    assert q8[5] < -100.0


def test_phased_batch_update():
    agent = new_model()
    batch_size = 8
    spatial = np.random.randn(batch_size, 10, 17, 17).astype(np.float32)
    features = np.random.randn(batch_size, 33).astype(np.float32)
    actions = np.random.randint(0, N_ACTIONS, size=batch_size)
    rewards = np.random.randn(batch_size).astype(np.float32)
    next_spatial = np.random.randn(batch_size, 10, 17, 17).astype(np.float32)
    next_features = np.random.randn(batch_size, 33).astype(np.float32)
    dones = np.zeros(batch_size, dtype=np.float32)

    # 1. Update while frozen
    agent.freeze_dqn()
    metrics_frozen = agent.update_batch(spatial, features, actions, rewards, next_spatial, next_features, dones)
    assert metrics_frozen["loss"] == 0.0
    assert metrics_frozen["wm_loss"] >= 0.0
    assert metrics_frozen["trans_loss"] >= 0.0
    assert "legal_loss" in metrics_frozen
    assert metrics_frozen["legal_loss"] >= 0.0

    # 2. Update while unfrozen
    agent.unfreeze_dqn()
    metrics_unfrozen = agent.update_batch(spatial, features, actions, rewards, next_spatial, next_features, dones)
    assert metrics_unfrozen["loss"] >= 0.0
    assert metrics_unfrozen["wm_loss"] >= 0.0
    assert "legal_loss" in metrics_unfrozen
    assert metrics_unfrozen["legal_loss"] >= 0.0


def test_plan_overlay_trace():
    from agent_code.my_spatial_qwm_agent import callbacks
    from agent_code.my_spatial_qwm_agent.plan_overlay import (
        get_latest_overlay_trace, clear_overlay_trace, render_overlay
    )

    class DummyAgent:
        train = False

    self_obj = DummyAgent()
    callbacks.setup(self_obj)
    self_obj.tree_search = True
    self_obj.search_depth = 2

    # Create dummy game state with standard border walls
    field = np.zeros((17, 17), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1

    game_state = {
        'round': 1,
        'step': 1,
        'field': field,
        'bombs': [],
        'explosion_map': np.zeros((17, 17), dtype=int),
        'coins': [(3, 3)],
        'self': ('my_spatial_qwm_agent', 0, True, (1, 1)),
        'others': [],
    }

    action = callbacks.act(self_obj, game_state)
    assert action in ACTIONS

    trace = get_latest_overlay_trace()
    assert trace is not None
    assert trace.round_id == 1
    assert trace.env_step == 1
    assert trace.planner_action == action
    assert len(trace.candidates) == N_ACTIONS
    assert trace.candidates[0].rank == 0
    assert len(trace.candidates[0].actions) == 2  # Depth 2 action sequence
    assert len(trace.candidates[0].tiles) >= 2    # Tiles walked

    # Test clean degrade on mock screen/gui
    class MockScreen:
        pass
    class MockGUI:
        def render_text(self, *args, **kwargs):
            pass

    render_overlay(MockScreen(), MockGUI())
    clear_overlay_trace()
    assert get_latest_overlay_trace() is None


def test_update_multistep_paths():
    agent = new_model()
    B = 4
    H = 4
    spatials = np.random.randn(B, H + 1, 10, 17, 17).astype(np.float32)
    features = np.random.randn(B, H + 1, 33).astype(np.float32)
    actions = np.random.randint(0, N_ACTIONS, size=(B, H))
    rewards = np.random.randn(B, H).astype(np.float32)
    dones = np.zeros((B, H), dtype=np.float32)

    metrics = agent.update_multistep_paths(spatials, features, actions, rewards, dones)
    assert metrics["wm_path_loss"] > 0.0
    assert metrics["trans_loss"] >= 0.0
    assert metrics["rew_loss"] >= 0.0
    assert metrics["future_val_loss"] >= 0.0
    assert "legal_loss" in metrics
    assert metrics["legal_loss"] >= 0.0
    assert len(metrics["horizon_mae"]) == H
    assert len(metrics["horizon_rmse"]) == H
    assert len(metrics["pred_vals_mean"]) == H
    assert len(metrics["actual_vals_mean"]) == H


def test_update_multistep_paths_random_horizons_1_through_8():
    agent = new_model()
    B = 4
    max_H = 8
    spatials = np.random.randn(B, max_H + 1, 10, 17, 17).astype(np.float32)
    features = np.random.randn(B, max_H + 1, 33).astype(np.float32)
    actions = np.random.randint(0, N_ACTIONS, size=(B, max_H))
    rewards = np.random.randn(B, max_H).astype(np.float32)
    dones = np.zeros((B, max_H), dtype=np.float32)

    # 1. Test random horizons 1 through 8
    observed_horizons = set()
    for _ in range(25):
        metrics = agent.update_multistep_paths(spatials, features, actions, rewards, dones, random_horizon=True)
        h = metrics["horizon"]
        assert 1 <= h <= 8
        assert len(metrics["horizon_mae"]) == h
        observed_horizons.add(h)
    assert len(observed_horizons) > 1, f"Expected multiple sampled horizons, got {observed_horizons}"

    # 2. Test explicit horizons 1 through 8
    for target_h in range(1, 9):
        metrics_h = agent.update_multistep_paths(spatials, features, actions, rewards, dones, horizon=target_h)
        assert metrics_h["horizon"] == target_h
        assert len(metrics_h["horizon_mae"]) == target_h


def test_evaluate_future_values():
    agent = new_model()
    B = 4
    H = 3
    spatials = np.random.randn(B, H + 1, 10, 17, 17).astype(np.float32)
    features = np.random.randn(B, H + 1, 33).astype(np.float32)
    actions = np.random.randint(0, N_ACTIONS, size=(B, H))
    rewards = np.random.randn(B, H).astype(np.float32)
    dones = np.zeros((B, H), dtype=np.float32)

    eval_res = agent.evaluate_future_values(spatials, features, actions, rewards, dones)
    assert "horizons" in eval_res
    assert len(eval_res["horizons"]) == H
    for h_data in eval_res["horizons"]:
        assert "pred_mean" in h_data
        assert "actual_mean" in h_data
        assert "mae" in h_data
        assert "rmse" in h_data


def test_plot_training_8_panels_and_comparison(tmp_path):
    from agent_code.my_spatial_qwm_agent.plot_training import (
        plot_stats_file, plot_future_value_comparison
    )
    import pandas as pd

    # 1. Create mock CSV with future_val_loss
    df_data = {
        'round': [1, 2, 3],
        'stage': ['s0_wm_warmup', 's0_wm_warmup', 's1_coin_heaven'],
        'epsilon': [0.05, 0.05, 0.5],
        'score': [15, 25, 45],
        'coins': [15, 25, 45],
        'kills': [0, 0, 0],
        'crates': [30, 40, 50],
        'bombs': [10, 15, 20],
        'survived': [0, 1, 1],
        'steps': [200, 250, 300],
        'invalid_acts': [0, 1, 0],
        'killed_self': [0, 0, 0],
        'got_killed': [1, 0, 0],
        'mean_reward': [0.4, 0.5, 0.6],
        'sum_reward': [80.0, 125.0, 180.0],
        'mean_td_error': [0.0, 0.0, 0.05],
        'wm_loss': [0.35, 0.25, 0.15],
        'trans_loss': [0.05, 0.04, 0.03],
        'rew_loss': [0.30, 0.21, 0.12],
        'future_val_loss': [0.45, 0.32, 0.18],
        'dqn_frozen': [1, 1, 0],
        'wall_time_s': [5.0, 6.0, 7.0],
    }
    csv_file = tmp_path / "test_stats.csv"
    pd.DataFrame(df_data).to_csv(csv_file, index=False)

    out_curves = tmp_path / "curves.png"
    assert plot_stats_file(str(csv_file), str(out_curves))
    assert out_curves.is_file()
    assert out_curves.stat().st_size > 1000

    # 2. Test second graph: plot_future_value_comparison
    out_comparison = tmp_path / "future_val_comp.png"
    assert plot_future_value_comparison(None, str(out_comparison))
    assert out_comparison.is_file()
    assert out_comparison.stat().st_size > 1000


def test_cuda_amp_updates():
    """Verify that update_batch_cuda and update_multistep_paths_cuda run cleanly without scaler errors."""
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    agent = SpatialQWMAgent(device=device)

    # 1. Test update_batch_cuda with frozen DQN
    agent.freeze_dqn()
    s = torch.randn(8, 10, 17, 17, device=device)
    f = torch.randn(8, 33, device=device)
    a = torch.randint(0, N_ACTIONS, (8, 1), device=device)
    r = torch.randn(8, 1, device=device)
    ns = torch.randn(8, 10, 17, 17, device=device)
    nf = torch.randn(8, 33, device=device)
    d = torch.zeros(8, 1, device=device)

    loss_dict = agent.update_batch_cuda(s, f, a, r, ns, nf, d)
    assert "wm_loss" in loss_dict
    assert loss_dict["loss"] == 0.0

    # 2. Test update_batch_cuda with unfrozen DQN (tests both scalers in the same step)
    agent.unfreeze_dqn()
    loss_dict = agent.update_batch_cuda(s, f, a, r, ns, nf, d)
    assert loss_dict["loss"] > 0.0
    assert loss_dict["wm_loss"] > 0.0

    # 3. Test update_multistep_paths_cuda immediately after batch update
    B, H = 4, 3
    spatials = torch.randn(B, H + 1, 10, 17, 17, device=device)
    features = torch.randn(B, H + 1, 33, device=device)
    actions = torch.randint(0, N_ACTIONS, (B, H), device=device)
    rewards = torch.randn(B, H, device=device)
    dones = torch.zeros(B, H, device=device)

    metrics = agent.update_multistep_paths_cuda(spatials, features, actions, rewards, dones)
    assert metrics["wm_path_loss"] > 0.0
    assert metrics["future_val_loss"] >= 0.0
    assert "legal_loss" in metrics
    assert metrics["legal_loss"] >= 0.0
    assert len(metrics["horizon_mae"]) == H


def test_action_legality_head_and_pruning():
    from agent_code.my_spatial_qwm_agent.model import get_action_legality_targets, F_BOMBS_LEFT, F_BLOCKED
    agent = new_model()

    # 1. Test get_action_legality_targets logic
    feat = torch.zeros((2, 33))
    # Batch 0: directions 0 (UP), 1 (RIGHT) blocked; bombs_left = 1
    feat[0, F_BLOCKED + 0] = 1.0  # UP blocked
    feat[0, F_BLOCKED + 1] = 1.0  # RIGHT blocked
    feat[0, F_BOMBS_LEFT] = 1.0   # bombs_left = 1
    # Batch 1: direction 2 (DOWN) blocked; bombs_left = 0
    feat[1, F_BLOCKED + 2] = 1.0  # DOWN blocked
    feat[1, F_BOMBS_LEFT] = 0.0   # bombs_left = 0

    targets = get_action_legality_targets(feat)
    assert targets.shape == (2, N_ACTIONS)
    assert targets[0, 0] == 0.0  # UP blocked -> illegal
    assert targets[0, 1] == 0.0  # RIGHT blocked -> illegal
    assert targets[0, 2] == 1.0  # DOWN free -> legal
    assert targets[0, 3] == 1.0  # LEFT free -> legal
    assert targets[0, 4] == 1.0  # WAIT -> always legal
    assert targets[0, 5] == 1.0  # BOMB -> legal

    assert targets[1, 2] == 0.0  # DOWN blocked -> illegal
    assert targets[1, 5] == 0.0  # no bombs -> illegal

    # 2. Test that illegal future moves are strictly pruned in tree search
    spatial = np.random.randn(10, 17, 17).astype(np.float32)
    features = np.zeros(33, dtype=np.float32)
    features[F_BOMBS_LEFT] = 1.0  # bombs_left = 1
    valid_mask = np.array([True, True, True, True, True, True], dtype=bool)

    # Force world_model.legal_head to predict negative logits for action 0 (UP)
    # and action 5 (BOMB) in all future states
    with torch.no_grad():
        agent.world_model.legal_head[-1].weight.data.zero_()
        agent.world_model.legal_head[-1].bias.data.fill_(100.0)
        agent.world_model.legal_head[-1].bias.data[0] = -100.0  # UP illegal
        agent.world_model.legal_head[-1].bias.data[5] = -100.0  # BOMB illegal

    q4, details = agent.tree_search_action_scores(
        spatial, features, depth=4, beam_size=12, discount=0.5, valid_mask=valid_mask, return_details=True
    )
    # For any root action that looks ahead, subsequent actions (step 1, 2, 3) must NOT pick UP (0) or BOMB (5)
    for d in details:
        acts = d['actions']
        for future_act in acts[1:]:
            assert future_act != 'UP', f"Hallucinated illegal UP move: {acts}"
            assert future_act != 'BOMB', f"Hallucinated illegal BOMB move: {acts}"


def make_test_arena():
    arena = np.zeros((17, 17), dtype=np.int32)
    arena[0, :] = -1
    arena[-1, :] = -1
    arena[:, 0] = -1
    arena[:, -1] = -1
    for x in range(2, 16, 2):
        for y in range(2, 16, 2):
            arena[x, y] = -1
    return arena


def test_multi_agent_crossing_bombs():
    from agent_code.my_spatial_qwm_agent.callbacks import get_blast_map, find_escape_moves
    arena = make_test_arena()
    # Bomb 1 at (1, 3) timer 3 (horizontal blast along y=3: (2,3), (3,3), (4,3))
    # Bomb 2 at (3, 1) timer 2 (vertical blast along x=3: (3,2), (3,3), (3,4))
    bombs = [((1, 3), 3), ((3, 1), 2)]
    # Block (3, 4) with a crate, so agent cannot escape down
    arena[3, 4] = 1

    blast_map = get_blast_map(arena, bombs)

    # Intersection is at (3, 3)
    assert blast_map[3, 3] == 2, f"Expected min timer 2 at intersection (3, 3), got {blast_map[3, 3]}"
    assert blast_map[1, 3] == 3
    assert blast_map[3, 1] == 2

    # Agent is at (3, 3) (in crossing blast zone)
    in_danger, escape_moves = find_escape_moves(arena, bombs, None, (3, 3))
    assert in_danger is True
    assert len(escape_moves) > 0
    # RIGHT is index 1 in DIRS, leading to safe tile at (5, 3)
    assert any(a == 1 for a, _ in escape_moves), f"Expected RIGHT in escape moves, got {escape_moves}"


def test_trapped_corridor_between_two_bombs():
    from agent_code.my_spatial_qwm_agent.callbacks import find_escape_moves
    arena = make_test_arena()
    # Bomb 1 at (3, 1) with timer 4
    # Bomb 2 at (3, 3) with timer 4
    # Corridor between them is (3, 2).
    # Walls or crates at (2, 2) and (4, 2)
    arena[2, 2] = -1
    arena[4, 2] = 1  # crate
    bombs = [((3, 1), 4), ((3, 3), 4)]

    # Agent at (3, 2) is completely trapped between two bombs with no side exits
    in_danger, escape_moves = find_escape_moves(arena, bombs, None, (3, 2))
    assert in_danger is True
    assert len(escape_moves) == 0, f"Trapped agent should have 0 escape moves, got {escape_moves}"


def test_opponent_trapping_evaluation():
    from agent_code.my_spatial_qwm_agent.callbacks import check_opponent_trapped_or_threatened
    arena = make_test_arena()
    # Opponent is at (1, 1), corner tile.
    # (0, 1) and (1, 0) are border walls (-1).
    # (2, 1) is blocked with a crate (1):
    arena[2, 1] = 1
    # If agent places a bomb at (1, 2), opponent at (1, 1) has no way out except through the bomb!
    others_raw = [('opp', 0, True, (1, 1))]
    bombs = []
    exp_map = None

    opp_trapped, opp_restricted, n_threat = check_opponent_trapped_or_threatened(
        arena, (1, 2), bombs, exp_map, others_raw, (1, 3)
    )
    assert opp_trapped is True, "Expected opponent at (1, 1) to be trapped by bomb at (1, 2)"
    assert n_threat == 1


def test_standoff_distance_vs_armed_opponent():
    from agent_code.my_spatial_qwm_agent.callbacks import find_escape_moves
    arena = make_test_arena()
    # Opponent is at (3, 3)
    # Agent is considering moving into (3, 2)
    # If opponent drops bomb at (3, 3), check escape from (3, 2)
    arena[3, 1] = -1
    arena[2, 2] = -1
    arena[4, 2] = -1
    hypo_opp_bombs = [((3, 3), 4)]
    _, opp_trap_esc = find_escape_moves(arena, hypo_opp_bombs, None, (3, 2), ((3, 3),))
    assert len(opp_trap_esc) == 0, "Agent in dead-end corridor adjacent to armed opponent must have 0 escapes"


def test_fire_survival_escape():
    from agent_code.my_spatial_qwm_agent import callbacks
    arena = make_test_arena()
    arena[2, 1] = -1  # Block RIGHT from (1, 1)
    arena[1, 0] = -1  # Block UP from (1, 1)
    arena[0, 1] = -1  # Block LEFT from (1, 1)

    # DOWN from (1, 1) is (1, 2)
    arena[0, 2] = -1  # Block LEFT from (1, 2)
    arena[1, 3] = -1  # Block DOWN from (1, 2)
    arena[2, 2] = 1   # Crate at (2, 2) blocks RIGHT from (1, 2)

    bombs = [((3, 2), 3)]  # Bomb timer 3 threatens (1, 2)
    exp_map = np.zeros((17, 17), dtype=int)
    exp_map[1, 1] = 1  # (1, 1) is ON FIRE

    class DummyAgent:
        train = False

    self_obj = DummyAgent()
    callbacks.setup(self_obj)
    self_obj.tree_search = False

    game_state = {
        'round': 1,
        'step': 10,
        'field': arena,
        'bombs': bombs,
        'explosion_map': exp_map,
        'coins': [],
        'self': ('my_spatial_qwm_agent', 0, True, (1, 1)),
        'others': [],
    }

    action = callbacks.act(self_obj, game_state)
    assert action in ('DOWN', 'WAIT', 'BOMB')


def test_active_penalties_detection():
    import events as events_module
    from agent_code.my_spatial_qwm_agent import train
    from agent_code.my_spatial_qwm_agent.model import state_to_features

    arena = make_test_arena()
    # Scenario 1: Dropping bomb while in danger
    bombs = [((2, 1), 3)]  # Bomb adjacent to (1, 1)
    old_state = {
        'round': 1, 'step': 1,
        'field': arena, 'bombs': bombs, 'explosion_map': np.zeros((17, 17), dtype=int),
        'coins': [], 'self': ('agent', 0, True, (1, 1)), 'others': []
    }
    old_f = state_to_features(old_state)

    events_1 = []
    extra_1 = train.custom_events(None, old_state, 'BOMB', None, old_f, None, events_1)
    assert train.DROPPED_BOMB_IN_DANGER in extra_1, f"Expected DROPPED_BOMB_IN_DANGER, got {extra_1}"

    # Scenario 2: Waiting in danger when escape route exists
    events_2 = []
    extra_2 = train.custom_events(None, old_state, 'WAIT', None, old_f, None, events_2)
    assert train.WAITED_IN_DANGER_WITH_ESCAPE in extra_2, f"Expected WAITED_IN_DANGER_WITH_ESCAPE, got {extra_2}"

    # Scenario 3: Stepping into active blast from safe tile
    # Bomb at (1, 1) blasts along x=1 and y=1. (2, 2) is diagonal, completely safe.
    # Stepping LEFT from (2, 2) moves onto (1, 2), which is in the blast line of (1, 1).
    safe_state = {
        'round': 1, 'step': 1,
        'field': arena, 'bombs': [((1, 1), 1)], 'explosion_map': np.zeros((17, 17), dtype=int),
        'coins': [], 'self': ('agent', 0, True, (2, 2)), 'others': []
    }
    safe_f = state_to_features(safe_state)
    events_3 = []
    extra_3 = train.custom_events(None, safe_state, 'LEFT', None, safe_f, None, events_3)
    assert train.STEPPED_INTO_BLAST in extra_3, f"Expected STEPPED_INTO_BLAST, got {extra_3}"

    # Scenario 4: Suicidal bomb placement in a 1-tile dead end
    dead_end_arena = make_test_arena()
    dead_end_arena[2, 1] = -1
    dead_end_arena[1, 2] = -1
    dead_end_arena[0, 1] = -1
    dead_end_arena[1, 0] = -1
    dead_end_state = {
        'round': 1, 'step': 1,
        'field': dead_end_arena, 'bombs': [], 'explosion_map': np.zeros((17, 17), dtype=int),
        'coins': [], 'self': ('agent', 0, True, (1, 1)), 'others': []
    }
    dead_end_f = state_to_features(dead_end_state)
    events_4 = []
    extra_4 = train.custom_events(None, dead_end_state, 'BOMB', None, dead_end_f, None, events_4)
    assert train.SUICIDAL_BOMB in extra_4, f"Expected SUICIDAL_BOMB in dead end, got {extra_4}"

    # Scenario 5: Fleeing from danger to safety
    in_danger_state = {
        'round': 1, 'step': 1,
        'field': arena, 'bombs': [((1, 1), 3)], 'explosion_map': np.zeros((17, 17), dtype=int),
        'coins': [], 'self': ('agent', 0, True, (1, 2)), 'others': []
    }
    escaped_state = {
        'round': 1, 'step': 2,
        'field': arena, 'bombs': [((1, 1), 2)], 'explosion_map': np.zeros((17, 17), dtype=int),
        'coins': [], 'self': ('agent', 0, True, (2, 2)), 'others': []
    }
    events_5 = []
    extra_5 = train.custom_events(
        None, in_danger_state, 'RIGHT', escaped_state,
        state_to_features(in_danger_state), state_to_features(escaped_state), events_5
    )
    # Scenario 6: Idle waiting and standing around when safe and targets exist
    safe_with_targets = {
        'round': 1, 'step': 1,
        'field': arena, 'bombs': [], 'explosion_map': np.zeros((17, 17), dtype=int),
        'coins': [(5, 5)], 'self': ('agent', 0, True, (2, 2)), 'others': []
    }
    class MockTrainer:
        _consecutive_stationary_steps = 2
    mock_tr = MockTrainer()
    events_6 = []
    extra_6 = train.custom_events(mock_tr, safe_with_targets, 'WAIT', None, state_to_features(safe_with_targets), None, events_6)
    assert train.IDLE_WAIT in extra_6, f"Expected IDLE_WAIT, got {extra_6}"
    assert train.STANDING_AROUND in extra_6, f"Expected STANDING_AROUND, got {extra_6}"

    # Scenario 7: Cowardice penalty at end_of_round
    class MockAgentEOR:
        coordinate_history = []
        _round_crates = 0
        _round_coins = 0
        _round_kills = 0
        _round_bombs = 0
        _round_invalid = 0
        _round_killed_self = 0
        _round_got_killed = 0
        _round_rewards = []
        transitions = []
        _total_rounds = 0
        epsilon = 0.1
        eps_end = 0.05
        eps_decay = 0.99
    ag_eor = MockAgentEOR()
    eor_events = [events_module.SURVIVED_ROUND]
    train.end_of_round(ag_eor, safe_with_targets, 'WAIT', eor_events)
    assert events_module.SURVIVED_ROUND not in eor_events
    assert train.COWARDICE_PENALTY in eor_events

    # Scenario 8: Hazard proximity check in callbacks
    from agent_code.my_spatial_qwm_agent.callbacks import is_hazard_nearby
    # No hazards anywhere
    assert not is_hazard_nearby((2, 2), arena, [], np.zeros((17, 17), dtype=int), radius=4)
    # Hazard nearby: bomb at (2, 4)
    assert is_hazard_nearby((2, 2), arena, [((2, 4), 3)], np.zeros((17, 17), dtype=int), radius=4)
    # Hazard nearby: flame at (2, 3)
    exp_with_flame = np.zeros((17, 17), dtype=int)
    exp_with_flame[2, 3] = 1
    assert is_hazard_nearby((2, 2), arena, [], exp_with_flame, radius=4)

    # Scenario 9: Won match detection, reward, and stats tracking
    assert train.compute_reward([train.WON_ROUND]) == 40.0
    assert train.compute_reward([train.LOST_ROUND]) == -25.0

    ag_won = MockAgentEOR()
    ag_won._round_coins = 3
    ag_won._round_crates = 4
    won_state = {
        'round': 1, 'step': 120,
        'field': arena, 'bombs': [], 'explosion_map': np.zeros((17, 17), dtype=int),
        'coins': [], 'self': ('agent', 3, True, (2, 2)), 'others': [('other', 1, False, (5, 5))]
    }
    won_events = [events_module.WON_ROUND, events_module.SURVIVED_ROUND]
    train.end_of_round(ag_won, won_state, 'WAIT', won_events)
    assert train.WON_ROUND in won_events
    assert getattr(ag_won, '_last_round_won', 0) == 1
    # Check that terminal reward includes WON_ROUND (+40)
    assert ag_won.transitions[-1][3] >= 40.0

    ag_lost = MockAgentEOR()
    ag_lost._round_coins = 1
    ag_lost._round_crates = 1
    lost_state = {
        'round': 1, 'step': 150,
        'field': arena, 'bombs': [], 'explosion_map': np.zeros((17, 17), dtype=int),
        'coins': [], 'self': ('agent', 1, True, (2, 2)), 'others': [('other', 5, True, (5, 5))]
    }
    lost_events = [events_module.SURVIVED_ROUND]
    train.end_of_round(ag_lost, lost_state, 'WAIT', lost_events)
    assert train.LOST_ROUND in lost_events
    assert getattr(ag_lost, '_last_round_won', 1) == 0
    assert ag_lost.transitions[-1][3] <= 0.0

    # Scenario 10: Active loop and direct reversal detection
    assert train.compute_reward([train.LOOP_DETECTED]) == -8.0
    assert train.compute_reward([train.REVERSED_MOVE]) == -5.0
    assert train.compute_reward([train.SEVERE_LOOP_DETECTED]) == -12.0
    assert train.compute_reward([train.TILE_REVISITED]) == -3.0

    class MockAgentLoop:
        coordinate_history = [(1, 1), (1, 2)]

    ag_loop = MockAgentLoop()
    state_at_1_2 = {
        'round': 1, 'step': 2,
        'field': arena, 'bombs': [], 'explosion_map': np.zeros((17, 17), dtype=int),
        'coins': [(5, 5)], 'self': ('agent', 0, True, (1, 2)), 'others': []
    }
    state_at_1_1 = {
        'round': 1, 'step': 3,
        'field': arena, 'bombs': [], 'explosion_map': np.zeros((17, 17), dtype=int),
        'coins': [(5, 5)], 'self': ('agent', 0, True, (1, 1)), 'others': []
    }
    events_rev = []
    extra_rev = train.custom_events(
        ag_loop, state_at_1_2, 'LEFT', state_at_1_1,
        state_to_features(state_at_1_2), state_to_features(state_at_1_1), events_rev
    )
    assert train.REVERSED_MOVE in extra_rev, f"Expected REVERSED_MOVE in {extra_rev}"

    # Cycle test: [(1, 1), (1, 2), (1, 1)] moving to (1, 2)
    ag_loop2 = MockAgentLoop()
    ag_loop2.coordinate_history = [(1, 1), (1, 2), (1, 1)]
    events_cycle = []
    extra_cycle = train.custom_events(
        ag_loop2, state_at_1_1, 'RIGHT', state_at_1_2,
        state_to_features(state_at_1_1), state_to_features(state_at_1_2), events_cycle
    )
    assert train.LOOP_DETECTED in extra_cycle, f"Expected LOOP_DETECTED in {extra_cycle}"
    assert train.REVERSED_MOVE in extra_cycle, f"Expected REVERSED_MOVE in {extra_cycle}"


def test_small_path_reward_model(tmp_path):
    import tempfile
    from agent_code.my_spatial_qwm_agent.path_penalty_model import (
        FEATURE_DIM, SmallPathRewardModel, save_small_model, load_small_model
    )
    from agent_code.my_spatial_qwm_agent import callbacks
    from agent_code.my_spatial_qwm_agent.model import dynActivation, ACTIONS

    # 1. Architecture and parameter count
    model = SmallPathRewardModel(input_dim=FEATURE_DIM, hidden_dim=64)
    param_count = sum(p.numel() for p in model.parameters())
    assert 5000 <= param_count <= 6500, f"Expected ~5,893 parameters, got {param_count}"

    # 2. Forward pass with 2D and 3D shapes
    x2d = torch.randn(8, FEATURE_DIM)
    out2d = model(x2d)
    assert out2d.shape == (8, 1)

    x3d = torch.randn(4, 6, FEATURE_DIM)
    out3d = model(x3d)
    assert out3d.shape == (4, 6)

    # 3. dynActivation gradient check
    dyn_act = dynActivation()
    assert hasattr(dyn_act, 'alpha')
    test_tensor = torch.randn(5, 5, requires_grad=True)
    out_act = dyn_act(test_tensor)
    loss = out_act.sum()
    loss.backward()
    assert dyn_act.alpha.grad is not None
    assert test_tensor.grad is not None

    # 4. Save and load roundtrip
    temp_ckpt = str(tmp_path / "test_small_model.pt")
    save_small_model(model, temp_ckpt)
    assert os.path.isfile(temp_ckpt)
    loaded_model = load_small_model(temp_ckpt, device="cpu")
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)

    # 5. predict_deltas method
    features_np = np.random.randn(6, FEATURE_DIM).astype(np.float32)
    deltas = loaded_model.predict_deltas(features_np, device="cpu")
    assert isinstance(deltas, np.ndarray)
    assert deltas.shape == (6,)

    # 6. Integration in callbacks.act
    class DummyAgent:
        train = False

    self_obj = DummyAgent()
    callbacks.setup(self_obj)
    self_obj.small_model = loaded_model
    self_obj.tree_search = False

    arena = make_test_arena()
    game_state = {
        'round': 1,
        'step': 1,
        'field': arena,
        'bombs': [],
        'explosion_map': np.zeros((17, 17), dtype=int),
        'coins': [(3, 3)],
        'self': ('agent', 0, True, (1, 1)),
        'others': [],
    }
    action = callbacks.act(self_obj, game_state)
    assert action in ACTIONS


def test_qwm_agent_clean_no_hardcoded_policies():
    from agent_code.qwm_agent_clean import callbacks as clean_callbacks
    from agent_code.qwm_agent_clean.model import ACTIONS

    # 1. Verify no hardcoded policy functions exist in clean callbacks
    forbidden_heuristics = [
        'find_escape_moves',
        'get_blast_map',
        'detect_cycle',
        'is_hazard_nearby',
        '_trace_penalty',
    ]
    for fn_name in forbidden_heuristics:
        assert not hasattr(clean_callbacks, fn_name), f"Forbidden hardcoded heuristic '{fn_name}' found in qwm_agent_clean!"

    # 2. Setup agent and verify main model + small model are loaded
    class CleanAgent:
        train = False

    agent = CleanAgent()
    clean_callbacks.setup(agent)
    assert agent.model is not None, "Main QWM model was not loaded!"
    assert agent.small_model is not None, "Small path reward model was not loaded!"

    # 3. Test act() execution using purely neural scoring
    arena = make_test_arena()
    game_state = {
        'round': 1,
        'step': 1,
        'field': arena,
        'bombs': [],
        'explosion_map': np.zeros((17, 17), dtype=int),
        'coins': [(3, 3)],
        'self': ('qwm_agent_clean', 0, True, (1, 1)),
        'others': [],
    }
    action = clean_callbacks.act(agent, game_state)
    assert action in ACTIONS


def test_curriculum_stages_and_resolution():
    """Verifies that all main curriculum stages exist and resolve_stages handles aliases."""
    import settings as s
    from agent_code.my_spatial_qwm_agent.train_small_model_phase2 import (
        ALL_CURRICULUM_STAGES,
        STAGE_ALIASES,
        TARGET_ENVIRONMENTS,
        resolve_stages,
    )

    # 1. Verify all 8 main curriculum stages exist
    expected_curriculum = [
        "s0_wm_warmup",
        "s1_coin_heaven",
        "s2_classic_alone",
        "s3_vs_passive",
        "s4_vs_rule_based",
        "s5_vs_rule_based_loot",
        "s6_vs_mixed_loot",
        "s7_vs_mixed_classic",
    ]
    assert ALL_CURRICULUM_STAGES == expected_curriculum

    # 2. Verify all environments have valid scenarios and opponent configurations
    for st_name in expected_curriculum:
        assert st_name in TARGET_ENVIRONMENTS, f"Missing stage: {st_name}"
        env = TARGET_ENVIRONMENTS[st_name]
        assert env["scenario"] in s.SCENARIOS, f"Invalid scenario {env['scenario']} in stage {st_name}"
        assert isinstance(env["opponents"], list)
        assert len(env["opponents"]) >= 1

    # 3. Test alias resolutions
    assert resolve_stages(None) == expected_curriculum
    assert resolve_stages(["all"]) == expected_curriculum
    assert resolve_stages(["curriculum"]) == expected_curriculum
    assert resolve_stages(["combat"]) == ["s4_vs_rule_based", "s5_vs_rule_based_loot", "s6_vs_mixed_loot", "s7_vs_mixed_classic"]
    assert resolve_stages(["loot"]) == ["s0_wm_warmup", "s2_classic_alone", "s3_vs_passive", "s5_vs_rule_based_loot", "s6_vs_mixed_loot"]
    assert resolve_stages(["coins"]) == ["s1_coin_heaven"]
    assert resolve_stages(["self_play"]) == ["sp_classic", "sp_loot", "sp_coins"]
    assert resolve_stages(["s1", "s4"]) == ["s1_coin_heaven", "s4_vs_rule_based"]
    assert resolve_stages(["0", "7"]) == ["s0_wm_warmup", "s7_vs_mixed_classic"]



