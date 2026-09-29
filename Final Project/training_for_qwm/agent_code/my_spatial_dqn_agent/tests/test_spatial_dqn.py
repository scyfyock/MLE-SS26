"""
test_spatial_dqn.py -- Comprehensive unit test suite for my_spatial_dqn_agent.
"""

import os
import sys
import unittest
from collections import deque

import numpy as np
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import events as e
from agent_code.my_spatial_dqn_agent.callbacks import act, detect_cycle, setup
from agent_code.my_spatial_dqn_agent.model import (
    ACTIONS,
    ACTION_TO_IDX,
    BOARD_H,
    BOARD_W,
    F_SAFE_WAIT,
    N_ACTIONS,
    N_FEATURES,
    N_SPATIAL_CHANNELS,
    SpatialDuelingDQN,
    SpatialDQNAgent,
    augment_spatial_transition,
    new_model,
    state_to_features,
    state_to_spatial_tensor,
)
from agent_code.my_spatial_dqn_agent.train import (
    MOVED_TOWARD_COIN,
    MOVED_TOWARD_OPPONENT,
    NO_PROGRESS,
    SPARSE_SUICIDE_PENALTY,
    STAGNATION,
    STANDING_AROUND,
    custom_events,
    end_of_round,
    is_better_round,
    reward_from_events,
    setup_training,
)


class DummyLogger:
    def info(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass


class MockAgent:
    def __init__(self, train=False):
        self.train = train
        self.logger = DummyLogger()


def make_arena():
    a = np.zeros((17, 17), int)
    a[:1, :] = a[-1:, :] = a[:, :1] = a[:, -1:] = -1
    for x in range(17):
        for y in range(17):
            if (x + 1) * (y + 1) % 2 == 1:
                a[x, y] = -1
    return a


def make_state(field, pos, step=1, round_num=1, coins=(), bombs=(), others=()):
    return {
        'round': round_num,
        'step': step,
        'field': field,
        'bombs': list(bombs),
        'explosion_map': np.zeros_like(field, dtype=float),
        'coins': list(coins),
        'self': ('spatial_agent', 0, True, pos),
        'others': [('o%d' % i, 0, True, p) for i, p in enumerate(others)],
        'user_input': None,
    }


class TestSpatialDQNAgent(unittest.TestCase):
    def setUp(self):
        self.field = make_arena()
        self.field[3, 1] = 1  # Crate
        self.state = make_state(self.field, (1, 1), step=1, coins=[(1, 3)], bombs=[((1, 5), 3)])

    def test_spatial_tensor_extraction(self):
        """Verifies full-board spatial representation extraction."""
        coord_hist = deque([(1, 1), (1, 2), (1, 1)], maxlen=40)
        spatial = state_to_spatial_tensor(self.state, coord_hist)

        self.assertEqual(spatial.shape, (N_SPATIAL_CHANNELS, BOARD_W, BOARD_H))
        # Channel 0: Walls
        self.assertEqual(spatial[0, 0, 0], 1.0)
        self.assertEqual(spatial[0, 1, 1], 0.0)
        # Channel 1: Crate at (3, 1)
        self.assertEqual(spatial[1, 3, 1], 1.0)
        # Channel 5: Coin at (1, 3)
        self.assertEqual(spatial[5, 1, 3], 1.0)
        # Channel 6: Self at (1, 1)
        self.assertEqual(spatial[6, 1, 1], 1.0)
        # Channel 8: Heatmap at (1, 1)
        self.assertGreater(spatial[8, 1, 1], 0.0)
        # Channel 9: Global BFS distance to nearest target
        self.assertGreater(spatial[9, 1, 3], 0.0)

    def test_network_forward_pass(self):
        """Verifies Dueling Spatial DQN forward pass."""
        net = SpatialDuelingDQN()
        spatial_batch = torch.zeros((4, N_SPATIAL_CHANNELS, BOARD_W, BOARD_H))
        features_batch = torch.zeros((4, N_FEATURES))

        q_out = net(spatial_batch, features_batch)
        self.assertEqual(q_out.shape, (4, N_ACTIONS))
        self.assertTrue(torch.all(torch.isfinite(q_out)))

    def test_agent_update_batch_and_polyak(self):
        """Verifies DQN loss calculation and target network Polyak update."""
        agent = new_model(lr=1e-4, device=torch.device('cpu'))
        B = 4
        S = np.zeros((B, N_SPATIAL_CHANNELS, BOARD_W, BOARD_H), dtype=np.float32)
        F = np.zeros((B, N_FEATURES), dtype=np.float32)
        A = np.array([0, 1, 2, 4], dtype=np.int64)
        R = np.array([1.0, 0.0, -1.0, 5.0], dtype=np.float32)
        NS = np.zeros((B, N_SPATIAL_CHANNELS, BOARD_W, BOARD_H), dtype=np.float32)
        NF = np.zeros((B, N_FEATURES), dtype=np.float32)
        D = np.array([False, False, False, True], dtype=bool)

        loss = agent.update_batch(S, F, A, R, NS, NF, D)
        self.assertGreater(loss, 0.0)
        self.assertFalse(np.isnan(loss))

    def test_d4_symmetry_augmentation(self):
        """Verifies dihedral D4 symmetry produces 8 consistent transformed transitions."""
        s = np.zeros((N_SPATIAL_CHANNELS, BOARD_W, BOARD_H), dtype=np.float32)
        f = np.zeros(N_FEATURES, dtype=np.float32)
        ns = np.zeros((N_SPATIAL_CHANNELS, BOARD_W, BOARD_H), dtype=np.float32)
        nf = np.zeros(N_FEATURES, dtype=np.float32)

        aug_list = augment_spatial_transition(s, f, 0, ns, nf)
        self.assertEqual(len(aug_list), 8)
        for aspat, afeat, aact, anspat, anfeat in aug_list:
            self.assertEqual(aspat.shape, (N_SPATIAL_CHANNELS, BOARD_W, BOARD_H))
            self.assertEqual(afeat.shape, (N_FEATURES,))
            self.assertIn(aact, range(N_ACTIONS))

    def test_detect_cycle_unit(self):
        """Verifies multi-period cycle detection logic."""
        # 2-step ping-pong
        is_c, k, reps = detect_cycle([(1, 1), (1, 2), (1, 1), (1, 2)])
        self.assertTrue(is_c)
        self.assertEqual(k, 2)

        # 4-step corridor loop: A -> B -> C -> B -> A -> B -> C -> B
        A, B, C = (1, 1), (1, 2), (1, 3)
        is_c, k, reps = detect_cycle([A, B, C, B, A, B, C, B])
        self.assertTrue(is_c)
        self.assertEqual(k, 4)

        # Non-loop
        is_c, _, _ = detect_cycle([(1, 1), (1, 2), (1, 3), (1, 4)])
        self.assertFalse(is_c)

    def test_immediate_non_progress_penalty(self):
        """Verifies non-progress penalty sets in directly on step 1 and increases over time."""
        agent = MockAgent(train=True)
        setup(agent)
        setup_training(agent)

        old_f = np.zeros(N_FEATURES, dtype=np.float32)
        old_f[F_SAFE_WAIT] = 1.0

        # Step 1: NO_PROGRESS must trigger directly on step 1
        s1 = make_state(self.field, (1, 1), step=1)
        extra1 = custom_events(agent, s1, 'WAIT', s1, old_f, old_f, [e.WAITED])
        self.assertIn(NO_PROGRESS, extra1)
        self.assertEqual(agent._steps_without_progress, 1)

        # Step 5: penalty count increases over time
        for step in range(2, 6):
            s = make_state(self.field, (1, 1), step=step)
            extra = custom_events(agent, s, 'WAIT', s, old_f, old_f, [e.WAITED])
        self.assertGreater(extra.count(NO_PROGRESS), extra1.count(NO_PROGRESS))

        # Progress event resets
        extra_coin = custom_events(agent, s1, 'WAIT', s1, old_f, old_f, [e.COIN_COLLECTED])
        self.assertEqual(agent._steps_without_progress, 0)
        self.assertNotIn(NO_PROGRESS, extra_coin)

    def test_act_inference_and_idle_wait_suppression(self):
        """Verifies act() makes decisions and suppresses idle WAIT when safe moves exist."""
        agent = MockAgent(train=False)
        setup(agent)

        s = make_state(self.field, (1, 1), step=1)
        action = act(agent, s)
        self.assertIn(action, ACTIONS)
        # In an empty starting position with no bombs, waiting is idle and should be suppressed
        self.assertNotEqual(action, 'WAIT')


    def test_plot_stats_file(self):
        """Verifies training progress curves plot generation and atomic save."""
        import tempfile
        from agent_code.my_spatial_dqn_agent.plot_training import plot_stats_file

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_stats.csv")
            png_path = os.path.join(tmpdir, "test_curves.png")

            # Non-existent or empty should return False
            self.assertFalse(plot_stats_file(csv_path, png_path))

            # Write valid sample stats data
            with open(csv_path, "w") as fh:
                fh.write(
                    "round,stage,epsilon,score,coins,kills,crates,bombs,survived,steps,invalid_acts,killed_self,got_killed,mean_reward,sum_reward,mean_td_error,unsafe_actions,wall_time_s\n"
                    "1,s1_coin_heaven,1.0,5.0,5,0,0,0,1,120,0,0,0,0.5,60.0,0.85,0,2.1\n"
                    "2,s1_coin_heaven,0.9,10.0,10,0,0,0,1,200,0,0,0,0.8,160.0,0.72,0,3.4\n"
                    "3,s2_classic_alone,0.8,8.0,3,0,5,2,0,150,1,1,0,-0.2,-30.0,0.61,1,2.8\n"
                )

            ok = plot_stats_file(csv_path, png_path, window=2)
            self.assertTrue(ok)
    def test_no_penalties_when_no_targets(self):
        """Verifies no inactivity/non-progress penalties and dynamic suicide scaling when board has no targets."""
        agent = MockAgent(train=True)
        setup(agent)
        setup_training(agent)

        empty_field = make_arena()  # No crates
        pos = (1, 1)
        old_f = np.zeros(N_FEATURES, dtype=np.float32)
        old_f[F_SAFE_WAIT] = 1.0

        # 1. Ten consecutive WAITs when nothing on board:
        for step in range(1, 11):
            s = make_state(empty_field, pos, step=step, coins=[], others=[])
            extra = custom_events(agent, s, 'WAIT', s, old_f, old_f, [e.WAITED])
            self.assertNotIn(STANDING_AROUND, extra)
            self.assertNotIn('IDLE_WAIT', extra)
            self.assertNotIn(NO_PROGRESS, extra)
            self.assertNotIn(STAGNATION, extra)
            self.assertEqual(agent._steps_without_progress, 0)
            self.assertEqual(agent._consecutive_stationary_steps, 0)

        # 2. Back and forth movement without targets does not trigger loop penalty
        s1 = make_state(empty_field, (1, 1), step=1, coins=[], others=[])
        s2 = make_state(empty_field, (2, 1), step=2, coins=[], others=[])
        agent.coordinate_history.append((1, 1))
        agent.coordinate_history.append((2, 1))
        extra_loop = custom_events(agent, s2, 'LEFT', s1, old_f, old_f, [e.MOVED_LEFT])
        self.assertNotIn('LOOP_DETECTED', extra_loop)

        # 3. Dynamic suicide penalty scaling when board is sparse
        # When alone on empty board (0 objects), suicide penalty escalates with 10 SPARSE_SUICIDE_PENALTY
        s_empty = make_state(empty_field, pos, step=1, coins=[], others=[])
        extra_suicide_sparse = custom_events(agent, s_empty, 'BOMB', s_empty, old_f, old_f, [e.KILLED_SELF])
        self.assertEqual(extra_suicide_sparse.count(SPARSE_SUICIDE_PENALTY), 10)
        total_r_sparse = reward_from_events([e.KILLED_SELF] + extra_suicide_sparse)
        self.assertEqual(total_r_sparse, -70.0)

        # When board is dense (e.g. 10+ crates), only base suicide penalty applies
        agent._steps_without_progress = 0
        dense_field = make_arena()
        for c in range(1, 11):
            dense_field[c, 1] = 1
        s_dense = make_state(dense_field, pos, step=1, coins=[], others=[])
        extra_suicide_dense = custom_events(agent, s_dense, 'BOMB', s_dense, old_f, old_f, [e.KILLED_SELF])
        self.assertEqual(extra_suicide_dense.count(SPARSE_SUICIDE_PENALTY), 0)
        self.assertNotIn(SPARSE_SUICIDE_PENALTY, extra_suicide_dense)

    def test_opponent_approach_and_no_progress_exemption(self):
        """Verifies moving toward opponent yields positive reward and exempts from NO_PROGRESS."""
        agent = MockAgent(train=True)
        setup(agent)
        setup_training(agent)

        empty_field = make_arena()
        pos = (1, 1)
        opp_pos = (5, 1)
        s_old = make_state(empty_field, pos, step=1, coins=[], others=[opp_pos])
        s_new = make_state(empty_field, (2, 1), step=2, coins=[], others=[opp_pos])

        old_f = state_to_features(s_old, deque())
        new_f = state_to_features(s_new, deque())

        extra = custom_events(agent, s_old, 'RIGHT', s_new, old_f, new_f, [e.MOVED_RIGHT])
        self.assertIn(MOVED_TOWARD_OPPONENT, extra)
        self.assertNotIn(NO_PROGRESS, extra)

    def test_is_better_round_criteria(self):
        """Verifies criteria: highest reward, tie-break lowest steps if survived, highest steps if dead."""
        # 1. Higher reward beats lower reward when both survived
        r_low = {'round': 1, 'score': 50, 'sum_reward': 210.0, 'survived': 1, 'steps': 300}
        r_high = {'round': 2, 'score': 50, 'sum_reward': 235.0, 'survived': 1, 'steps': 300}
        self.assertTrue(is_better_round(r_high, r_low))
        self.assertFalse(is_better_round(r_low, r_high))

        # 2. Tied reward: lowest step count wins (fastest completion)
        r_fast = {'round': 3, 'score': 50, 'sum_reward': 235.0, 'survived': 1, 'steps': 250}
        self.assertTrue(is_better_round(r_fast, r_high))
        self.assertFalse(is_better_round(r_high, r_fast))

        # 3. Survived round beats dead round with same score even if dead had higher reward
        r_dead_high_rew = {'round': 4, 'score': 50, 'sum_reward': 240.0, 'survived': 0, 'steps': 280}
        self.assertFalse(is_better_round(r_dead_high_rew, r_fast))
        self.assertTrue(is_better_round(r_fast, r_dead_high_rew))

        # 4. If no agents with highest score survived, take highest survival time (steps)
        r_dead_early = {'round': 5, 'score': 50, 'sum_reward': 150.0, 'survived': 0, 'steps': 60}
        r_dead_late = {'round': 6, 'score': 50, 'sum_reward': 150.0, 'survived': 0, 'steps': 220}
        self.assertTrue(is_better_round(r_dead_late, r_dead_early))
        self.assertFalse(is_better_round(r_dead_early, r_dead_late))

    def test_save_stage_best_and_last_models(self):
        """Verifies best and last models of each stage are saved in run folder."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ['MY_WM_RUN_DIR'] = tmpdir
            os.environ['MY_WM_STAGE'] = 's1_test'
            os.environ['MY_WM_N_ROUNDS'] = '2'

            agent = MockAgent(train=True)
            setup(agent)
            setup_training(agent)

            field = make_arena()
            s1 = make_state(field, (1, 1), step=1)
            # Round 1: score 10, survived 1
            agent.round_stats['coins'] = 10
            end_of_round(agent, s1, 'WAIT', [e.SURVIVED_ROUND])

            best_pt = os.path.join(tmpdir, "best_model_s1_test.pt")
            last_pt = os.path.join(tmpdir, "last_model_s1_test.pt")
            best_json = os.path.join(tmpdir, "best_model_s1_test.json")
            self.assertTrue(os.path.isfile(best_pt))
            self.assertTrue(os.path.isfile(last_pt))
            self.assertTrue(os.path.isfile(best_json))

            # Round 2: score 50 (better than round 1!), survived 1
            agent.round_stats['coins'] = 50
            end_of_round(agent, s1, 'WAIT', [e.SURVIVED_ROUND])

            import json
            with open(best_json) as fh:
                binfo = json.load(fh)
            self.assertEqual(binfo['score'], 50)
            self.assertEqual(binfo['round'], 2)


if __name__ == '__main__':
    unittest.main()
