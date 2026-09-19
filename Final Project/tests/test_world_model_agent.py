import pickle
import random
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import events as e
from agent_code.world_model_agent import callbacks
from agent_code.world_model_agent import train
from agent_code.world_model_agent.evaluate_world_model import chronological_replay
from agent_code.world_model_agent.q_network import NeuralQFunction, QExperience
from agent_code.world_model_agent.world_model import (
    BombOutcomePrediction,
    Experience,
    LearnedWorldModel,
    Prediction,
    REPLAY_BOMB,
    REPLAY_DANGER,
    REPLAY_NORMAL,
    REPLAY_TERMINAL,
    STATE_DOMAINS,
    decode_state,
    encode_state,
    game_state_to_world_state,
    world_state_to_game_state,
)


def make_field(size=7):
    field = np.full((size, size), -1, dtype=int)
    field[1:-1, 1:-1] = 0
    return field


def make_game_state(position, coins, bombs_left=True):
    field = make_field()
    return {
        "self": ("world_model_agent", 0, bombs_left, position),
        "coins": list(coins),
        "field": field,
        "bombs": [],
        "others": [],
        "explosion_map": np.zeros(field.shape, dtype=int),
    }


class WorldModelStateTests(unittest.TestCase):
    def test_state_contains_previous_tile_and_coin_distance(self):
        state = callbacks.state_to_key(
            make_game_state((2, 2), [(2, 4)]),
            previous_position=(2, 1),
        )

        self.assertEqual(42, len(state))
        self.assertEqual((1, 0, 0, 0), state[8:12])
        self.assertEqual(2, state[12])
        self.assertEqual(
            callbacks.ACTION_TO_INDEX["DOWN"],
            callbacks.direction_to_nearest_coin(
                ("world_model_agent", 0, True, (2, 2)),
                [(2, 4)],
                make_field(),
            ),
        )

    def test_equal_translated_local_views_share_a_state(self):
        first = callbacks.state_to_key(
            make_game_state((2, 2), [(2, 4)]),
            previous_position=(2, 1),
        )
        second = callbacks.state_to_key(
            make_game_state((3, 2), [(3, 4)]),
            previous_position=(3, 1),
        )

        self.assertEqual(first, second)

    def test_previous_tile_direction_distinguishes_context(self):
        game_state = make_game_state((2, 2), [(2, 4)])

        arrived_from_above = callbacks.state_to_key(
            game_state,
            previous_position=(2, 1),
        )
        arrived_from_left = callbacks.state_to_key(
            game_state,
            previous_position=(1, 2),
        )

        self.assertNotEqual(arrived_from_above, arrived_from_left)

    def test_equal_direction_at_different_distances_is_distinct(self):
        nearby = callbacks.state_to_key(
            make_game_state((2, 2), [(2, 3)])
        )
        farther = callbacks.state_to_key(
            make_game_state((2, 2), [(2, 5)])
        )

        self.assertNotEqual(nearby, farther)
        self.assertEqual(1, nearby[12])
        self.assertEqual(3, farther[12])

    def test_bomb_is_valid_only_when_available(self):
        with_bomb = make_game_state((2, 2), [], bombs_left=True)
        without_bomb = make_game_state((2, 2), [], bombs_left=False)

        self.assertIn("BOMB", callbacks.get_valid_actions(with_bomb))
        self.assertNotIn("BOMB", callbacks.get_valid_actions(without_bomb))

    def test_bomb_remains_valid_when_no_escape_route_exists(self):
        game_state = make_game_state((2, 2), [])
        game_state["field"][:] = -1
        game_state["field"][2, 2] = 0
        game_state["field"][3, 2] = 0
        game_state["field"][4, 2] = 0

        self.assertIn("BOMB", callbacks.get_valid_actions(game_state))
        self.assertFalse(
            callbacks.can_escape_after_placing_bomb(game_state)
        )


class DangerFeatureTests(unittest.TestCase):
    def test_danger_map_uses_timer_and_stops_at_stone_wall(self):
        game_state = make_game_state((1, 1), [])
        game_state["bombs"] = [((2, 2), 1)]
        game_state["field"][3, 2] = -1

        danger_map = callbacks.build_danger_map(game_state)

        self.assertEqual(2, danger_map[2, 2])
        self.assertEqual(2, danger_map[2, 5])
        self.assertEqual(callbacks.SAFE_TIME, danger_map[4, 2])

    def test_escape_search_turns_out_of_blast_line(self):
        game_state = make_game_state((3, 3), [])
        game_state["bombs"] = [((3, 3), 3)]

        escape_action = callbacks.find_escape_direction(game_state)

        self.assertEqual("UP", escape_action)

    def test_immediate_blast_marks_all_neighbor_actions_unsafe(self):
        game_state = make_game_state((3, 3), [])
        game_state["bombs"] = [((3, 3), 0)]

        safe_actions = callbacks.safe_action_mask(game_state)

        self.assertEqual([0, 0, 0, 0, 0], safe_actions)

    def test_danger_does_not_filter_mechanically_valid_actions(self):
        game_state = make_game_state((3, 3), [])
        game_state["bombs"] = [((3, 3), 3)]

        valid_actions = callbacks.get_valid_actions(game_state)
        self.assertEqual(
            set(callbacks.SURVIVAL_ACTIONS),
            set(valid_actions),
        )

    def test_danger_is_a_feature_but_does_not_block_action(self):
        game_state = make_game_state((3, 3), [])
        game_state["explosion_map"][2, 3] = 1

        valid_actions = callbacks.get_valid_actions(game_state)
        safe_actions = callbacks.safe_action_mask(game_state)

        self.assertIn("LEFT", valid_actions)
        self.assertIn("RIGHT", valid_actions)
        self.assertEqual(
            0,
            safe_actions[callbacks.ACTION_TO_INDEX["LEFT"]],
        )

    def test_state_contains_danger_escape_and_bomb_features(self):
        game_state = make_game_state((3, 3), [])
        game_state["bombs"] = [((3, 3), 3)]
        game_state["field"][4, 3] = 1

        state = callbacks.state_to_key(game_state)

        self.assertEqual(3, state[13])
        self.assertEqual((1, 0, 0, 0, 0), state[19:24])
        self.assertEqual(1, state[24])
        self.assertEqual(1, state[25])

    def test_hazard_states_ignore_coin_and_arrival_context(self):
        first = make_game_state((3, 3), [(1, 3)])
        first["bombs"] = [((3, 3), 3)]
        second = make_game_state((3, 3), [(5, 3)])
        second["bombs"] = [((3, 3), 3)]

        first_state = callbacks.state_to_key(
            first,
            previous_position=(3, 2),
        )
        second_state = callbacks.state_to_key(
            second,
            previous_position=(2, 3),
        )

        self.assertEqual(first_state, second_state)
        self.assertEqual((0, 0, 0, 0), first_state[:4])
        self.assertEqual((0, 0, 0, 0), first_state[8:12])
        self.assertEqual(0, first_state[12])


class WorldModelAgentTests(unittest.TestCase):
    def test_opponent_model_extracts_movement_and_bomb_labels(self):
        current = make_game_state((2, 2), [])
        current["others"] = [("rule", 0, True, (4, 2))]
        following = make_game_state((2, 2), [])
        following["others"] = [("rule", 0, True, (5, 2))]
        state = callbacks.state_to_key(current)
        experience = Experience(
            state=state,
            world_state=game_state_to_world_state(current),
            action_index=callbacks.ACTION_TO_INDEX["WAIT"],
            reward=0.0,
            next_state=callbacks.state_to_key(following),
            next_world_state=game_state_to_world_state(following),
            done=False,
            valid_actions=(callbacks.ACTION_TO_INDEX["WAIT"],),
            next_valid_actions=(callbacks.ACTION_TO_INDEX["WAIT"],),
        )
        model = LearnedWorldModel(min_samples=1, hidden_size=8)

        _inputs, targets = model._opponent_training_samples([experience])

        self.assertEqual([callbacks.ACTION_TO_INDEX["RIGHT"]], targets)

        following["others"] = [("rule", 0, False, (4, 2))]
        following["bombs"] = [((4, 2), 3)]
        bomb_experience = replace(
            experience,
            next_world_state=game_state_to_world_state(following),
        )
        _inputs, targets = model._opponent_training_samples(
            [bomb_experience]
        )
        self.assertEqual([callbacks.ACTION_TO_INDEX["BOMB"]], targets)

    def test_factored_opponent_prediction_corrects_world_position(self):
        current = make_game_state((2, 2), [])
        current["others"] = [("rule", 0, True, (4, 2))]
        model = LearnedWorldModel(min_samples=1, hidden_size=8)

        class AlwaysRight:
            classes_ = np.arange(7)

            @staticmethod
            def predict(_inputs):
                return np.asarray([callbacks.ACTION_TO_INDEX["RIGHT"]])

        model.opponent_model = AlwaysRight()
        corrected = model._correct_predicted_opponents(
            game_state_to_world_state(current),
            callbacks.ACTION_TO_INDEX["WAIT"],
            game_state_to_world_state(current),
        )
        corrected_game = world_state_to_game_state(corrected)

        self.assertEqual([(5, 2)], [other[3] for other in corrected_game["others"]])

    def test_opponent_features_encode_direction_distance_and_adjacency(self):
        game_state = make_game_state((2, 2), [])
        game_state["others"] = [
            ("opponent", 0, True, (2, 4)),
        ]

        state = callbacks.state_to_key(game_state)

        self.assertEqual(42, len(state))
        self.assertEqual((0, 0, 1, 0), state[26:30])
        self.assertEqual(2, state[30])
        self.assertEqual((0, 0, 0, 0), state[31:35])
        self.assertEqual((0, 0, 1, 0), state[35:39])
        self.assertEqual(4, state[39])
        self.assertEqual(0, state[40])
        self.assertGreater(state[41], 0)

    def test_tactical_features_detect_a_trapped_opponent(self):
        game_state = make_game_state((2, 2), [])
        game_state["field"][:] = -1
        for position in ((2, 2), (2, 3), (3, 3)):
            game_state["field"][position] = 0
        game_state["others"] = [
            ("opponent", 0, True, (2, 3)),
        ]

        state = callbacks.state_to_key(game_state)

        self.assertEqual((0, 0, 1, 0), state[35:39])
        self.assertEqual(1, state[39])
        self.assertEqual(1, state[40])
        self.assertEqual(0, state[41])

    def test_bomb_outcome_collects_delayed_events_until_explosion(self):
        model = LearnedWorldModel(min_samples=1, hidden_size=8)
        game_state = make_game_state((2, 2), [])
        game_state["others"] = [("opponent", 0, True, (2, 4))]
        state = callbacks.state_to_key(game_state)

        model.observe_bomb_events(state, bomb_dropped=True)
        model.observe_bomb_events(state, crate_destroyed=True)
        model.observe_bomb_events(
            state,
            killed_opponent=True,
            exploded=True,
        )

        self.assertEqual(1, len(model.bomb_outcome_replay))
        outcome = model.bomb_outcome_replay[0]
        self.assertTrue(outcome.survived)
        self.assertTrue(outcome.destroyed_crate)
        self.assertTrue(outcome.killed_opponent)
        self.assertFalse(outcome.opponent_escaped)
        self.assertEqual(1, len(model.bomb_kill_replay))
        self.assertEqual(0, len(model.bomb_failure_replay))

    def test_bomb_outcome_marks_terminal_death_and_escape(self):
        model = LearnedWorldModel(min_samples=1, hidden_size=8)
        game_state = make_game_state((2, 2), [])
        game_state["others"] = [("opponent", 0, True, (2, 4))]
        state = callbacks.state_to_key(game_state)

        model.observe_bomb_events(state, bomb_dropped=True)
        model.observe_bomb_events(state, death=True, done=True)

        outcome = model.bomb_outcome_replay[0]
        self.assertFalse(outcome.survived)
        self.assertFalse(outcome.killed_opponent)
        self.assertTrue(outcome.opponent_escaped)
        self.assertEqual(1, len(model.bomb_failure_replay))

    def test_rare_bomb_outcome_survives_main_replay_rollover(self):
        model = LearnedWorldModel(
            min_samples=1,
            hidden_size=8,
            bomb_outcome_capacity=1,
        )
        state = callbacks.state_to_key(make_game_state((2, 2), []))
        model.observe_bomb_events(state, bomb_dropped=True)
        model.observe_bomb_events(state, killed_opponent=True, exploded=True)
        model.observe_bomb_events(state, bomb_dropped=True)
        model.observe_bomb_events(state, exploded=True)

        self.assertFalse(model.bomb_outcome_replay[0].killed_opponent)
        self.assertTrue(model.bomb_kill_replay[0].killed_opponent)

    def test_escape_prediction_is_zero_without_threatened_opponent(self):
        model = LearnedWorldModel(min_samples=1, hidden_size=8)
        state = callbacks.state_to_key(make_game_state((2, 2), []))

        prediction = model.predict_bomb_outcome(state)

        self.assertEqual(0.0, prediction.escape_probability)

    def test_suicide_labels_cover_current_action_and_four_future_steps(self):
        model = LearnedWorldModel(
            replay_capacity=20,
            min_samples=1,
            hidden_size=8,
            suicide_risk_horizon=5,
        )
        game_state = make_game_state((2, 2), [])
        state = callbacks.state_to_key(game_state)
        world_state = game_state_to_world_state(game_state)

        for step in range(6):
            terminal = step == 5
            model.observe(
                state=state,
                world_state=world_state,
                action_index=callbacks.ACTION_TO_INDEX["WAIT"],
                reward=-10.0 if terminal else 0.0,
                next_state=None if terminal else state,
                next_world_state=None if terminal else world_state,
                done=terminal,
                valid_actions=(callbacks.ACTION_TO_INDEX["WAIT"],),
                next_valid_actions=(
                    ()
                    if terminal
                    else (callbacks.ACTION_TO_INDEX["WAIT"],)
                ),
                replay_category=(
                    REPLAY_TERMINAL if terminal else REPLAY_NORMAL
                ),
                death=terminal,
                suicide=terminal,
            )

        self.assertEqual(
            [False, True, True, True, True, True],
            [item.suicide_within_horizon for item in model.replay],
        )
        self.assertEqual(
            [False, True, True, True, True, True],
            [item.death_within_horizon for item in model.replay],
        )
        self.assertEqual(5, len(model.death_risk_replay))
        self.assertEqual(5, len(model.suicide_risk_replay))

    def test_opponent_death_is_not_stored_as_suicide(self):
        model = LearnedWorldModel(min_samples=1, hidden_size=8)
        game_state = make_game_state((2, 2), [])
        state = callbacks.state_to_key(game_state)
        world_state = game_state_to_world_state(game_state)

        model.observe(
            state=state,
            world_state=world_state,
            action_index=callbacks.ACTION_TO_INDEX["WAIT"],
            reward=-5.0,
            next_state=None,
            next_world_state=None,
            done=True,
            valid_actions=(callbacks.ACTION_TO_INDEX["WAIT"],),
            next_valid_actions=(),
            replay_category=REPLAY_TERMINAL,
            death=True,
            suicide=False,
        )

        self.assertTrue(model.replay[0].death_within_horizon)
        self.assertFalse(model.replay[0].suicide_within_horizon)
        self.assertEqual(1, len(model.death_risk_replay))
        self.assertEqual(0, len(model.suicide_risk_replay))

    def test_balanced_replay_batch_contains_rare_categories(self):
        model = LearnedWorldModel(
            replay_capacity=20,
            min_samples=1,
            hidden_size=8,
        )
        game_state = make_game_state((2, 2), [])
        state = callbacks.state_to_key(game_state)
        world_state = game_state_to_world_state(game_state)

        for category in (
            REPLAY_NORMAL,
            REPLAY_BOMB,
            REPLAY_DANGER,
            REPLAY_TERMINAL,
        ):
            model.observe(
                state=state,
                world_state=world_state,
                action_index=callbacks.ACTION_TO_INDEX["WAIT"],
                reward=0.0,
                next_state=state,
                next_world_state=world_state,
                done=False,
                valid_actions=(callbacks.ACTION_TO_INDEX["WAIT"],),
                next_valid_actions=(callbacks.ACTION_TO_INDEX["WAIT"],),
                replay_category=category,
            )

        batch = model._balanced_batch(20)

        self.assertEqual(20, len(batch))
        self.assertEqual(
            {
                REPLAY_NORMAL: 8,
                REPLAY_BOMB: 4,
                REPLAY_DANGER: 5,
                REPLAY_TERMINAL: 3,
            },
            model.last_batch_category_counts,
        )

    def test_action_selection_subtracts_learned_suicide_risk(self):
        game_state = make_game_state((2, 2), [])
        state = callbacks.state_to_key(game_state)
        q_values = np.zeros(len(callbacks.ACTIONS), dtype=np.float32)
        q_values[callbacks.ACTION_TO_INDEX["RIGHT"]] = 10.0
        q_values[callbacks.ACTION_TO_INDEX["DOWN"]] = 1.0

        class FakeRiskModel:
            is_death_model_ready = True

            @staticmethod
            def predict_death_probability(_state, action_index):
                return float(
                    action_index == callbacks.ACTION_TO_INDEX["RIGHT"]
                )

        class FakeQNetwork:
            @staticmethod
            def predict(_state):
                return q_values

        agent = SimpleNamespace(
            train=False,
            action_rng=random.Random(0),
            q_network=FakeQNetwork(),
            world_model=FakeRiskModel(),
            previous_position=None,
            current_state_key=None,
            logger=Mock(),
        )

        self.assertEqual("DOWN", callbacks.act(agent, game_state))

    def test_action_selection_also_uses_specialized_suicide_risk(self):
        game_state = make_game_state((2, 2), [])
        q_values = np.zeros(len(callbacks.ACTIONS), dtype=np.float32)
        q_values[callbacks.ACTION_TO_INDEX["RIGHT"]] = 10.0
        q_values[callbacks.ACTION_TO_INDEX["DOWN"]] = 1.0

        class FakeRiskModel:
            is_death_model_ready = False
            is_suicide_model_ready = True

            @staticmethod
            def predict_suicide_probability(_state, action_index):
                return float(
                    action_index == callbacks.ACTION_TO_INDEX["RIGHT"]
                )

        class FakeQNetwork:
            @staticmethod
            def predict(_state):
                return q_values

        agent = SimpleNamespace(
            train=False,
            action_rng=random.Random(0),
            q_network=FakeQNetwork(),
            world_model=FakeRiskModel(),
            previous_position=None,
            current_state_key=None,
            logger=Mock(),
        )

        self.assertEqual("DOWN", callbacks.act(agent, game_state))

    def test_action_selection_uses_learned_bomb_outcome(self):
        game_state = make_game_state((2, 2), [])
        q_values = np.zeros(len(callbacks.ACTIONS), dtype=np.float32)
        q_values[callbacks.ACTION_TO_INDEX["WAIT"]] = 1.0

        class FakeWorldModel:
            is_death_model_ready = False
            is_bomb_outcome_ready = True

            @staticmethod
            def predict_bomb_outcome(_state):
                return BombOutcomePrediction(
                    survived_probability=1.0,
                    crate_probability=0.0,
                    kill_probability=1.0,
                    escape_probability=0.0,
                )

        class FakeQNetwork:
            @staticmethod
            def predict(_state):
                return q_values

        agent = SimpleNamespace(
            train=False,
            action_rng=random.Random(0),
            q_network=FakeQNetwork(),
            world_model=FakeWorldModel(),
            previous_position=None,
            current_state_key=None,
            logger=Mock(),
        )

        self.assertEqual("BOMB", callbacks.act(agent, game_state))

    def test_low_kill_probability_does_not_create_bomb_bonus(self):
        class FakeWorldModel:
            is_bomb_outcome_ready = True

            @staticmethod
            def predict_bomb_outcome(_state):
                return BombOutcomePrediction(
                    survived_probability=1.0,
                    crate_probability=0.0,
                    kill_probability=0.1,
                    escape_probability=0.0,
                )

        adjustment, _prediction = callbacks.bomb_outcome_adjustment(
            FakeWorldModel(),
            (0,) * 42,
        )

        self.assertEqual(0.0, adjustment)

    def test_bomb_outcome_survival_probability_limits_bomb_choice(self):
        game_state = make_game_state((2, 2), [])
        q_values = np.zeros(len(callbacks.ACTIONS), dtype=np.float32)
        q_values[callbacks.ACTION_TO_INDEX["BOMB"]] = 10.0
        q_values[callbacks.ACTION_TO_INDEX["WAIT"]] = 1.0

        class FakeWorldModel:
            is_death_model_ready = False
            is_bomb_outcome_ready = True

            @staticmethod
            def predict_bomb_outcome(_state):
                return BombOutcomePrediction(
                    survived_probability=0.0,
                    crate_probability=0.0,
                    kill_probability=0.0,
                    escape_probability=0.0,
                )

        class FakeQNetwork:
            @staticmethod
            def predict(_state):
                return q_values

        agent = SimpleNamespace(
            train=False,
            action_rng=random.Random(0),
            q_network=FakeQNetwork(),
            world_model=FakeWorldModel(),
            previous_position=None,
            current_state_key=None,
            logger=Mock(),
        )

        self.assertEqual("WAIT", callbacks.act(agent, game_state))

    def test_incompatible_saved_model_is_ignored(self):
        old_model = {
            "q_table": {(1,) * 8: np.ones(len(callbacks.ACTIONS))},
            "world_model": {("old", "UP"): []},
            "epsilon": 0.02,
        }

        with tempfile.NamedTemporaryFile() as model_file:
            pickle.dump(old_model, model_file)
            model_file.flush()

            agent = SimpleNamespace(logger=Mock())

            with patch.object(callbacks, "MODEL_FILE", model_file.name):
                callbacks.setup(agent)

        self.assertIsInstance(agent.q_network, NeuralQFunction)
        self.assertIsNone(agent.world_model)
        self.assertEqual(0.2, agent.epsilon)
        agent.logger.warning.assert_called_once()

    def test_terminal_transition_is_stored_and_planned(self):
        agent = SimpleNamespace(
            gamma=0.95,
            planning_rollouts=0,
            planning_horizon=3,
            world_model_done_threshold=0.5,
            world_model_train_every=1,
            planning_rng=random.Random(1),
            q_network=NeuralQFunction(
                min_samples=1,
                batch_size=1,
                hidden_size=8,
                learning_rate=0.01,
                target_update_interval=10,
            ),
            world_model=LearnedWorldModel(
                min_samples=1,
                batch_size=1,
                hidden_size=8,
            ),
            epsilon=0.2,
            logger=Mock(),
        )
        game_state = make_game_state((2, 2), [(2, 3)])
        state = callbacks.state_to_key(game_state)

        with patch.object(train, "save_model"):
            train.end_of_round(
                agent,
                game_state,
                "DOWN",
                [e.COIN_COLLECTED],
            )

        self.assertEqual(1, len(agent.world_model))
        experience = agent.world_model.replay[0]
        self.assertEqual(5.0, experience.reward)
        self.assertIsNone(experience.next_state)
        self.assertTrue(experience.done)
        self.assertEqual((), experience.next_valid_actions)
        self.assertEqual(1, len(agent.q_network))
        self.assertEqual(1, agent.q_network.training_steps)
        self.assertGreater(
            agent.q_network.predict(state)[
                callbacks.ACTION_TO_INDEX["DOWN"]
            ],
            0.0,
        )

    def test_learned_model_projects_predictions_to_valid_states(self):
        model = LearnedWorldModel(
            min_samples=4,
            batch_size=4,
            hidden_size=8,
            seed=3,
        )
        game_state = make_game_state((2, 2), [(2, 3)])
        next_game_state = make_game_state((2, 3), [])
        state = callbacks.state_to_key(game_state)
        next_state = callbacks.state_to_key(next_game_state)
        world_state = game_state_to_world_state(game_state)
        next_world_state = game_state_to_world_state(next_game_state)

        for _ in range(8):
            model.observe(
                state=state,
                world_state=world_state,
                action_index=callbacks.ACTION_TO_INDEX["DOWN"],
                reward=5.0,
                next_state=next_state,
                next_world_state=next_world_state,
                done=False,
                valid_actions=(1, 2, 4, 5),
                next_valid_actions=(0, 1, 3, 4, 5),
            )
        for _ in range(5):
            model.train_step()

        prediction = model.predict(
            world_state,
            state,
            callbacks.ACTION_TO_INDEX["DOWN"],
        )

        self.assertEqual(state, decode_state(encode_state(state)))
        predicted_game_state = world_state_to_game_state(
            prediction.next_world_state
        )
        self.assertEqual(42, len(prediction.next_state))
        self.assertTrue(
            all(
                value in domain
                for value, domain in zip(
                    prediction.next_state,
                    STATE_DOMAINS,
                )
            )
        )
        self.assertEqual(game_state["field"].shape, predicted_game_state["field"].shape)
        self.assertEqual(0, predicted_game_state["field"][predicted_game_state["self"][3]])
        self.assertIn(
            callbacks.ACTION_TO_INDEX["WAIT"],
            prediction.next_valid_actions,
        )

    def test_complete_world_state_round_trip(self):
        game_state = make_game_state((3, 3), [(1, 1), (5, 5)])
        game_state["round"] = 4
        game_state["step"] = 27
        game_state["bombs"] = [((3, 3), 2), ((1, 3), 0)]
        game_state["others"] = [
            ("opponent", 0, False, (5, 3)),
        ]
        game_state["explosion_map"][3, 2] = 1

        decoded = world_state_to_game_state(
            game_state_to_world_state(game_state)
        )

        self.assertTrue(np.array_equal(game_state["field"], decoded["field"]))
        self.assertEqual(set(game_state["coins"]), set(decoded["coins"]))
        self.assertEqual(set(game_state["bombs"]), set(decoded["bombs"]))
        self.assertEqual(game_state["self"][2:], decoded["self"][2:])
        self.assertEqual(game_state["others"][0][2:], decoded["others"][0][2:])
        self.assertTrue(
            np.array_equal(
                game_state["explosion_map"],
                decoded["explosion_map"],
            )
        )

    def test_planning_chains_generated_states_and_updates_backwards(self):
        game_0 = make_game_state((2, 2), [(2, 4)])
        game_1 = make_game_state((3, 2), [(2, 4)])
        game_2 = make_game_state((4, 2), [(2, 4)])
        state_0 = callbacks.state_to_key(game_0)
        state_1 = callbacks.state_to_key(
            game_1,
            previous_position=game_0["self"][3],
        )
        state_2 = callbacks.state_to_key(
            game_2,
            previous_position=game_1["self"][3],
        )
        world_0 = game_state_to_world_state(game_0)
        world_1 = game_state_to_world_state(game_1)
        world_2 = game_state_to_world_state(game_2)
        right_index = callbacks.ACTION_TO_INDEX["RIGHT"]

        class FakeWorldModel:
            is_ready = True

            def __init__(self):
                self.predicted_from = []

            def sample_start(self, _rng):
                return Experience(
                    state=state_0,
                    world_state=world_0,
                    action_index=right_index,
                    reward=0.0,
                    next_state=state_1,
                    next_world_state=world_1,
                    done=False,
                    valid_actions=(right_index,),
                    next_valid_actions=(right_index,),
                )

            def predict(self, _world_state, state, _action_index):
                self.predicted_from.append(state)
                if state == state_0:
                    return Prediction(
                        reward=0.0,
                        next_world_state=world_1,
                        next_state=state_1,
                        done_probability=0.0,
                        next_valid_actions=(right_index,),
                    )
                return Prediction(
                    reward=-10.0,
                    next_world_state=world_2,
                    next_state=state_2,
                    done_probability=1.0,
                    next_valid_actions=(),
                )

        agent = SimpleNamespace(
            gamma=0.95,
            planning_rollouts=1,
            planning_horizon=2,
            world_model_done_threshold=0.5,
            planning_rng=random.Random(1),
            q_network=NeuralQFunction(
                min_samples=1,
                batch_size=2,
                hidden_size=8,
                learning_rate=0.01,
                target_update_interval=10,
            ),
            world_model=FakeWorldModel(),
        )

        errors = train.perform_planning_updates(agent)

        self.assertEqual([state_0, state_1], agent.world_model.predicted_from)
        self.assertEqual(2, len(errors))
        self.assertEqual(1, agent.q_network.training_steps)
        self.assertEqual(0, len(agent.q_network))

    def test_double_dqn_uses_online_selection_and_target_evaluation(self):
        state = callbacks.state_to_key(make_game_state((2, 2), []))
        network = NeuralQFunction(
            min_samples=1,
            batch_size=1,
            hidden_size=8,
            gamma=0.5,
            n_step=1,
        )
        experience = QExperience(
            state=state,
            action_index=callbacks.ACTION_TO_INDEX["UP"],
            reward=1.0,
            next_state=state,
            done=False,
            next_valid_actions=(
                callbacks.ACTION_TO_INDEX["UP"],
                callbacks.ACTION_TO_INDEX["RIGHT"],
            ),
        )
        current = np.zeros((1, len(callbacks.ACTIONS)))
        online_next = np.asarray([[2.0, 5.0, 0.0, 0.0, 0.0, 0.0]])
        target_next = np.asarray([[9.0, 3.0, 0.0, 0.0, 0.0, 0.0]])

        with patch.object(
            network.online_model,
            "predict",
            side_effect=[current, online_next],
        ), patch.object(
            network.target_model,
            "predict",
            return_value=target_next,
        ):
            _states, targets, _errors = network._targets([experience])

        self.assertEqual(2.5, targets[0, callbacks.ACTION_TO_INDEX["UP"]])

    def test_five_step_return_assigns_delayed_death_to_bomb(self):
        state = callbacks.state_to_key(make_game_state((2, 2), []))
        network = NeuralQFunction(
            min_samples=100,
            hidden_size=8,
            gamma=0.5,
            n_step=5,
        )
        for step in range(5):
            network.observe(
                state=state,
                action_index=(
                    callbacks.ACTION_TO_INDEX["BOMB"]
                    if step == 0
                    else callbacks.ACTION_TO_INDEX["WAIT"]
                ),
                reward=-10.0 if step == 4 else 0.0,
                next_state=None if step == 4 else state,
                done=step == 4,
                next_valid_actions=(
                    ()
                    if step == 4
                    else (callbacks.ACTION_TO_INDEX["WAIT"],)
                ),
            )

        bomb_experience = network.replay[0]
        self.assertEqual(callbacks.ACTION_TO_INDEX["BOMB"], bomb_experience.action_index)
        self.assertAlmostEqual(-0.625, bomb_experience.reward)
        self.assertTrue(bomb_experience.done)
        self.assertAlmostEqual(0.5 ** 5, bomb_experience.bootstrap_discount)

    def test_freeze_q_network_skips_real_and_imagined_updates(self):
        state = callbacks.state_to_key(make_game_state((2, 2), []))
        agent = SimpleNamespace(
            q_network=Mock(),
            world_model=SimpleNamespace(is_ready=True),
        )

        with patch.object(train, "FREEZE_Q_NETWORK", True):
            error = train.update_q(
                agent,
                state,
                "WAIT",
                0.0,
                state,
                False,
                next_valid_actions=("WAIT",),
            )
            imagined = train.perform_planning_updates(agent)

        self.assertEqual(0.0, error)
        self.assertEqual([], imagined)
        agent.q_network.observe.assert_not_called()

    def test_q_replay_batch_keeps_rare_bomb_actions_visible(self):
        state = callbacks.state_to_key(make_game_state((2, 2), []))
        network = NeuralQFunction(
            min_samples=100,
            batch_size=20,
            hidden_size=8,
            n_step=1,
        )
        for index in range(100):
            network.observe(
                state=state,
                action_index=(
                    callbacks.ACTION_TO_INDEX["BOMB"]
                    if index == 0
                    else callbacks.ACTION_TO_INDEX["WAIT"]
                ),
                reward=0.0,
                next_state=state,
                done=False,
                next_valid_actions=(callbacks.ACTION_TO_INDEX["WAIT"],),
            )

        batch = network._sample_replay_batch(20)

        self.assertGreaterEqual(
            sum(item.action_index == callbacks.ACTION_TO_INDEX["BOMB"] for item in batch),
            5,
        )

    def test_q_replay_retains_and_samples_rare_reward_tails(self):
        state = callbacks.state_to_key(make_game_state((2, 2), []))
        network = NeuralQFunction(
            replay_capacity=20,
            min_samples=100,
            batch_size=16,
            hidden_size=8,
            n_step=1,
        )
        for reward in (20.0, -25.0):
            network.observe(
                state=state,
                action_index=callbacks.ACTION_TO_INDEX["WAIT"],
                reward=reward,
                next_state=state,
                done=False,
                next_valid_actions=(callbacks.ACTION_TO_INDEX["WAIT"],),
            )
        for _index in range(30):
            network.observe(
                state=state,
                action_index=callbacks.ACTION_TO_INDEX["WAIT"],
                reward=0.0,
                next_state=state,
                done=False,
                next_valid_actions=(callbacks.ACTION_TO_INDEX["WAIT"],),
            )

        self.assertNotIn(20.0, [item.reward for item in network.replay])
        self.assertNotIn(-25.0, [item.reward for item in network.replay])
        batch = network._sample_replay_batch(16)
        self.assertIn(20.0, [item.reward for item in batch])
        self.assertIn(-25.0, [item.reward for item in batch])

    def test_evaluation_reads_wrapped_replay_in_chronological_order(self):
        model = LearnedWorldModel(
            replay_capacity=3,
            min_samples=1,
            hidden_size=8,
        )
        state = callbacks.state_to_key(
            make_game_state((2, 2), [])
        )
        world_state = game_state_to_world_state(
            make_game_state((2, 2), [])
        )

        for reward in range(5):
            model.observe(
                state=state,
                world_state=world_state,
                action_index=callbacks.ACTION_TO_INDEX["WAIT"],
                reward=float(reward),
                next_state=state,
                next_world_state=world_state,
                done=False,
                valid_actions=(callbacks.ACTION_TO_INDEX["WAIT"],),
                next_valid_actions=(callbacks.ACTION_TO_INDEX["WAIT"],),
            )

        self.assertEqual(
            [2.0, 3.0, 4.0],
            [item.reward for item in chronological_replay(model)],
        )


class BombRewardTests(unittest.TestCase):
    def test_bomb_next_to_crate_is_rewarded_as_useful(self):
        game_state = make_game_state((2, 2), [])
        game_state["field"][3, 2] = 1
        events = [e.BOMB_DROPPED]

        train.add_custom_events(game_state, "BOMB", events)

        self.assertIn(train.USEFUL_BOMB, events)
        self.assertNotIn(train.USELESS_BOMB, events)

    def test_wall_blocks_useful_bomb_target(self):
        game_state = make_game_state((2, 2), [])
        game_state["field"][3, 2] = -1
        game_state["field"][4, 2] = 1
        events = [e.BOMB_DROPPED]

        train.add_custom_events(game_state, "BOMB", events)

        self.assertIn(train.USELESS_BOMB, events)
        self.assertNotIn(train.USEFUL_BOMB, events)

    def test_rejected_bomb_is_not_classified(self):
        game_state = make_game_state((2, 2), [], bombs_left=False)
        game_state["field"][3, 2] = 1
        events = [e.INVALID_ACTION]

        train.add_custom_events(game_state, "BOMB", events)

        self.assertNotIn(train.USEFUL_BOMB, events)
        self.assertNotIn(train.USELESS_BOMB, events)

    def test_bomb_rewards_penalize_suicide(self):
        agent = SimpleNamespace(logger=Mock())

        useful_reward = train.reward_from_events(
            agent,
            [e.BOMB_DROPPED, train.USEFUL_BOMB],
        )
        crate_reward = train.reward_from_events(
            agent,
            [e.CRATE_DESTROYED, e.COIN_FOUND],
        )
        suicide_reward = train.reward_from_events(
            agent,
            [e.KILLED_SELF, e.GOT_KILLED],
        )

        self.assertEqual(0.25, useful_reward)
        self.assertEqual(1.5, crate_reward)
        self.assertEqual(-40.0, suicide_reward)

    def test_danger_transition_events(self):
        old_state = make_game_state((2, 3), [])
        old_state["bombs"] = [((2, 2), 3)]
        escaped_state = make_game_state((3, 3), [])
        escaped_state["bombs"] = [((2, 2), 2)]
        escaped_events = [e.MOVED_RIGHT]

        train.add_custom_events(
            old_state,
            "RIGHT",
            escaped_events,
            escaped_state,
        )

        self.assertIn(train.MOVED_OUT_OF_DANGER, escaped_events)

        safe_state = make_game_state((3, 3), [])
        safe_state["bombs"] = [((2, 2), 3)]
        entered_state = make_game_state((2, 3), [])
        entered_state["bombs"] = [((2, 2), 2)]
        entered_events = [e.MOVED_LEFT]

        train.add_custom_events(
            safe_state,
            "LEFT",
            entered_events,
            entered_state,
        )

        self.assertIn(train.MOVED_INTO_DANGER, entered_events)

        immediate_state = make_game_state((2, 3), [])
        immediate_state["bombs"] = [((2, 2), 0)]
        waited_state = make_game_state((2, 3), [])
        waited_state["bombs"] = [((2, 2), 0)]
        waited_events = [e.WAITED]

        train.add_custom_events(
            immediate_state,
            "WAIT",
            waited_events,
            waited_state,
        )

        self.assertIn(train.STAYED_IN_DANGER, waited_events)

        center_state = make_game_state((3, 3), [])
        center_state["bombs"] = [((3, 3), 3)]
        escaping_state = make_game_state((3, 2), [])
        escaping_state["bombs"] = [((3, 3), 2)]
        escaping_events = [e.MOVED_UP]

        train.add_custom_events(
            center_state,
            "UP",
            escaping_events,
            escaping_state,
        )

        self.assertIn(train.MOVED_TOWARD_SAFETY, escaping_events)

    def test_unsafe_actions_receive_immediate_reward_penalties(self):
        no_escape_state = make_game_state((2, 2), [])
        no_escape_state["field"][:] = -1
        no_escape_state["field"][2, 2] = 0
        no_escape_state["field"][3, 2] = 1
        bomb_events = [e.BOMB_DROPPED]

        train.add_custom_events(
            no_escape_state,
            "BOMB",
            bomb_events,
        )

        self.assertIn(train.USEFUL_BOMB, bomb_events)
        self.assertIn(train.BOMB_WITHOUT_ESCAPE, bomb_events)

        explosion_state = make_game_state((3, 3), [])
        explosion_state["explosion_map"][2, 3] = 1
        death_events = [e.MOVED_LEFT, e.KILLED_SELF, e.GOT_KILLED]

        train.add_custom_events(
            explosion_state,
            "LEFT",
            death_events,
        )

        self.assertIn(
            train.MOVED_INTO_IMMEDIATE_DANGER,
            death_events,
        )
        agent = SimpleNamespace(logger=Mock())
        self.assertEqual(
            -43.0,
            train.reward_from_events(agent, death_events),
        )


if __name__ == "__main__":
    unittest.main()
