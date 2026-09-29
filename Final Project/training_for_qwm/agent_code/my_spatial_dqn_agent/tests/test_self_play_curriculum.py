import os
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from agent_code.my_spatial_dqn_agent.callbacks import setup as spatial_setup


def compute_stage_slices(n_rounds: int, interval: int):
    """Slices a stage into intervals for intra-stage self-play."""
    if interval <= 0 or n_rounds <= interval:
        return [n_rounds]
    slices = []
    rem = n_rounds
    while rem > interval:
        slices.append(interval)
        rem -= interval
    if rem > 0:
        slices.append(rem)
    return slices


def test_stage_slicing():
    assert compute_stage_slices(600, 200) == [200, 200, 200]
    assert compute_stage_slices(300, 200) == [200, 100]
    assert compute_stage_slices(150, 200) == [150]
    assert compute_stage_slices(200, 200) == [200]
    assert compute_stage_slices(600, 0) == [600]
    assert compute_stage_slices(600, -1) == [600]


def test_scenario_selection():
    scenarios = ["classic", "loot-crate", "coin-heaven"]
    chosen = [random.choice(scenarios) for _ in range(30)]
    assert all(c in scenarios for c in chosen)
    assert len(set(chosen)) > 1


def test_callbacks_checkpoint_resolution():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        dummy_model = run_dir / "my-saved-model-spatial-dqn.pt"
        dummy_model.write_text("checkpoint_data")

        # Test clone (train=False) loads from run_dir when available
        agent_clone = SimpleNamespace(train=False)
        with mock.patch.dict(os.environ, {"MY_WM_RUN_DIR": str(run_dir), "MY_WM_MODEL_FILE": dummy_model.name}):
            with mock.patch("agent_code.my_spatial_dqn_agent.callbacks.load_model") as mock_load:
                mock_load.return_value = SimpleNamespace(device="cpu", set_lr=lambda lr: None)
                spatial_setup(agent_clone)
                mock_load.assert_called_once()
                loaded_file = mock_load.call_args[0][0]
                assert Path(loaded_file).resolve() == dummy_model.resolve()

        # Test learner (train=True) also loads from run_dir
        agent_train = SimpleNamespace(train=True)
        with mock.patch.dict(os.environ, {"MY_WM_RUN_DIR": str(run_dir), "MY_WM_MODEL_FILE": dummy_model.name}):
            with mock.patch("agent_code.my_spatial_dqn_agent.callbacks.load_model") as mock_load:
                mock_load.return_value = SimpleNamespace(device="cpu", set_lr=lambda lr: None)
                spatial_setup(agent_train)
                mock_load.assert_called_once()
                loaded_file = mock_load.call_args[0][0]
                assert Path(loaded_file).resolve() == dummy_model.resolve()
                assert Path(agent_train.model_file).resolve() == dummy_model.resolve()


def test_spatial_dqn_arg_parser():
    from agent_code.my_spatial_dqn_agent.train_curriculum import build_arg_parser
    parser = build_arg_parser()
    args = parser.parse_args(["--stages", "2", "--sp-rounds", "50", "--intra-sp-interval", "150"])
    assert args.stages == 2
    assert args.sp_rounds == 50
    assert args.intra_sp_interval == 150
    assert not args.no_self_play
    assert "classic" in args.sp_scenarios


def test_wm_agent_arg_parser():
    import argparse
    from agent_code.my_wm_agent.train_curriculum import main
    # test parse through mocked main argv
    with mock.patch("agent_code.my_wm_agent.train_curriculum.run_stage") as mock_run_stage, \
         mock.patch("agent_code.my_wm_agent.train_curriculum.run_self_play_stage") as mock_sp:
        # run 0 rounds to exit quickly
        main(["--stages", "1", "--s1", "0", "--no-self-play"])
        assert mock_run_stage.call_count == 0
        assert mock_sp.call_count == 0


def test_spatial_dqn_stage_checkpoint_flow_with_self_play():
    """Verify that stage slices, intra self-play, and post-stage self-play share the canonical checkpoint_name and save_last is only True at the very end."""
    """Verify that stage slices save their own best/last checkpoints, and all self-play shares a single s{st}_self_play checkpoint pair."""
    from agent_code.my_spatial_dqn_agent.train_curriculum import main
    with mock.patch("parallel_train.run_parallel_stage") as mock_parallel_stage, \
         mock.patch("parallel_train.run_parallel_self_play_stage") as mock_parallel_sp, \
         mock.patch("parallel_train.load_or_create_shared_model") as mock_load_model, \
         mock.patch("sys.argv", ["train_curriculum.py", "--stages", "1", "--s1", "400", "--intra-sp-interval", "200", "--s1-sp", "50"]):
        mock_load_model.return_value = SimpleNamespace()
        main()
        # Stage 1 has 400 rounds, split into 2 slices (200, 200), 1 intra-sp, and 1 post-sp
        assert mock_parallel_stage.call_count == 2
        # Slice 1: normal training intermediate slice
        assert mock_parallel_stage.call_args_list[0].kwargs["checkpoint_name"] == "s1_coin_heaven"
        assert mock_parallel_stage.call_args_list[0].kwargs["save_best"] is True
        assert mock_parallel_stage.call_args_list[0].kwargs["save_last"] is False
        assert mock_parallel_stage.call_args_list[0].kwargs["is_self_play"] is False
        assert mock_parallel_stage.call_args_list[0].kwargs["eps_start"] == 1.0

        # Slice 2: normal training final slice -> saves last_model_s1_coin_heaven
        assert mock_parallel_stage.call_args_list[1].kwargs["checkpoint_name"] == "s1_coin_heaven"
        assert mock_parallel_stage.call_args_list[1].kwargs["save_best"] is True
        assert mock_parallel_stage.call_args_list[1].kwargs["save_last"] is True
        assert mock_parallel_stage.call_args_list[1].kwargs["is_self_play"] is False
        # Epsilon maintained and advanced across slices, not reset to 1.0 or 0.0
        assert 0.05 < mock_parallel_stage.call_args_list[1].kwargs["eps_start"] < 1.0

        assert mock_parallel_sp.call_count == 2
        # Intra self-play: uses s1_self_play, save_last is False because post-stage self-play exists
        assert mock_parallel_sp.call_args_list[0].kwargs["checkpoint_name"] == "s1_self_play"
        assert mock_parallel_sp.call_args_list[0].kwargs["save_best"] is True
        assert mock_parallel_sp.call_args_list[0].kwargs["save_last"] is False

        # Post-stage self-play: uses same s1_self_play, saves last_model_s1_self_play!
        assert mock_parallel_sp.call_args_list[1].kwargs["checkpoint_name"] == "s1_self_play"
        assert mock_parallel_sp.call_args_list[1].kwargs["save_best"] is True
        assert mock_parallel_sp.call_args_list[1].kwargs["save_last"] is True


def test_spatial_dqn_stage_checkpoint_flow_without_self_play():
    """Verify that when post-stage self-play is disabled, the final slice saves last_model."""
    from agent_code.my_spatial_dqn_agent.train_curriculum import main
    with mock.patch("parallel_train.run_parallel_stage") as mock_parallel_stage, \
         mock.patch("parallel_train.run_parallel_self_play_stage") as mock_parallel_sp, \
         mock.patch("parallel_train.load_or_create_shared_model") as mock_load_model, \
         mock.patch("sys.argv", ["train_curriculum.py", "--stages", "1", "--s1", "400", "--intra-sp-interval", "200", "--no-self-play"]):
        mock_load_model.return_value = SimpleNamespace()
        main()
        assert mock_parallel_stage.call_count == 1  # no self-play means [400]
        assert mock_parallel_stage.call_args_list[0].kwargs["checkpoint_name"] == "s1_coin_heaven"
        assert mock_parallel_stage.call_args_list[0].kwargs["save_best"] is True
        assert mock_parallel_stage.call_args_list[0].kwargs["save_last"] is True
        assert mock_parallel_sp.call_count == 0


def test_self_play_zero_discovery_spatial_dqn():
    from collections import deque
    from agent_code.my_spatial_dqn_agent.callbacks import act
    import numpy as np

    fake_self = SimpleNamespace(
        train=True,
        epsilon=1.0,  # normally would always explore randomly
        model=SimpleNamespace(q_values=lambda s, f: np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])),  # action 1 is 'RIGHT'
        logger=SimpleNamespace(debug=lambda msg: None, info=lambda msg: None),
        coordinate_history=deque(maxlen=20),
        _last_round=-1,
        _consecutive_stationary_steps=0,
        _steps_without_progress=0,
        _last_score=0,
        _last_crates=0,
        _think_times=[],
    )
    field = np.zeros((17, 17), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    fake_game_state = {
        "step": 1,
        "round": 1,
        "field": field,
        "explosion_map": np.zeros((17, 17), dtype=int),
        "self": ("learner", 0, True, (1, 1)),
        "others": [],
        "bombs": [],
        "coins": [],
        "user_input": None,
    }

    # With MY_WM_SELF_PLAY="1", act must NOT explore randomly and must return model's greedy choice ('RIGHT')
    with mock.patch.dict(os.environ, {"MY_WM_SELF_PLAY": "1"}):
        with mock.patch("random.choice", side_effect=AssertionError("random.choice should not be called during self-play")):
            action = act(fake_self, fake_game_state)
            assert action == "RIGHT"


def test_self_play_setup_training_spatial_dqn():
    from agent_code.my_spatial_dqn_agent.train import setup_training
    import logging
    fake_self = SimpleNamespace(
        model=SimpleNamespace(kind="spatial_dqn"),
        model_file="dummy.pt",
        logger=logging.getLogger("test_spatial"),
    )
    with mock.patch.dict(os.environ, {"MY_WM_SELF_PLAY": "1", "MY_WM_EPSILON": "0.8"}):
        setup_training(fake_self)
        assert fake_self.epsilon == 0.0
        assert fake_self.eps_end == 0.0
        assert fake_self.eps_decay == 1.0


