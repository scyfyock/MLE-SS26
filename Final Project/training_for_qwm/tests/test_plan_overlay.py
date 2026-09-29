#!/usr/bin/env python3
"""
Unit tests for qwm_agent_veryclean.plan_overlay.
Verifies top-path-only rendering, self-contained pygame hooks, and zero external dependencies.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from agent_code.qwm_agent_veryclean.plan_overlay import (
    CandidatePlanTrace,
    PlannerOverlayTrace,
    publish_overlay_trace,
    clear_overlay_trace,
    get_latest_overlay_trace,
    walk_path,
    render_overlay,
    install_render_hook,
    tile_center,
    _format_actions,
    HAVE_PYGAME,
)


def test_walk_path_simulation():
    """Verify walking action sequences and dynamic bomb simulation."""
    arena = np.zeros((17, 17), dtype=int)
    arena[0, :] = -1
    arena[-1, :] = -1
    arena[:, 0] = -1
    arena[:, -1] = -1
    arena[3, 3] = 1   # Soft crate

    # 1. Clear movement
    tiles, blocked, bombs = walk_path((1, 1), ['RIGHT', 'DOWN'], arena=arena)
    assert tiles == [(1, 1), (2, 1), (2, 2)]
    assert blocked is None
    assert bombs == []

    # 2. Blocked by wall
    tiles_wall, blocked_wall, _ = walk_path((1, 1), ['UP', 'UP'], arena=arena)
    assert blocked_wall == 0  # Blocked on step 0 (into wall at y=0)

    # 3. Bomb action tracking
    tiles_bomb, _, bomb_steps = walk_path((1, 1), ['BOMB', 'RIGHT'], arena=arena)
    assert 0 in bomb_steps


def test_top_path_only_selection():
    """Verify render_overlay renders strictly the top (rank 0) path."""
    clear_overlay_trace()

    top_cand = CandidatePlanTrace(
        rank=0,
        actions=['RIGHT', 'DOWN', 'BOMB'],
        tiles=[(1, 1), (2, 1), (2, 2)],
        total_score=5.50,
        predicted_return=1.20,
        value_bootstrap=4.30,
        bomb_steps=[2],
    )
    second_cand = CandidatePlanTrace(
        rank=1,
        actions=['LEFT', 'LEFT'],
        tiles=[(1, 1), (0, 1), (0, 1)],
        total_score=-2.0,
        predicted_return=-1.0,
        value_bootstrap=-1.0,
    )

    trace = PlannerOverlayTrace(
        round_id=1,
        env_step=5,
        agent_position=(1, 1),
        planner_elapsed_ms=4.2,
        actor_action='RIGHT',
        planner_action='RIGHT',
        planner_changed_action=False,
        fallback_used=False,
        candidates=[second_cand, top_cand],  # Intentionally unordered
    )

    publish_overlay_trace(trace)
    assert get_latest_overlay_trace() is trace

    # Mock screen and verify only top path is drawn
    drawn_paths = []
    with patch('agent_code.qwm_agent_veryclean.plan_overlay._draw_path') as mock_draw:
        mock_draw.side_effect = lambda screen, path, color, width: drawn_paths.append(path)
        render_overlay(MagicMock(), gui=None)

        # Assert exactly ONE path was drawn, and it is the rank 0 path!
        assert len(drawn_paths) == 1
        assert drawn_paths[0].rank == 0
        assert drawn_paths[0].actions == ['RIGHT', 'DOWN', 'BOMB']


def test_self_contained_render_hook():
    """Verify install_render_hook patches pygame flip/update without external dependencies."""
    if not HAVE_PYGAME:
        pytest.skip("Pygame not installed")

    import pygame
    install_render_hook()

    # Verify pygame.display.flip is hooked
    assert getattr(pygame.display, '_qwm_flip_hooked', False) is True


def test_format_actions():
    """Verify compact action string formatting."""
    assert _format_actions(['UP', 'RIGHT', 'RIGHT', 'WAIT', 'BOMB']) == 'URR.B'
    assert _format_actions(['DOWN'] * 15, limit=5) == 'DDDDD+10'


def test_tile_center():
    """Verify tile center calculation."""
    cx, cy = tile_center(0, 0)
    assert cx > 0 and cy > 0
