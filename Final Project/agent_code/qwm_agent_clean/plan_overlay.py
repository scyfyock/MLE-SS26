"""
Pygame overlay for interactive visual debugging of QWM Tree Search candidate plans.
Renders candidate paths on the board and detailed score panel on the right.
Degrades cleanly to no-ops when Pygame is unavailable or headless.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import settings as s

try:
    import pygame
    HAVE_PYGAME = True
except ImportError:
    HAVE_PYGAME = False


# Rank colors, best first (bright green, cyan, amber, pink, violet, grey)
PATH_COLORS = [
    (0, 255, 128),    # rank 0 - best, bright green
    (0, 190, 255),    # rank 1 - cyan
    (255, 210, 0),    # rank 2 - amber
    (255, 105, 180),  # rank 3 - pink
    (170, 130, 255),  # rank 4 - violet
    (150, 150, 150),  # rank 5 - grey
]

LINE_WIDTHS = [5, 3, 3, 2, 2, 1]
BLOCKED_COLOR = (255, 60, 60)


@dataclass
class CandidatePlanTrace:
    """Diagnostic trace for a single evaluated QWM candidate branch."""
    rank: int
    actions: List[str]
    tiles: List[Tuple[int, int]]
    total_score: float
    predicted_return: Optional[float] = None
    value_bootstrap: Optional[float] = None
    continuation_probability: Optional[float] = 1.0
    risk_penalty: Optional[float] = 0.0
    uncertainty_penalty: Optional[float] = 0.0
    actor_prior_penalty: Optional[float] = 0.0
    blocked_from: Optional[int] = None
    bomb_steps: List[int] = field(default_factory=list)
    first_action: str = 'WAIT'
    first_action_legal: bool = True


@dataclass
class PlannerOverlayTrace:
    """Complete diagnostic trace of QWM Tree Search for one game step."""
    round_id: int
    env_step: int
    agent_position: Tuple[int, int]
    planner_elapsed_ms: float
    actor_action: str
    planner_action: str
    planner_changed_action: bool
    fallback_used: bool
    candidates: List[CandidatePlanTrace] = field(default_factory=list)


def walk_path(
    start_pos: Tuple[int, int],
    actions: List[str],
    arena: Optional[Any] = None,
    bombs: Optional[List[Tuple[Tuple[int, int], int]]] = None
) -> Tuple[List[Tuple[int, int]], Optional[int], List[int]]:
    """
    Simulates walking an action sequence on the board with dynamic bomb explosions
    and future crate destruction physics.
    
    If a bomb is dropped or already active, its explosion after timer ticks will
    vaporize destructible crates in its blast radius, allowing future lookahead
    steps to walk through the cleared passageways without being falsely blocked!
    
    Returns: (visited_tiles, blocked_step_idx_or_None, bomb_steps)
    """
    import numpy as np

    cx, cy = start_pos
    tiles = [(cx, cy)]
    blocked_from = None
    bomb_steps = []

    del_map = {
        'UP': (0, -1),
        'DOWN': (0, 1),
        'LEFT': (-1, 0),
        'RIGHT': (1, 0),
        'WAIT': (0, 0),
        'BOMB': (0, 0)
    }

    # Make mutable copy of arena so future explosions destroy crates
    dynamic_arena = np.copy(arena) if arena is not None else None
    
    # Active bombs: list of [bx, by, timer]
    active_bombs = []
    if bombs:
        for b_entry in bombs:
            # Handle both ((x,y), timer) and (x, y, timer)
            if isinstance(b_entry, (tuple, list)):
                if isinstance(b_entry[0], (tuple, list)):
                    active_bombs.append([b_entry[0][0], b_entry[0][1], int(b_entry[1])])
                elif len(b_entry) >= 3:
                    active_bombs.append([b_entry[0], b_entry[1], int(b_entry[2])])

    for step_idx, act in enumerate(actions):
        # 1. Process bomb placement
        if act == 'BOMB':
            bomb_steps.append(step_idx)
            active_bombs.append([cx, cy, 4])  # Standard 4-tick bomb timer

        # 2. Advance bomb countdowns and simulate blast destruction
        newly_active_bombs = []
        for b in active_bombs:
            b[2] -= 1  # Countdown 1 tick
            if b[2] <= 0:
                # Bomb detonates! Destroy soft crates (1) in 4 directions
                bx, by = b[0], b[1]
                if dynamic_arena is not None:
                    for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                        for r in range(1, 4):
                            tx, ty = bx + dx * r, by + dy * r
                            if tx < 0 or tx >= dynamic_arena.shape[0] or ty < 0 or ty >= dynamic_arena.shape[1]:
                                break
                            if dynamic_arena[tx, ty] == -1:
                                break  # Indestructible stone wall stops blast
                            if dynamic_arena[tx, ty] == 1:
                                dynamic_arena[tx, ty] = 0  # Crate vaporized into walkable path!
                                break  # Crate absorbs blast
            else:
                newly_active_bombs.append(b)
        active_bombs = newly_active_bombs

        # 3. Compute prospective movement
        dx, dy = del_map.get(act, (0, 0))
        nx, ny = cx + dx, cy + dy

        # Check bounds and obstruction against dynamic arena state at this prospective time step
        if dynamic_arena is not None:
            # Out of bounds or indestructible stone wall (-1)
            if nx < 0 or nx >= dynamic_arena.shape[0] or ny < 0 or ny >= dynamic_arena.shape[1] or dynamic_arena[nx, ny] == -1:
                if blocked_from is None:
                    blocked_from = step_idx
                nx, ny = cx, cy
            # Soft crate that has NOT yet been destroyed
            elif dynamic_arena[nx, ny] == 1:
                if blocked_from is None:
                    blocked_from = step_idx
                nx, ny = cx, cy

        cx, cy = nx, ny
        tiles.append((cx, cy))

    return tiles, blocked_from, bomb_steps


_LATEST_TRACE: Optional[PlannerOverlayTrace] = None


def publish_overlay_trace(trace: Optional[PlannerOverlayTrace]):
    """Publishes latest PlannerOverlayTrace for GUI render."""
    global _LATEST_TRACE
    _LATEST_TRACE = trace


def clear_overlay_trace():
    """Clears published overlay trace on round reset or disable."""
    global _LATEST_TRACE
    _LATEST_TRACE = None


def get_latest_overlay_trace() -> Optional[PlannerOverlayTrace]:
    """Returns currently published trace."""
    return _LATEST_TRACE


def tile_center(x: int, y: int) -> Tuple[int, int]:
    """Converts board grid coordinate (x, y) to surface pixel center."""
    return (
        s.GRID_OFFSET[0] + s.GRID_SIZE * x + s.GRID_SIZE // 2,
        s.GRID_OFFSET[1] + s.GRID_SIZE * y + s.GRID_SIZE // 2,
    )


def _draw_path(screen: Any, path: CandidatePlanTrace, color: Tuple[int, int, int], width: int):
    """Draws one candidate path polyline with markers."""
    if not HAVE_PYGAME or not isinstance(screen, pygame.Surface) or not hasattr(path, 'tiles') or not path.tiles:
        return

    points = [tile_center(x, y) for (x, y) in path.tiles]
    offset = (path.rank - 2) * 3
    points = [(px + offset, py + offset) for (px, py) in points]

    if len(points) > 1:
        pygame.draw.lines(screen, color, False, points, width)

    # Start marker
    pygame.draw.circle(screen, color, points[0], max(4, width + 1))

    # Step dots along path
    for i, pt in enumerate(points[1:], start=1):
        pygame.draw.circle(screen, color, pt, max(2, width - 1))

        # Blocked marker if agent tried to walk into wall/crate
        if path.blocked_from is not None and i == path.blocked_from + 1:
            pygame.draw.circle(screen, BLOCKED_COLOR, pt, max(5, width + 2), 2)

    # Hollow BOMB rings on bomb steps
    for step_idx in path.bomb_steps:
        if step_idx < len(points):
            pygame.draw.circle(screen, color, points[step_idx], s.GRID_SIZE // 3, 2)

    # Heading arrowhead at end of path
    if len(points) > 1 and points[-1] != points[-2]:
        _draw_arrowhead(screen, points[-2], points[-1], color, width)


def _draw_arrowhead(screen: Any, from_pt: Tuple[int, int], to_pt: Tuple[int, int], color: Tuple[int, int, int], width: int):
    """Draws heading arrowhead triangle at end of path."""
    dx, dy = to_pt[0] - from_pt[0], to_pt[1] - from_pt[1]
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / length, dy / length

    size = 5 + width
    tip = (to_pt[0] + ux * size * 0.6, to_pt[1] + uy * size * 0.6)
    left = (to_pt[0] - uy * size * 0.5, to_pt[1] + ux * size * 0.5)
    right = (to_pt[0] + uy * size * 0.5, to_pt[1] - ux * size * 0.5)

    pygame.draw.polygon(screen, color, [(int(px), int(py)) for (px, py) in (tip, left, right)])


def _format_actions(actions: List[str], limit: int = 8) -> str:
    """Formats compact action string: e.g. URR.B"""
    mapping = {'UP': 'U', 'DOWN': 'D', 'LEFT': 'L', 'RIGHT': 'R', 'WAIT': '.', 'BOMB': 'B'}
    head = ''.join(mapping.get(a, '?') for a in actions[:limit])
    extra = len(actions) - limit
    return head + (f'+{extra}' if extra > 0 else '')


def render_overlay(screen: Any, gui: Any):
    """
    Renders top-K candidate plan paths and legend panel.
    Called after board elements are drawn and before display flip.
    Ensures candidate path origins are dynamically anchored to the agent's
    live coordinates so drawn lines never lag behind movement.
    """
    if not HAVE_PYGAME or _LATEST_TRACE is None:
        return

    trace = _LATEST_TRACE
    candidates = getattr(trace, 'candidates', []) or []
    if not candidates:
        return

    # Find the active QWM agent's live coordinates on the board
    agent_pos = None
    if hasattr(gui, 'world') and hasattr(gui.world, 'active_agents'):
        for a in gui.world.active_agents:
            code_name = getattr(a, 'code_name', '').lower()
            name = getattr(a, 'name', '').lower()
            if 'qwm' in code_name or 'qwm' in name or 'spatial' in code_name:
                agent_pos = (a.x, a.y)
                break
        if agent_pos is None and gui.world.active_agents:
            agent_pos = (gui.world.active_agents[0].x, gui.world.active_agents[0].y)

    # Draw worst-first so best path renders on top
    for path in sorted(candidates, key=lambda p: -p.rank):
        color = PATH_COLORS[path.rank % len(PATH_COLORS)]
        width = LINE_WIDTHS[path.rank % len(LINE_WIDTHS)]

        # If the agent has already advanced past trace.agent_position (e.g. post-step render),
        # slice path so it originates directly at the agent's current position:
        path_to_draw = path
        if agent_pos is not None and hasattr(path, 'tiles') and len(path.tiles) > 1:
            if agent_pos != path.tiles[0] and agent_pos == path.tiles[1]:
                # Agent has traversed step 0; align path to start at current tile
                path_to_draw = CandidatePlanTrace(
                    rank=path.rank,
                    actions=path.actions[1:] if len(path.actions) > 1 else path.actions,
                    tiles=path.tiles[1:],
                    total_score=path.total_score,
                    predicted_return=path.predicted_return,
                    value_bootstrap=path.value_bootstrap,
                    continuation_probability=path.continuation_probability,
                    risk_penalty=path.risk_penalty,
                    uncertainty_penalty=path.uncertainty_penalty,
                    actor_prior_penalty=path.actor_prior_penalty,
                    blocked_from=max(0, path.blocked_from - 1) if path.blocked_from is not None else None,
                    bomb_steps=[max(0, b - 1) for b in path.bomb_steps if b >= 1],
                    first_action=path.actions[1] if len(path.actions) > 1 else path.first_action,
                    first_action_legal=path.first_action_legal,
                )

        _draw_path(screen, path_to_draw, color, width)

    _render_legend(screen, gui, trace)


def _render_legend(screen: Any, gui: Any, trace: PlannerOverlayTrace):
    """Renders right-side score panel legend."""
    if not HAVE_PYGAME or not isinstance(screen, pygame.Surface):
        return
    x = s.GRID_OFFSET[0] + s.COLS * s.GRID_SIZE + 15
    y = 250

    if hasattr(gui, 'render_text'):
        gui.render_text("QWM TREE SEARCH PLANS", x, y, (200, 200, 200), size='small')
        y += 16

        for path in sorted(trace.candidates, key=lambda p: p.rank):
            color = PATH_COLORS[path.rank % len(PATH_COLORS)]
            pygame.draw.rect(screen, color, pygame.Rect(x, y + 3, 10, 4))

            label = f"#{path.rank + 1} {_format_actions(path.actions)} Q:{path.total_score:+.2f}"
            gui.render_text(label, x + 16, y, color, size='small')
            y += 13

            details = []
            if path.predicted_return is not None:
                details.append(f"R:{path.predicted_return:+.2f}")
            if path.value_bootstrap is not None:
                details.append(f"V:{path.value_bootstrap:+.2f}")
            if path.continuation_probability is not None and path.continuation_probability < 0.99:
                details.append(f"Cont:{path.continuation_probability:.0%}")

            if details:
                gui.render_text("  " + " | ".join(details), x + 16, y, (140, 140, 140), size='small')
                y += 14

        y += 5
        actor_act = getattr(trace, 'actor_action', 'WAIT')
        plan_act = getattr(trace, 'planner_action', 'WAIT')
        timing = getattr(trace, 'planner_elapsed_ms', 0.0)

        gui.render_text(f"actor: {actor_act}  qwm: {plan_act}", x, y, (180, 180, 180), size='small')
        y += 13
        gui.render_text(f"eval time: {timing:.1f}ms", x, y, (150, 150, 150), size='small')

