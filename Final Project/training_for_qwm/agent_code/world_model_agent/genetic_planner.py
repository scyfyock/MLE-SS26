"""Genetic Algorithm (GA) sequence planner using the learned World Model.

Searches for an optimal action sequence of length H using forward simulation
through LearnedWorldModel.predict(). Incorporates step rewards, death/suicide
penalties, and delayed bomb outcomes into sequence fitness.
"""

from dataclasses import dataclass, field
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .callbacks import ACTION_TO_INDEX, ACTIONS, state_to_key
from .world_model import LearnedWorldModel, WorldState, game_state_to_world_state


@dataclass
class GeneticPlannerConfig:
    horizon: int = 3
    pop_size: int = 10
    generations: int = 3
    mutation_rate: float = 0.25
    crossover_rate: float = 0.70
    elite_count: int = 2
    tournament_size: int = 2
    gamma: float = 0.95
    death_penalty: float = 40.0
    suicide_penalty: float = 30.0
    seed: Optional[int] = None


@dataclass
class PlanningStats:
    best_sequence: List[str]
    best_fitness: float
    initial_best_fitness: float
    fitness_history: List[float]  # Best fitness per generation
    mean_fitness_history: List[float]
    eval_count: int
    cache_hits: int
    elapsed_ms: float


class GeneticAlgorithmPlanner:
    """Plans multi-step action sequences using a GA over imagined world states."""

    def __init__(self, world_model: LearnedWorldModel, config: Optional[GeneticPlannerConfig] = None):
        self.world_model = world_model
        self.config = config or GeneticPlannerConfig()
        self.rng = random.Random(self.config.seed)
        self.cache: Dict[Tuple, Tuple] = {}
        self.history: List[PlanningStats] = []

    def _evaluate_sequence(
        self,
        sequence: List[str],
        root_world_state: WorldState,
        root_state: tuple,
        root_valid_actions: List[str],
    ) -> float:
        """Evaluate cumulative discounted fitness of an action sequence using prefix caching."""
        cfg = self.config
        cumulative_fitness = 0.0
        current_world_state = root_world_state
        current_state = root_state
        current_valid = root_valid_actions

        prefix = ()
        for depth, action in enumerate(sequence):
            prefix = prefix + (action,)
            discount = cfg.gamma ** depth

            # Check action validity
            if current_valid and action not in current_valid:
                cumulative_fitness -= 15.0 * discount
                break

            # Cache lookup for transition from this sequence prefix
            if prefix in self.cache:
                self.cache_hits += 1
                (
                    next_world_state,
                    next_state,
                    step_fitness,
                    next_valid,
                    is_terminal,
                ) = self.cache[prefix]
            else:
                self.eval_count += 1
                action_idx = ACTION_TO_INDEX[action]
                pred = self.world_model.predict(
                    current_world_state,
                    current_state,
                    action_idx,
                )

                # Fitness components
                step_reward = float(pred.reward)
                death_cost = cfg.death_penalty * float(pred.death_probability)
                suicide_cost = cfg.suicide_penalty * float(pred.suicide_probability)

                bomb_bonus = 0.0
                if action == "BOMB" and getattr(self.world_model, "is_bomb_outcome_ready", False):
                    bomb_pred = self.world_model.predict_bomb_outcome(current_state)
                    bomb_bonus = (
                        2.0 * bomb_pred.crate_probability
                        + 20.0 * max(0.0, bomb_pred.kill_probability - 0.10)
                        - 4.0 * bomb_pred.escape_probability
                    )

                step_fitness = step_reward - death_cost - suicide_cost + bomb_bonus
                next_world_state = pred.next_world_state
                next_state = pred.next_state
                next_valid = [ACTIONS[idx] for idx in pred.next_valid_actions]
                is_terminal = pred.done_probability >= 0.5

                if is_terminal and pred.death_probability > 0.3:
                    step_fitness -= 40.0

                self.cache[prefix] = (
                    next_world_state,
                    next_state,
                    step_fitness,
                    next_valid,
                    is_terminal,
                )

            cumulative_fitness += discount * step_fitness

            if is_terminal:
                break

            current_world_state = next_world_state
            current_state = next_state
            current_valid = next_valid

        return float(cumulative_fitness)

    def _initialize_population(
        self,
        valid_actions: List[str],
        greedy_action: Optional[str] = None,
    ) -> List[List[str]]:
        """Create initial population with greedy seeding and diverse random paths."""
        cfg = self.config
        population = []
        action_pool = list(valid_actions) if valid_actions else list(ACTIONS)

        # Individual 0: Seeded with Q-greedy action followed by survival moves
        if greedy_action and greedy_action in action_pool:
            first_action = greedy_action
        else:
            first_action = self.rng.choice(action_pool)

        greedy_seq = [first_action] + [self.rng.choice(ACTIONS) for _ in range(cfg.horizon - 1)]
        population.append(greedy_seq)

        # Individual 1: Pure safe move / wait sequence
        safe_first = "WAIT" if "WAIT" in action_pool else action_pool[0]
        safe_seq = [safe_first] + ["WAIT"] * (cfg.horizon - 1)
        if len(population) < cfg.pop_size:
            population.append(safe_seq)

        # Remaining individuals: Random sequences starting with legal root actions
        while len(population) < cfg.pop_size:
            root_act = self.rng.choice(action_pool)
            rest = [self.rng.choice(ACTIONS) for _ in range(cfg.horizon - 1)]
            population.append([root_act] + rest)

        return population[: cfg.pop_size]

    def _crossover(self, parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
        """Single-point crossover between two action sequences."""
        cfg = self.config
        if cfg.horizon <= 1 or self.rng.random() > cfg.crossover_rate:
            return list(parent1), list(parent2)

        point = self.rng.randint(1, cfg.horizon - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutate(self, individual: List[str], valid_root_actions: List[str]) -> List[str]:
        """Mutate actions in the sequence with mutation_rate probability."""
        cfg = self.config
        mutated = list(individual)
        for depth in range(cfg.horizon):
            if self.rng.random() < cfg.mutation_rate:
                pool = valid_root_actions if depth == 0 and valid_root_actions else ACTIONS
                mutated[depth] = self.rng.choice(pool)
        return mutated

    def _tournament_select(self, population: List[List[str]], fitnesses: List[float]) -> List[str]:
        """Tournament selection."""
        cfg = self.config
        k = min(cfg.tournament_size, len(population))
        contestant_indices = self.rng.sample(range(len(population)), k)
        best_idx = max(contestant_indices, key=lambda idx: fitnesses[idx])
        return list(population[best_idx])

    def plan(
        self,
        game_state: dict,
        valid_actions: List[str],
        q_values: Optional[np.ndarray] = None,
    ) -> Tuple[str, List[str], float, PlanningStats]:
        """Execute GA search from the given game_state and return the best action."""
        start_time = time.perf_counter()
        cfg = self.config
        self.cache.clear()
        self.eval_count = 0
        self.cache_hits = 0

        # Encode current state for model predictions
        root_world_state = game_state_to_world_state(game_state)
        root_state = state_to_key(game_state)
        action_pool = list(valid_actions) if valid_actions else list(ACTIONS)

        # Determine greedy seed from Q-values if available
        greedy_action = None
        if q_values is not None and len(q_values) == len(ACTIONS):
            valid_q = {a: q_values[ACTION_TO_INDEX[a]] for a in action_pool}
            if valid_q:
                greedy_action = max(valid_q.items(), key=lambda kv: kv[1])[0]

        # Handle horizon 0 or empty search
        if cfg.horizon <= 0:
            chosen = greedy_action or action_pool[0]
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return chosen, [chosen], 0.0, PlanningStats(
                best_sequence=[chosen],
                best_fitness=0.0,
                initial_best_fitness=0.0,
                fitness_history=[0.0],
                mean_fitness_history=[0.0],
                eval_count=0,
                cache_hits=0,
                elapsed_ms=elapsed_ms,
            )

        # 1. Initialize population
        population = self._initialize_population(action_pool, greedy_action=greedy_action)
        fitnesses = [
            self._evaluate_sequence(ind, root_world_state, root_state, action_pool)
            for ind in population
        ]

        best_idx = int(np.argmax(fitnesses))
        best_sequence = list(population[best_idx])
        best_fitness = float(fitnesses[best_idx])
        initial_best_fitness = best_fitness

        fitness_history = [best_fitness]
        mean_fitness_history = [float(np.mean(fitnesses))]

        # 2. Evolution Loop
        for _gen in range(cfg.generations):
            # Elitism: retain top individuals
            ranked_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
            new_population = [list(population[i]) for i in ranked_indices[: cfg.elite_count]]

            # Fill rest of population with crossover and mutation
            while len(new_population) < cfg.pop_size:
                parent1 = self._tournament_select(population, fitnesses)
                parent2 = self._tournament_select(population, fitnesses)

                child1, child2 = self._crossover(parent1, parent2)
                new_population.append(self._mutate(child1, action_pool))
                if len(new_population) < cfg.pop_size:
                    new_population.append(self._mutate(child2, action_pool))

            population = new_population[: cfg.pop_size]
            fitnesses = [
                self._evaluate_sequence(ind, root_world_state, root_state, action_pool)
                for ind in population
            ]

            gen_best_idx = int(np.argmax(fitnesses))
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = float(fitnesses[gen_best_idx])
                best_sequence = list(population[gen_best_idx])

            fitness_history.append(best_fitness)
            mean_fitness_history.append(float(np.mean(fitnesses)))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        stats = PlanningStats(
            best_sequence=best_sequence,
            best_fitness=best_fitness,
            initial_best_fitness=initial_best_fitness,
            fitness_history=fitness_history,
            mean_fitness_history=mean_fitness_history,
            eval_count=self.eval_count,
            cache_hits=self.cache_hits,
            elapsed_ms=elapsed_ms,
        )
        self.history.append(stats)

        first_action = best_sequence[0] if best_sequence else (action_pool[0] if action_pool else "WAIT")
        return first_action, best_sequence, best_fitness, stats
