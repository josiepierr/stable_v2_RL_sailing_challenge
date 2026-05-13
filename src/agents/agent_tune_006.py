
import numpy as np

try:
    from evaluator.base_agent import BaseAgent
except Exception:
    try:
        from agents.base_agent import BaseAgent
    except Exception:
        class BaseAgent:
            def reset(self): pass
            def seed(self, seed=None): pass


class MyAgent(BaseAgent):
    GRID = 128
    WIND_SIZE = GRID * GRID * 2
    WORLD_SIZE = GRID * GRID

    GOAL = np.array([64.0, 127.0], dtype=np.float32)

    DIRECTIONS = np.array([
        [0, 1],
        [1, 1],
        [1, 0],
        [1, -1],
        [0, -1],
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, 0],
    ], dtype=np.float32)

    LEFT_WAYPOINTS = np.array([
        [28.0, 30.0],
        [23.0, 62.0],
        [27.0, 96.0],
        [48.0, 116.0],
        [64.0, 127.0],
    ], dtype=np.float32)

    RIGHT_WAYPOINTS = np.array([
        [100.0, 30.0],
        [105.0, 62.0],
        [101.0, 96.0],
        [80.0, 116.0],
        [64.0, 127.0],
    ], dtype=np.float32)

    def __init__(self):
        self.np_random = np.random.default_rng(0)
        self.side = None
        self.waypoint_index = 0

        self.horizon = 2
        self.progress_weight = 4.0
        self.waypoint_weight = 1.6
        self.north_speed_weight = 7.0
        self.goal_weight = 0.35
        self.collision_penalty = 500000.0
        self.border_penalty = 2000.0
        self.center_penalty = 60.0
        self.wp_radius = 14.0
        self.side_margin = 0.05
        self.stay_penalty = 200.0
        self.south_penalty = 300.0

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    def reset(self):
        self.side = None
        self.waypoint_index = 0

    def act(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        x, y, vx, vy, wx, wy = obs[:6]

        wind_field, world = self._extract_maps(obs)

        position = np.array([x, y], dtype=np.float32)

        candidate_waypoints = self._candidate_waypoints(position, wind_field)

        best_action = 0
        best_score = -1e30

        candidates = self._candidate_sequences()

        for waypoint in candidate_waypoints:
            for seq in candidates:
                score = self._simulate_sequence(
                    position=np.array([x, y], dtype=np.float32),
                    velocity=np.array([vx, vy], dtype=np.float32),
                    seq=seq,
                    wind_field=wind_field,
                    world=world,
                    waypoint=waypoint,
                )

                if score > best_score:
                    best_score = score
                    best_action = int(seq[0])

        return int(best_action)

    def _candidate_waypoints(self, position, wind_field):
        y = float(position[1])

        if y < 40:
            return [
                np.array([28.0, 35.0], dtype=np.float32),
                np.array([100.0, 35.0], dtype=np.float32),
                np.array([64.0, 65.0], dtype=np.float32),
            ]

        if y < 90:
            return [
                np.array([23.0, 92.0], dtype=np.float32),
                np.array([105.0, 92.0], dtype=np.float32),
                np.array([64.0, 105.0], dtype=np.float32),
            ]

        if y < 115:
            return [
                np.array([48.0, 116.0], dtype=np.float32),
                np.array([80.0, 116.0], dtype=np.float32),
                self.GOAL,
            ]

        return [self.GOAL]


    def _extract_maps(self, obs):
        wind_flat = obs[6:6 + self.WIND_SIZE]
        world_flat = obs[6 + self.WIND_SIZE:6 + self.WIND_SIZE + self.WORLD_SIZE]

        wind_field = wind_flat.reshape(self.GRID, self.GRID, 2)
        world = world_flat.reshape(self.GRID, self.GRID)

        return wind_field, world

    def _candidate_sequences(self):
        h = int(self.horizon)

        if h <= 1:
            return [[a] for a in range(9)]

        # Horizon 2 is fast enough and quite effective.
        if h == 2:
            useful = [0, 1, 2, 6, 7, 8]
            candidates = []

            for a in range(9):
                candidates.append([a, a])

            for a in range(9):
                for b in useful:
                    candidates.append([a, b])

            return self._deduplicate(candidates)

        # Horizon >= 3: reduced candidates to keep evaluation fast.
        useful = [0, 1, 2, 6, 7, 8]
        candidates = []

        for a in range(9):
            candidates.append([a] * h)

        for a in range(9):
            for b in useful:
                candidates.append([a] + [b] * (h - 1))

        motifs = [[1, 7], [7, 1], [0, 1], [0, 7], [1, 0], [7, 0]]
        for motif in motifs:
            seq = (motif * ((h + len(motif) - 1) // len(motif)))[:h]
            candidates.append(seq)

        return self._deduplicate(candidates)

    @staticmethod
    def _deduplicate(candidates):
        seen = set()
        out = []
        for seq in candidates:
            key = tuple(int(x) for x in seq)
            if key not in seen:
                seen.add(key)
                out.append(list(key))
        return out

    def _choose_side(self, wind_field):
        left = wind_field[15:112, 12:40, :]
        right = wind_field[15:112, 88:116, :]

        left_score = float(np.mean(left[..., 1]) - 0.10 * abs(np.mean(left[..., 0])))
        right_score = float(np.mean(right[..., 1]) - 0.10 * abs(np.mean(right[..., 0])))

        if abs(left_score - right_score) <= self.side_margin:
            wx_start = float(wind_field[0, 64, 0])
            return "RIGHT" if wx_start >= 0 else "LEFT"

        return "LEFT" if left_score > right_score else "RIGHT"

    def _waypoints(self):
        return self.LEFT_WAYPOINTS if self.side == "LEFT" else self.RIGHT_WAYPOINTS

    def _current_waypoint(self):
        wps = self._waypoints()
        return wps[min(self.waypoint_index, len(wps) - 1)]

    def _update_waypoint(self, position):
        wps = self._waypoints()

        if self.waypoint_index >= len(wps) - 1:
            return

        current = wps[self.waypoint_index]
        dist = np.linalg.norm(position - current)

        if dist <= self.wp_radius:
            self.waypoint_index += 1

        y = position[1]
        if y > 45 and self.waypoint_index < 1:
            self.waypoint_index = 1
        if y > 82 and self.waypoint_index < 2:
            self.waypoint_index = 2
        if y > 108 and self.waypoint_index < 3:
            self.waypoint_index = 3

    @staticmethod
    def _sailing_efficiency(boat_direction, wind_direction):
        wind_from = -wind_direction
        angle = np.arccos(np.clip(np.dot(wind_from, boat_direction), -1.0, 1.0))

        if angle < np.pi / 4:
            return 0.05
        if angle < np.pi / 2:
            return 0.5 + 0.5 * (angle - np.pi / 4) / (np.pi / 4)
        if angle < 3 * np.pi / 4:
            return 1.0

        eff = 1.0 - 0.5 * (angle - 3 * np.pi / 4) / (np.pi / 4)
        return max(0.5, eff)

    def _next_state(self, position, velocity, action, wind_field):
        x = int(np.clip(round(float(position[0])), 0, self.GRID - 1))
        y = int(np.clip(round(float(position[1])), 0, self.GRID - 1))

        wind = wind_field[y, x]
        direction = self.DIRECTIONS[int(action)].astype(np.float32)

        wind_norm = np.linalg.norm(wind)

        if wind_norm > 1e-12:
            wind_normalized = wind / wind_norm
            direction_norm = np.linalg.norm(direction)

            if direction_norm < 1e-12:
                direction_normalized = np.array([1.0, 0.0], dtype=np.float32)
            else:
                direction_normalized = direction / direction_norm

            eff = self._sailing_efficiency(direction_normalized, wind_normalized)
            theoretical_velocity = direction * eff * wind_norm * 0.4

            speed = np.linalg.norm(theoretical_velocity)
            if speed > 8.0:
                theoretical_velocity = theoretical_velocity / speed * 8.0

            new_velocity = theoretical_velocity + 0.3 * (velocity - theoretical_velocity)

            speed = np.linalg.norm(new_velocity)
            if speed > 8.0:
                new_velocity = new_velocity / speed * 8.0
        else:
            new_velocity = 0.3 * velocity

        # Match environment discretization.
        new_velocity = np.where(new_velocity < 0, np.ceil(new_velocity), np.floor(new_velocity)).astype(np.float32)

        new_position = position + new_velocity
        new_position = np.clip(new_position, [0, 0], [self.GRID - 1, self.GRID - 1]).astype(np.float32)

        return new_position, new_velocity

    def _collision(self, position, world):
        x = int(np.clip(round(float(position[0])), 0, self.GRID - 1))
        y = int(np.clip(round(float(position[1])), 0, self.GRID - 1))
        return bool(world[y, x] > 0.5)

    def _distance_to_island(self, position, world):
        x0 = int(np.clip(round(float(position[0])), 0, self.GRID - 1))
        y0 = int(np.clip(round(float(position[1])), 0, self.GRID - 1))

        # Local window approximation for speed.
        r = 10
        x1, x2 = max(0, x0-r), min(self.GRID, x0+r+1)
        y1, y2 = max(0, y0-r), min(self.GRID, y0+r+1)

        sub = world[y1:y2, x1:x2]
        ys, xs = np.where(sub > 0.5)

        if len(xs) == 0:
            return 999.0

        xs = xs + x1
        ys = ys + y1

        d2 = (xs - position[0])**2 + (ys - position[1])**2
        return float(np.sqrt(np.min(d2)))

    def _score_state(self, position, velocity, world, waypoint, action, step_in_rollout):
        dist_wp = np.linalg.norm(position - waypoint)
        dist_goal = np.linalg.norm(position - self.GOAL)

        score = 0.0
        score += self.progress_weight * position[1]
        score -= self.waypoint_weight * dist_wp
        score -= self.goal_weight * dist_goal
        score += self.north_speed_weight * velocity[1]

        if int(action) == 8:
            score -= self.stay_penalty

        if int(action) in [3, 4, 5]:
            score -= self.south_penalty

        if position[0] < 3 or position[0] > 124:
            score -= self.border_penalty

        if 34 <= position[0] <= 94 and 12 <= position[1] <= 92:
            score -= self.center_penalty

        dist_island = self._distance_to_island(position, world)
        if dist_island < 8:
            score -= 500.0 * (8.0 - dist_island)

        if self._collision(position, world):
            score -= self.collision_penalty

        if dist_goal < 1.5:
            score += 1_000_000.0 / (1 + step_in_rollout)

        return float(score)

    def _simulate_sequence(self, position, velocity, seq, wind_field, world, waypoint):
        score = 0.0
        pos = position.copy()
        vel = velocity.copy()

        previous_goal_dist = np.linalg.norm(pos - self.GOAL)

        for j, action in enumerate(seq):
            pos, vel = self._next_state(pos, vel, action, wind_field)

            if self._collision(pos, world):
                return -self.collision_penalty

            current_goal_dist = np.linalg.norm(pos - self.GOAL)
            score += 10.0 * (previous_goal_dist - current_goal_dist)
            previous_goal_dist = current_goal_dist

            score += self._score_state(pos, vel, world, waypoint, action, j)

        return float(score)
