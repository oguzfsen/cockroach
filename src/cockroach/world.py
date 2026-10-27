"""Deterministic 30 Hz physics. No rendering, file I/O or learning lives here."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from cockroach.config import WorldConfig

TAU = math.tau
FOOD_RADIUS = 7.0


class Action(IntEnum):
    WALK = 0
    LEFT = 1
    RIGHT = 2
    SNIFF = 3
    BACK = 4
    COURT = 5
    REST = 6


ACTION_NAMES = ("Yürü", "Sola dön", "Sağa dön", "Kokla", "Geri yürü", "Kur yap", "Dinlen")
OBS_NAMES = (
    "scent_left",
    "scent_front",
    "scent_right",
    "scent_change",
    "scent_lateral",
    "scent_ahead",
    "sample_freshness",
    "wall_left",
    "wall_front",
    "wall_right",
    "mate_left",
    "mate_front",
    "mate_right",
    "mate_rear",
    "fertility",
    "mate_fertility",
    "contact",
    "cooldown",
    "day_sin",
    "day_cos",
    "last_walk",
    "last_left",
    "last_right",
    "last_sniff",
    "last_back",
    "last_court",
    "last_rest",
)
OBS_SIZE = len(OBS_NAMES)


def wrap(angle: float) -> float:
    return (angle + math.pi) % TAU - math.pi


@dataclass
class Food:
    x: float
    y: float
    day: int
    vx: float = 0.0
    vy: float = 0.0


def reflect_axis(position: float, velocity: float, dt: float, low: float, high: float):
    """Reflect motion without losing overshoot, including multiple wall crossings."""
    span = high - low
    phase = (position - low + velocity * dt) % (2 * span)
    reflected = phase > span or (phase == span and velocity > 0) or (phase == 0 and velocity < 0)
    return low + (
        phase if phase <= span else 2 * span - phase
    ), -velocity if reflected else velocity


@dataclass
class Insect:
    sex: str
    x: float
    y: float
    angle: float
    action: Action = Action.REST
    score: float = 0.0
    eaten: int = 0
    missed: int = 0
    ate_day: int = -1
    fertile_until: float = 0.0
    cooldown_until: float = 0.0
    mating_until: float = 0.0
    mating_count: int = 0
    distance: float = 0.0
    collision: bool = False
    scent: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scent_change: float = 0.0
    sample_time: float = -10.0
    previous_scent: float = 0.0

    def fertile(self, now: float) -> bool:
        return self.fertile_until > now

    def mating(self, now: float) -> bool:
        return self.mating_until > now


class World:
    """Two agents share resources but receive independent observations and rewards.

    `step` holds a motor action for a complete decision. The viewer calls the same
    `begin_actions` + `tick` methods one physical frame at a time.
    """

    def __init__(self, config: WorldConfig | None = None, seed: int = 0):
        self.config = config or WorldConfig()
        self.rng = random.Random(seed)
        self.frame = 0
        self.day = 0
        self.eggs = 0
        self.pairings = 0
        self.food_spawned = 0
        self.events: list[tuple[float, str]] = []
        self.rewards = [0.0, 0.0]
        self.insects: list[Insect] = []
        self.food: list[Food] = []
        self.egg_sites: list[tuple[float, float]] = []
        self._mating_active = False
        r = self.config.radius
        for sex in ("female", "male"):
            for _ in range(1000):
                x = self.rng.uniform(3 * r, self.config.width - 3 * r)
                y = self.rng.uniform(3 * r, self.config.height - 3 * r)
                if all(math.hypot(x - b.x, y - b.y) > 4 * r for b in self.insects):
                    break
            self.insects.append(Insect(sex, x, y, self.rng.uniform(-math.pi, math.pi)))
        self._spawn_food()
        for b in self.insects:
            self._sample(b, active=False)

    @property
    def now(self) -> float:
        return self.frame / self.config.fps

    @property
    def done(self) -> bool:
        return self.frame >= self.config.episode_frames

    def log(self, message: str):
        self.events.append((self.now, message))
        self.events = self.events[-10:]

    def _spawn_food(self):
        c = self.config
        self.food.clear()
        for _ in range(2):
            for _attempt in range(1000):
                x = self.rng.uniform(c.radius * 2, c.width - c.radius * 2)
                y = self.rng.uniform(c.radius * 2, c.height - c.radius * 2)
                if all(math.hypot(x - f.x, y - f.y) > c.radius * 3 for f in self.food) and all(
                    math.hypot(x - b.x, y - b.y) > c.radius * 2 for b in self.insects
                ):
                    break
            heading = self.rng.uniform(-math.pi, math.pi)
            speed = c.speed * c.food_speed_ratio
            self.food.append(
                Food(x, y, self.day, math.cos(heading) * speed, math.sin(heading) * speed)
            )
        self.food_spawned += 2
        self.log(f"Gün {self.day + 1}: iki yeni besin")

    def _move_food(self):
        c = self.config
        for food in self.food:
            food.x, food.vx = reflect_axis(
                food.x, food.vx, c.dt, FOOD_RADIUS, c.width - FOOD_RADIUS
            )
            food.y, food.vy = reflect_axis(
                food.y, food.vy, c.dt, FOOD_RADIUS, c.height - FOOD_RADIUS
            )

    def scent_at(self, x: float, y: float, reach: float | None = None) -> float:
        """Finite-range, continuous odor. No food coordinates enter observations."""
        reach = self.config.sniff_range if reach is None else reach
        total = 0.0
        for f in self.food:
            d = math.hypot(x - f.x, y - f.y)
            if d < reach:
                total += math.exp(-d / self.config.scent_scale) * (1 - (d / reach) ** 2)
        return min(total, 1.0)

    def _sample(self, b: Insect, active: bool):
        reach = self.config.sniff_range if active else self.config.passive_range
        samples = []
        for offset in (-0.65, 0.0, 0.65):
            theta = b.angle + offset
            samples.append(
                self.scent_at(b.x + math.cos(theta) * 32, b.y + math.sin(theta) * 32, reach)
            )
        level = sum(samples) / 3
        b.scent_change = max(-1.0, min(1.0, (level - b.previous_scent) * 4))
        b.previous_scent = level
        b.scent = samples
        b.sample_time = self.now

    def _reward(self, i: int, value: float):
        self.rewards[i] += value
        self.insects[i].score += value

    def begin_actions(self, actions):
        if len(actions) != 2:
            raise ValueError("İki böcek için iki eylem gerekli.")
        parsed = [Action(a) for a in actions]
        self.rewards = [0.0, 0.0]
        for b, action in zip(self.insects, parsed, strict=False):
            b.action = action
            b.collision = False

    def _move(self, b: Insect):
        if b.mating(self.now):
            return
        c = self.config
        velocity = 0.0
        if b.action == Action.WALK:
            velocity = c.speed
        elif b.action in (Action.LEFT, Action.RIGHT):
            b.angle = wrap(b.angle + (-1 if b.action == Action.LEFT else 1) * c.turn_speed * c.dt)
        elif b.action == Action.BACK:
            velocity = -c.speed * 0.6
        elif b.action == Action.COURT and b.fertile(self.now):
            mate = self.insects[1] if b is self.insects[0] else self.insects[0]
            if math.hypot(mate.x - b.x, mate.y - b.y) < c.radius * 6:
                # A short-range motor reflex: turn the abdomen toward a sensed mate.
                target = math.atan2(b.y - mate.y, b.x - mate.x)
                delta = wrap(target - b.angle)
                b.angle = wrap(b.angle + max(-c.turn_speed * c.dt, min(c.turn_speed * c.dt, delta)))
                if abs(delta) < 0.4:
                    velocity = -c.speed * 0.3
        dx, dy = math.cos(b.angle) * velocity * c.dt, math.sin(b.angle) * velocity * c.dt
        b.x += dx
        b.y += dy
        b.distance += math.hypot(dx, dy)

    def _collisions(self):
        c = self.config
        a, b = self.insects
        for bug in self.insects:
            x = max(c.radius, min(c.width - c.radius, bug.x))
            y = max(c.radius, min(c.height - c.radius, bug.y))
            if (x, y) != (bug.x, bug.y):
                bug.collision = True
            bug.x, bug.y = x, y
        dx, dy = b.x - a.x, b.y - a.y
        d = math.hypot(dx, dy)
        if d >= c.radius * 2:
            return
        nx, ny = (dx / d, dy / d) if d > 1e-9 else (1.0, 0.0)
        overlap = c.radius * 2 - d

        def room(bug, vx, vy):
            limits = [overlap]
            if abs(vx) > 1e-9:
                wall = c.width - c.radius if vx > 0 else c.radius
                limits.append(max(0, (wall - bug.x) / vx))
            if abs(vy) > 1e-9:
                wall = c.height - c.radius if vy > 0 else c.radius
                limits.append(max(0, (wall - bug.y) / vy))
            return min(limits)

        # Give blocked displacement to the other circle, including at corners.
        room_a, room_b = room(a, -nx, -ny), room(b, nx, ny)
        move_a = min(overlap / 2, room_a)
        move_b = min(overlap - move_a, room_b)
        move_a += min(overlap - move_a - move_b, room_a - move_a)
        a.x -= nx * move_a
        a.y -= ny * move_a
        b.x += nx * move_b
        b.y += ny * move_b
        a.collision = b.collision = True

    def abdomen_tip(self, b: Insect) -> tuple[float, float]:
        r = self.config.radius
        return b.x - math.cos(b.angle) * r, b.y - math.sin(b.angle) * r

    def _mate(self):
        a, b = self.insects
        now = self.now
        if self._mating_active:
            if now >= a.mating_until:
                self._mating_active = False
                self.pairings += 1
                self.eggs += 8
                x = (a.x + b.x) / 2
                y = (a.y + b.y) / 2
                self.egg_sites.append((x, y))
                self.egg_sites = self.egg_sites[-30:]
                for i, bug in enumerate(self.insects):
                    bug.mating_count += 1
                    bug.cooldown_until = now + self.config.mating_cooldown
                    self._reward(i, self.config.mating_reward)
                self.log("Abdomen teması tamamlandı · +8 yumurta")
            return
        if not (a.fertile(now) and b.fertile(now)):
            return
        if any(bug.cooldown_until > now for bug in (a, b)):
            return
        if not any(bug.action == Action.COURT for bug in (a, b)):
            return
        ax, ay = self.abdomen_tip(a)
        bx, by = self.abdomen_tip(b)
        if math.hypot(ax - bx, ay - by) < 8.0 and abs(wrap(a.angle - b.angle)) > 2.5:
            # Duration must fit inside both animals' remaining fed period.
            if min(a.fertile_until, b.fertile_until) < now + self.config.mating_seconds:
                return
            for bug in (a, b):
                bug.mating_until = now + self.config.mating_seconds
            self._mating_active = True
            self.log("Abdomen uçları birleşti · çiftleşme")

    def tick(self):
        if self.done:
            return
        c = self.config
        self._move_food()
        # Measure both potentials against the same food positions. A moving worm
        # cannot reward an idle insect merely by approaching it.
        before = [self.scent_at(b.x, b.y) for b in self.insects]
        for bug in self.insects:
            self._move(bug)
        self._collisions()
        after = [self.scent_at(b.x, b.y) for b in self.insects]
        for i, bug in enumerate(self.insects):
            # Local odor-potential shaping, never a hidden target direction.
            self._reward(i, 3.0 * (after[i] - before[i]))
            self._reward(i, -0.001 / c.decision_frames)
            if bug.collision:
                self._reward(i, -0.06 / c.decision_frames)
        # Food belongs to whoever reaches it first, including an already fed bug.
        # Choose the nearest circle if both touch in one tick; alternate exact ties.
        for food in list(self.food):
            distances = [math.hypot(b.x - food.x, b.y - food.y) for b in self.insects]
            i = min(range(2), key=lambda j: (distances[j], (j - self.frame) % 2))
            if distances[i] < c.radius + FOOD_RADIUS:
                bug = self.insects[i]
                self.food.remove(food)
                bug.eaten += 1
                bug.ate_day = self.day
                bug.fertile_until = self.now + c.meal_seconds
                self._reward(i, 20.0)
                self.log(f"{'Dişi' if i == 0 else 'Erkek'} beslendi · +20")
        self._mate()
        self.frame += 1
        if self.frame % c.day_frames == 0:
            for i, bug in enumerate(self.insects):
                if bug.ate_day != self.day:
                    bug.missed += 1
                    self._reward(i, -8.0)
                    self.log(f"{'Dişi' if i == 0 else 'Erkek'} öğünü kaçırdı · −8")
            self.day += 1
            if not self.done:
                self._spawn_food()

    def _wall_sensor(self, b: Insect, offset: float) -> float:
        c = self.config
        dx, dy = math.cos(b.angle + offset), math.sin(b.angle + offset)
        distances = [120.0]
        if abs(dx) > 1e-8:
            distances.append(((c.width - c.radius if dx > 0 else c.radius) - b.x) / dx)
        if abs(dy) > 1e-8:
            distances.append(((c.height - c.radius if dy > 0 else c.radius) - b.y) / dy)
        return 1.0 - max(0.0, min(distances)) / 120.0

    def observe(self, index: int) -> np.ndarray:
        b, mate = self.insects[index], self.insects[1 - index]
        now = self.now
        distance = math.hypot(mate.x - b.x, mate.y - b.y)
        bearing = wrap(math.atan2(mate.y - b.y, mate.x - b.x) - b.angle)
        strength = max(0.0, 1.0 - distance / 180.0)
        mate_sensors = [
            strength * max(0.0, math.cos(bearing - offset)) ** 4
            for offset in (-math.pi / 2, 0.0, math.pi / 2, math.pi)
        ]
        phase = TAU * (self.frame % self.config.day_frames) / self.config.day_frames
        left, front, right = b.scent
        lateral = max(-1.0, min(1.0, (right - left) * 10))
        ahead = max(-1.0, min(1.0, (front - (left + right) / 2) * 20))
        state = [
            *b.scent,
            b.scent_change,
            lateral,
            ahead,
            math.exp(-(now - b.sample_time) / 2),
            *(self._wall_sensor(b, offset) for offset in (-0.8, 0.0, 0.8)),
            *mate_sensors,
            max(0.0, b.fertile_until - now) / self.config.day_seconds,
            float(mate.fertile(now)) if strength > 0 else 0.0,
            float(b.collision),
            float(b.cooldown_until > now),
            math.sin(phase),
            math.cos(phase),
            *(float(b.action == action) for action in Action),
        ]
        return np.asarray(state, dtype=np.float32)

    def observations(self) -> list[np.ndarray]:
        return [self.observe(i) for i in range(2)]

    def step(self, actions):
        self.begin_actions(actions)
        for _ in range(self.config.decision_frames):
            if self.done:
                break
            self.tick()
        return self.finish_actions()

    def finish_actions(self):
        """End a motor decision; shared by the headless runner and live viewer."""
        # Antennae passively notice nearby food; sniffing increases the range.
        for bug in self.insects:
            self._sample(bug, active=bug.action == Action.SNIFF)
        return self.observations(), np.array(self.rewards, dtype=np.float32), self.done

    def summary(self) -> dict:
        return {
            "days": self.day,
            "food_spawned": self.food_spawned,
            "eaten": sum(b.eaten for b in self.insects),
            "missed": sum(b.missed for b in self.insects),
            "pairings": self.pairings,
            "eggs": self.eggs,
            "score": sum(b.score for b in self.insects),
        }
