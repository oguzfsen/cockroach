"""Pygame observer with opt-in live learning and shared headless physics."""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

from cockroach import checkpoint
from cockroach.brain import Brain
from cockroach.config import WorldConfig
from cockroach.training import random_action
from cockroach.world import ACTION_NAMES, Action, Insect, World

BG = (13, 17, 19)
PANEL = (20, 26, 28)
EDGE = (43, 53, 53)
TEXT = (226, 231, 217)
MUTED = (135, 151, 145)
GREEN = (183, 218, 130)
GOLD = (227, 177, 107)
BLUE = (132, 191, 210)
TEAL = (90, 186, 155)
ARENA = (52, 168, 960, 640)
SIZE = (1440, 960)
NAMES = ("ADA", "ATLAS")


def run(
    *,
    model: Path | None,
    seed: int,
    frames: int | None = None,
    screenshot: Path | None = None,
    headless: bool = False,
    learning: bool = False,
    output: Path = Path("runs/live.npz"),
    natural: bool = False,
    episode_days: int | None = None,
):
    try:
        import pygame
    except ImportError as error:
        raise ValueError("Görüntüleyici için: pip install -e '.[viewer]'") from error
    pygame.display.init()
    pygame.font.init()
    try:
        app = Viewer(
            pygame,
            model,
            seed,
            learning=learning,
            output=output,
            natural=natural,
            episode_days=episode_days,
        )
        try:
            app.loop(frames, screenshot, headless)
        except KeyboardInterrupt:
            pass
    finally:
        pygame.quit()


class Viewer:
    def __init__(
        self,
        pygame,
        model: Path | None,
        seed: int,
        *,
        learning: bool = False,
        output: Path = Path("runs/live.npz"),
        natural: bool = False,
        episode_days: int | None = None,
    ):
        self.pg = pygame
        self.model = model
        self.seed = seed
        self.output = output
        self.episode_days_override = episode_days
        self.brains, config, self.training = (
            checkpoint.load(model, training=learning) if model else (None, WorldConfig(), {})
        )
        for key in ("decisions", "episodes"):
            self.training.setdefault(key, 0)
        if episode_days is not None:
            config = replace(config, episode_days=episode_days)
        self.learning = learning
        if learning and self.brains is None:
            self.brains = [Brain(seed * 2 + i) for i in range(2)]
        self.live_dirty = False
        self.live_elapsed = 0.0
        self.learning_started = time.monotonic() if learning else None
        self.last_save = time.monotonic()
        self.decision_learning = False
        self.demo_world = False
        self.world = World(config, seed)
        self.rng = np.random.default_rng(seed + 555)
        self.screen = pygame.display.set_mode(SIZE, pygame.RESIZABLE)
        pygame.display.set_caption("Cockroach Lab · Davranış gözlemevi")
        self.canvas = pygame.Surface(SIZE)
        self.fonts = {
            size: pygame.font.SysFont("DejaVu Sans", size)
            for size in (11, 12, 13, 14, 16, 18, 22, 30, 42)
        }
        self.mono = {
            size: pygame.font.SysFont("DejaVu Sans Mono", size) for size in (11, 12, 14, 18, 26)
        }
        self.selected = 0
        self.manual = False
        self.pending_action = None
        self.courtship_demo = False
        self.paused = False
        self.show_scent = True
        self.show_circles = False
        self.show_trails = True
        self.show_annotations = not natural
        self.speeds = [0.5, 1, 2, 4, 8]
        self.speed_index = 1
        self.accumulator = 0.0
        self.decision_frame = 0
        self.trails = [deque(maxlen=240), deque(maxlen=240)]
        self.buttons = []
        self.toast = ""
        self.toast_until = 0.0
        self.texture = self._make_texture()
        self.scent_key = None
        self.scent_surface = None
        self._choose_actions()

    def _make_texture(self):
        pg = self.pg
        surface = pg.Surface(ARENA[2:])
        surface.fill((28, 31, 27))
        rng = random.Random(49)
        for _ in range(8500):
            x, y = rng.randrange(960), rng.randrange(640)
            value = rng.randrange(25, 42)
            pg.draw.circle(
                surface, (value + 3, value + 2, value - 2), (x, y), rng.choice((1, 1, 2))
            )
        # Decoration stays outside the physical habitat: no invisible obstacles.
        return surface

    def text(self, text, pos, size=14, color=TEXT, mono=False, target=None):
        font = (self.mono if mono else self.fonts)[size]
        surface = font.render(str(text), True, color)
        (target or self.canvas).blit(surface, pos)
        return surface.get_width()

    def panel(self, rect, color=PANEL):
        self.pg.draw.rect(self.canvas, color, rect, border_radius=12)
        self.pg.draw.rect(self.canvas, EDGE, rect, 1, border_radius=12)

    def pill(self, label, rect, active=False, callback=None, color=GREEN):
        pg = self.pg
        pg.draw.rect(self.canvas, (40, 50, 40) if active else PANEL, rect, border_radius=6)
        pg.draw.rect(self.canvas, (85, 104, 66) if active else EDGE, rect, 1, border_radius=6)
        self.text(label, (rect[0] + 12, rect[1] + 7), 12, color if active else MUTED)
        if callback:
            self.buttons.append((pg.Rect(rect), callback))

    def point(self, x, y):
        c = self.world.config
        return ARENA[0] + x / c.width * ARENA[2], ARENA[1] + y / c.height * ARENA[3]

    def _choose_actions(self):
        observations = self.world.observations()
        self.decision_learning = self.learning and not self.manual and not self.demo_world
        actions = [
            self.brains[i].act(observations[i], explore=self.decision_learning)
            if self.brains
            else random_action(self.rng)
            for i in range(2)
        ]
        if self.courtship_demo:
            if self.world.pairings == 0:
                actions = [Action.COURT, Action.COURT]
            else:
                self.courtship_demo = False
        if self.manual:
            keys = self.pg.key.get_pressed()
            action = Action.REST
            for codes, candidate in (
                ((self.pg.K_w, self.pg.K_UP), Action.WALK),
                ((self.pg.K_s, self.pg.K_DOWN), Action.BACK),
                ((self.pg.K_a, self.pg.K_LEFT), Action.LEFT),
                ((self.pg.K_d, self.pg.K_RIGHT), Action.RIGHT),
                ((self.pg.K_e,), Action.SNIFF),
                ((self.pg.K_r,), Action.COURT),
            ):
                if any(keys[key] for key in codes):
                    action = candidate
            actions[self.selected] = (
                self.pending_action if self.pending_action is not None else action
            )
            self.pending_action = None
        self.decision_states = observations
        self.decision_actions = actions
        self.world.begin_actions(actions)

    def toggle(self, name):
        setattr(self, name, not getattr(self, name))

    def toggle_manual(self):
        self.manual = not self.manual
        # A transition touched by manual intervention is never replayed.
        self.decision_learning = False

    def toggle_learning(self):
        if self.learning:
            self.live_elapsed += time.monotonic() - self.learning_started
            self.learning_started = None
            self.learning = False
            self.decision_learning = False
            self.save_live()
            self._notify("Canlı öğrenme kapalı")
        else:
            if self.brains is None:
                self.brains = [Brain(self.seed * 2 + i) for i in range(2)]
            else:
                for brain in self.brains:
                    brain.enable_learning()
            self.learning = True
            self.learning_started = time.monotonic()
            self._notify(f"Canlı öğrenme açık · kayıt: {self.output}")

    def save_live(self):
        if not self.live_dirty:
            return
        elapsed = self.live_elapsed
        if self.learning_started is not None:
            elapsed += time.monotonic() - self.learning_started
        state = {
            **self.training,
            "decisions": self.training.get("decisions", 0),
            "episodes": self.training.get("episodes", 0),
            "elapsed_seconds": self.training.get("elapsed_seconds", 0.0) + elapsed,
            "seed": self.training.get("seed", self.seed),
            "source": str(self.model) if self.model else None,
            "mode": "live",
            "best_validation": None,
        }
        checkpoint.save(self.output, self.brains, self.world.config, state)
        self.live_dirty = False
        self.last_save = time.monotonic()

    def _notify(self, text):
        self.toast = text
        self.toast_until = time.monotonic() + 3

    def reset(self):
        self.seed += 1
        self.world = World(self.world.config, self.seed)
        self.trails = [deque(maxlen=240), deque(maxlen=240)]
        self.accumulator = 0
        self.decision_frame = 0
        self.courtship_demo = False
        self.demo_world = False
        self._choose_actions()

    def mating_demo(self):
        """An explicitly labelled observation fixture, never used for training."""
        self.reset()
        c = self.world.config
        for i, bug in enumerate(self.world.insects):
            bug.x = c.width / 2 + (-1 if i == 0 else 1) * (c.radius + 12)
            bug.y = c.height / 2
            bug.angle = math.pi if i == 0 else 0
            bug.fertile_until = c.meal_seconds
            bug.ate_day = 0
        self.manual = False
        self.courtship_demo = True
        self.demo_world = True
        self._choose_actions()
        self.world.log("Üreme deneyi · beslenmiş çift")
        self._notify("Üreme deneyi: beslenmiş çift, kur eylemi. Öğrenme yapılmaz.")

    def reload(self):
        if self.model is None:
            self._notify("Model yok. Önce terminalde cockroach train çalıştır.")
            return
        try:
            self.save_live()
            brains, config, training = checkpoint.load(self.model, training=self.learning)
        except (ValueError, OSError) as error:
            self._notify(f"Model açılamadı: {error}")
            return
        if self.episode_days_override is not None:
            config = replace(config, episode_days=self.episode_days_override)
        self.brains, self.training = brains, training
        self.live_elapsed = 0.0
        self.learning_started = time.monotonic() if self.learning else None
        self.world = World(config, self.seed)
        self.trails = [deque(maxlen=240), deque(maxlen=240)]
        self.decision_frame = 0
        self.accumulator = 0
        self.courtship_demo = False
        self.demo_world = False
        self._choose_actions()
        self._notify("Model yenilendi · yeni gözlem bölümü")

    def _mouse_position(self, pos):
        w, h = self.screen.get_size()
        scale = min(w / SIZE[0], h / SIZE[1])
        return (
            (pos[0] - (w - SIZE[0] * scale) / 2) / scale,
            (pos[1] - (h - SIZE[1] * scale) / 2) / scale,
        )

    def handle_events(self):
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
            if event.type == pg.KEYDOWN:
                key = event.key
                if key == pg.K_ESCAPE:
                    return False
                if key == pg.K_SPACE:
                    self.paused = not self.paused
                elif key == pg.K_TAB:
                    self.selected = 1 - self.selected
                elif key == pg.K_m:
                    self.toggle_manual()
                elif key == pg.K_o:
                    self.toggle_learning()
                elif key == pg.K_F1:
                    self.show_annotations = not self.show_annotations
                elif key == pg.K_h:
                    self.show_scent = not self.show_scent
                elif key == pg.K_c:
                    self.show_circles = not self.show_circles
                elif key == pg.K_t:
                    self.show_trails = not self.show_trails
                elif key == pg.K_n:
                    self.reset()
                elif key == pg.K_l:
                    self.reload()
                elif key == pg.K_F2:
                    self.mating_demo()
                elif key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                    self.speed_index = min(len(self.speeds) - 1, self.speed_index + 1)
                elif key in (pg.K_MINUS, pg.K_KP_MINUS):
                    self.speed_index = max(0, self.speed_index - 1)
                elif key == pg.K_p:
                    path = Path("runs/screenshots") / f"habitat-{time.time_ns()}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pg.image.save(self.canvas, path)
                    self._notify(f"Kaydedildi: {path}")
                elif self.manual:
                    mapping = {
                        pg.K_w: Action.WALK,
                        pg.K_UP: Action.WALK,
                        pg.K_s: Action.BACK,
                        pg.K_DOWN: Action.BACK,
                        pg.K_a: Action.LEFT,
                        pg.K_LEFT: Action.LEFT,
                        pg.K_d: Action.RIGHT,
                        pg.K_RIGHT: Action.RIGHT,
                        pg.K_e: Action.SNIFF,
                        pg.K_r: Action.COURT,
                    }
                    if key in mapping:
                        self.pending_action = mapping[key]
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                pos = self._mouse_position(event.pos)
                for rect, callback in self.buttons:
                    if rect.collidepoint(pos):
                        callback()
                        break
                else:
                    for i, bug in enumerate(self.world.insects):
                        bx, by = self.point(bug.x, bug.y)
                        if math.hypot(pos[0] - bx, pos[1] - by) < 45:
                            self.selected = i
                            break
        return True

    def _tick(self):
        if self.world.done:
            self.reset()
            self._notify("Bölüm tamamlandı · yeni başlangıç")
        self.world.tick()
        self.decision_frame += 1
        if self.world.frame % 4 == 0:
            for trail, bug in zip(self.trails, self.world.insects, strict=False):
                trail.append(self.point(bug.x, bug.y))
        if self.decision_frame >= self.world.config.decision_frames or self.world.done:
            observations, rewards, done = self.world.finish_actions()
            if self.decision_learning:
                for i, brain in enumerate(self.brains):
                    brain.remember(
                        self.decision_states[i],
                        self.decision_actions[i],
                        rewards[i],
                        observations[i],
                        done,
                    )
                self.training["decisions"] = self.training.get("decisions", 0) + 1
                if done:
                    self.training["episodes"] = self.training.get("episodes", 0) + 1
                self.live_dirty = True
            self.decision_frame = 0
            if not self.world.done:
                self._choose_actions()

    def _odor(self):
        pg = self.pg
        c = self.world.config
        key = (
            tuple((f.x, f.y) for f in self.world.food),
            c.width,
            c.height,
            c.sniff_range,
            c.scent_scale,
        )
        if key != self.scent_key:
            self.scent_key = key
            xs = np.linspace(0, c.width, 160)[:, None]
            ys = np.linspace(0, c.height, 108)[None, :]
            values = np.zeros((160, 108))
            for food in self.world.food:
                d = np.hypot(xs - food.x, ys - food.y)
                values += np.exp(-d / c.scent_scale) * np.maximum(0, 1 - (d / c.sniff_range) ** 2)
            layer = pg.Surface((160, 108), pg.SRCALPHA)
            layer.fill((*TEAL, 0))
            alpha = pg.surfarray.pixels_alpha(layer)
            alpha[:] = (np.minimum(values, 1) ** 0.65 * 75).astype(np.uint8)
            del alpha
            self.scent_surface = pg.transform.smoothscale(layer, ARENA[2:])
        self.canvas.blit(self.scent_surface, ARENA[:2])

    def _draw_food(self, food):
        """A segmented, wriggling larva; its actual velocity comes from physics."""
        pg = self.pg
        heading = math.atan2(food.vy, food.vx)
        ux, uy = math.cos(heading), math.sin(heading)
        scale = ARENA[2] / self.world.config.width
        for segment in range(6):
            offset = (segment - 2.5) * 3
            wiggle = math.sin(self.world.now * 9 - segment * 0.85) * 1.4
            x, y = self.point(
                food.x + ux * offset - uy * wiggle, food.y + uy * offset + ux * wiggle
            )
            radius = (3.4 if 0 < segment < 5 else 2.5) * scale
            pg.draw.circle(self.canvas, (17, 22, 17), (x, y + 2 * scale), radius + scale)
            pg.draw.circle(self.canvas, (164, 155, 108), (x, y), radius)
            pg.draw.circle(
                self.canvas, (218, 210, 161), (x - 0.6 * scale, y - scale), max(1, radius - scale)
            )
        if self.show_annotations:
            x, y = self.point(food.x, food.y)
            pulse = 0.5 + math.sin(self.world.now * 3 + food.x) * 0.5
            pg.draw.circle(self.canvas, (67, 110, 74), (x, y), 15 + pulse * 3, 1)
            self.text("KURTÇUK", (x + 18, y - 8), 11, GREEN)

    def _insect_sprite(self, bug: Insect, index: int):
        pg = self.pg
        scale = 2
        sprite = pg.Surface((160 * scale, 160 * scale), pg.SRCALPHA)
        color = GOLD if index == 0 else BLUE
        center = 80

        def point(x, y):
            return round((x + center) * scale), round((y + center) * scale)

        def line(points, col, width=1):
            pg.draw.lines(
                sprite, col, False, [point(x, y) for x, y in points], max(1, round(width * scale))
            )

        def ellipse(rect, col, width=0):
            x, y, w, h = rect
            pg.draw.ellipse(
                sprite,
                col,
                ((x + center) * scale, (y + center) * scale, w * scale, h * scale),
                width * scale,
            )

        moving = bug.action in (Action.WALK, Action.BACK, Action.COURT, Action.LEFT, Action.RIGHT)
        phase = bug.distance * 0.3
        if bug.action in (Action.LEFT, Action.RIGHT):
            phase = self.world.now * 12
        for side in (-1, 1):
            for leg, anchor in enumerate((-10, 0, 10)):
                swing = math.sin(phase + (leg % 2 + (side == 1)) * math.pi) if moving else 0
                knee = (anchor + (leg - 1) * 5 + swing * 3, side * (24 + 2 * math.cos(phase)))
                foot = (anchor + (leg - 1) * 13 - swing * 6, side * (35 + swing * 2))
                line(
                    [(anchor, side * 10), knee, foot, (foot[0] - 3, foot[1] + side * 3)],
                    (71, 57, 42),
                    3,
                )
                line([(anchor, side * 10), knee, foot], (160, 125, 80), 1.2)
                pg.draw.circle(sprite, color, point(*knee), 2)
        female = index == 0
        ellipse((-20, -18 if female else -15, 34, 36 if female else 30), (62, 43, 29))
        ellipse((-20, -18 if female else -15, 34, 36 if female else 30), (176, 126, 67), 1)
        for segment in range(5):
            x = -16 + segment * 4
            half = (15 if female else 12) * math.sqrt(max(0.0, 1 - ((x + 3) / 18) ** 2))
            line([(x - 2, -half), (x, 0), (x - 2, half)], (111, 77, 43), 1)
        if not female:
            ellipse((-14, -12, 25, 24), (83, 65, 42))
            line([(-12, 0), (9, 0)], (153, 128, 81), 1)
            line([(-10, -7), (8, -3)], (107, 92, 63), 1)
            line([(-10, 7), (8, 3)], (107, 92, 63), 1)
        ellipse((7, -9, 17, 18), (40, 34, 26))
        ellipse((10, -8, 13, 16), (114, 79, 40), 1)
        ellipse((1, -13, 18, 26), (99, 66, 35))
        ellipse((1, -13, 18, 26), (191, 147, 87), 1)
        ellipse((4, -9, 10, 18), (121, 83, 43))
        for side in (-1, 1):
            ellipse((16, side * 6 - 2, 4, 4), (196, 202, 161))
            sniffing = bug.action == Action.SNIFF
            sway = math.sin(self.world.now * (9 if sniffing else 3) + side) * (8 if sniffing else 3)
            points = [
                (22, side * 5),
                (31, side * 11),
                (43, side * (18 + sway)),
                (54 if sniffing else 49, side * (21 + sway)),
            ]
            # Cubic Bézier antennae, articulated independently of the legs.
            curve = []
            for step in range(17):
                t = step / 16
                weights = ((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t**3)
                curve.append(
                    tuple(
                        sum(w * p[axis] for w, p in zip(weights, points, strict=False))
                        for axis in (0, 1)
                    )
                )
            line(curve, (195, 165, 112), 1)
            pg.draw.circle(sprite, color if sniffing else (187, 167, 124), point(*curve[-1]), 3)
            line([(-18, side * 4), (-24, side * 7)], (182, 142, 84), 1)
        if bug.mating(self.world.now):
            pg.draw.circle(sprite, GOLD, point(-20, 0), 4 * scale)
        angle = -math.degrees(bug.angle)
        sprite = pg.transform.rotozoom(sprite, angle, 1 / scale)
        c = self.world.config
        world_scale = min(ARENA[2] / c.width, ARENA[3] / c.height)
        if abs(world_scale - 1) > 0.001:
            sprite = pg.transform.smoothscale_by(sprite, world_scale)
        return sprite

    def _draw_arena(self):
        pg = self.pg
        world = self.world
        self.panel((32, 148, 1000, 680))
        self.canvas.blit(self.texture, ARENA[:2])
        if self.show_annotations:
            for x in range(0, 961, 80):
                pg.draw.line(
                    self.canvas,
                    (40, 44, 37),
                    (ARENA[0] + x, ARENA[1]),
                    (ARENA[0] + x, ARENA[1] + ARENA[3]),
                )
            for y in range(0, 641, 80):
                pg.draw.line(
                    self.canvas,
                    (40, 44, 37),
                    (ARENA[0], ARENA[1] + y),
                    (ARENA[0] + ARENA[2], ARENA[1] + y),
                )
        if self.show_annotations and self.show_scent:
            self._odor()
        self.canvas.set_clip(pg.Rect(ARENA))
        for x, y in world.egg_sites:
            cx, cy = self.point(x, y)
            for k in range(4):
                pg.draw.ellipse(self.canvas, (157, 133, 85), (cx + k * 4 - 8, cy + k % 2 * 3, 4, 7))
        if self.show_annotations and self.show_trails:
            for i, trail in enumerate(self.trails):
                points = list(trail)
                for j in range(1, len(points)):
                    alpha = j / len(points) * 0.25
                    color = GOLD if i == 0 else BLUE
                    muted = tuple(round(30 + (c - 30) * alpha) for c in color)
                    pg.draw.line(self.canvas, muted, points[j - 1], points[j], 2)
        for food in world.food:
            self._draw_food(food)
        for i, bug in enumerate(world.insects):
            x, y = self.point(bug.x, bug.y)
            color = GOLD if i == 0 else BLUE
            radius = world.config.radius * ARENA[2] / world.config.width
            pg.draw.ellipse(
                self.canvas,
                (19, 22, 19),
                (x - radius - 2, y - radius + 4, radius * 2 + 4, radius * 2),
            )
            if self.show_annotations and self.show_circles:
                pg.draw.circle(
                    self.canvas, (219, 117, 98) if bug.collision else color, (x, y), radius, 1
                )
                tip = self.point(*world.abdomen_tip(bug))
                pg.draw.circle(self.canvas, GOLD, tip, 3)
            if self.show_annotations and i == self.selected:
                for offset in (-math.pi * 0.25, math.pi * 0.75):
                    pg.draw.arc(
                        self.canvas, color, (x - 48, y - 48, 96, 96), offset, offset + 0.7, 2
                    )
            if self.show_annotations and bug.action == Action.SNIFF:
                for offset in (-0.65, 0, 0.65):
                    tip = self.point(
                        bug.x + math.cos(bug.angle + offset) * 65,
                        bug.y + math.sin(bug.angle + offset) * 65,
                    )
                    pg.draw.line(self.canvas, (71, 104, 86), (x, y), tip, 1)
                    pg.draw.circle(self.canvas, TEAL, tip, 3, 1)
            sprite = self._insect_sprite(bug, i)
            self.canvas.blit(sprite, sprite.get_rect(center=(x, y)))
            if self.show_annotations:
                self.text(f"{NAMES[i]}  {'♀' if i == 0 else '♂'}", (x - 30, y - 64), 11, color)
        self.canvas.set_clip(None)
        pg.draw.rect(self.canvas, (75, 81, 62), ARENA, 1)
        if self.show_annotations:
            for value in range(0, 961, 160):
                self.text(str(value), (ARENA[0] + value - 5, 811), 11, MUTED, mono=True)
        if self.paused and self.show_annotations:
            self.panel((379, 442, 306, 92))
            self.text("DURAKLATILDI", (425, 459), 22)
            self.text("Devam etmek için SPACE", (434, 496), 12, MUTED)

    def _draw_card(self, index, y):
        pg = self.pg
        bug = self.world.insects[index]
        color = GOLD if index == 0 else BLUE
        self.panel((1060, y, 348, 208))
        pg.draw.circle(self.canvas, color, (1085, y + 28), 4)
        self.text(NAMES[index], (1099, y + 15), 22, color)
        self.text("DİŞİ / ♀" if index == 0 else "ERKEK / ♂", (1287, y + 22), 12, MUTED)
        if index == self.selected:
            self.text("ELLE KONTROL" if self.manual else "SEÇİLİ", (1080, y + 50), 11, color)
        mode = "Çiftleşiyor" if bug.mating(self.world.now) else ACTION_NAMES[bug.action]
        self.text(mode, (1080, y + 73), 18)
        self.text(f"{bug.score:+.1f}", (1285, y + 70), 18, color, mono=True)
        self.text("SKOR", (1340, y + 97), 11, MUTED)
        fertility = max(0, bug.fertile_until - self.world.now)
        self.text("ÜREME ENERJİSİ", (1080, y + 112), 11, MUTED)
        self.text(
            f"{fertility:.0f} s" if fertility > 0 else "Besin aranıyor",
            (1270, y + 112),
            11,
            color if fertility > 0 else MUTED,
        )
        pg.draw.rect(self.canvas, (43, 48, 39), (1080, y + 136, 308, 4), border_radius=2)
        width = min(1, fertility / self.world.config.meal_seconds) * 308
        if width > 0:
            pg.draw.rect(self.canvas, color, (1080, y + 136, width, 4), border_radius=2)
        self.text(
            f"Besin  {bug.eaten:02d}    Kaçan  {bug.missed:02d}",
            (1080, y + 151),
            12,
            MUTED,
            mono=True,
        )
        self.text("ANTEN", (1080, y + 180), 11, MUTED)
        for j, value in enumerate(bug.scent):
            x = 1140 + j * 82
            pg.draw.rect(self.canvas, (42, 51, 44), (x, y + 185, 60, 4), border_radius=2)
            if value > 0:
                pg.draw.rect(
                    self.canvas, TEAL, (x, y + 185, max(1, value * 60), 4), border_radius=2
                )

    def draw(self):
        pg = self.pg
        world = self.world
        self.canvas.fill(BG)
        self.buttons = []
        self.text("C / L", (32, 22), 14, GREEN, mono=True)
        self.text("DAVRANIŞ GÖZLEMEVİ", (104, 23), 12, MUTED)
        self.text("Cockroach Lab", (30, 46), 42)
        self.text("Küçük bir dünya. İki bağımsız zeka.", (405, 66), 14, MUTED)
        self.pill(
            "DURDU" if self.paused else "CANLI",
            (1198, 26, 91, 30),
            not self.paused,
            lambda: self.toggle("paused"),
        )
        if self.learning and (self.manual or self.demo_world):
            learning_label = "ÖĞRENME ASKIDA [O]"
        else:
            learning_label = "ÖĞRENME AÇIK [O]" if self.learning else "ÖĞRENME KAPALI [O]"
        self.pill(
            learning_label, (1195, 66, 213, 30), self.learning, self.toggle_learning, color=TEAL
        )
        self.pill(
            "İşaretler [F1]",
            (778, 66, 254, 30),
            self.show_annotations,
            lambda: self.toggle("show_annotations"),
        )
        self.text(f"{self.speeds[self.speed_index]:g}×", (1320, 29), 18, GREEN, mono=True)
        self.text("HABİTAT / 01", (32, 113), 12, MUTED, mono=True)
        self.pill(
            "Koku [H]", (225, 107, 112, 30), self.show_scent, lambda: self.toggle("show_scent")
        )
        self.pill(
            "Çember [C]",
            (347, 107, 125, 30),
            self.show_circles,
            lambda: self.toggle("show_circles"),
        )
        self.pill(
            "İzler [T]", (482, 107, 111, 30), self.show_trails, lambda: self.toggle("show_trails")
        )
        self.pill(
            "Üreme deneyi [F2]",
            (607, 107, 242, 30),
            self.courtship_demo,
            self.mating_demo,
            color=GOLD,
        )
        self.pill(
            "ELLE [M]" if self.manual else "OTONOM [M]",
            (870, 107, 162, 30),
            self.manual,
            self.toggle_manual,
        )
        self._draw_arena()
        self.panel((1060, 148, 348, 146))
        self.text(
            f"GÜN {min(world.day + 1, world.config.episode_days):02d}",
            (1080, 167),
            12,
            GREEN,
            mono=True,
        )
        seconds = world.now % (world.config.day_frames / world.config.fps)
        day_length = world.config.day_frames / world.config.fps
        self.text(f"{seconds:04.1f} / {day_length:g} s", (1198, 162), 18, TEXT, mono=True)
        for i in range(30):
            col = GREEN if i / 30 <= seconds / day_length else (48, 58, 46)
            pg.draw.rect(self.canvas, col, (1080 + i * 10.4, 203, 7, 9), border_radius=1)
        self.text("BESİN", (1080, 233), 11, MUTED)
        self.text("YUMURTA", (1196, 233), 11, MUTED)
        self.text("ÇİFTLEŞME", (1300, 233), 11, MUTED)
        self.text(f"{len(world.food):02d}", (1080, 252), 18, GREEN, mono=True)
        self.text(f"{world.eggs:02d}", (1196, 252), 18, GOLD, mono=True)
        self.text(f"{world.pairings:02d}", (1300, 252), 18, TEXT, mono=True)
        self._draw_card(0, 312)
        self._draw_card(1, 536)
        self.panel((1060, 760, 348, 172))
        self.text("ALAN NOTLARI", (1080, 777), 11, MUTED, mono=True)
        for i, (timestamp, message) in enumerate(world.events[-4:]):
            self.text(f"{timestamp:05.1f}", (1080, 807 + i * 27), 11, TEAL, mono=True)
            self.text(message, (1134, 807 + i * 27), 11, TEXT)
        self.panel((32, 848, 1000, 84))
        self.text("W A S D  hareket    E  kokla    R  kur yap", (52, 865), 13, TEXT)
        self.text("TAB  böcek seç    SPACE  duraklat    + / −  hız", (52, 893), 12, MUTED)
        self.text("N  yeni ortam    L  modeli yenile", (656, 865), 12, MUTED)
        self.text("P  ekran görüntüsü    ESC  çıkış", (656, 893), 12, MUTED)
        source = self.model.name if self.model else "rastgele başlangıç"
        self.text(
            f"30 Hz fizik  /  {world.config.decision_seconds:.1f} s karar  /  {source}",
            (32, 940),
            11,
            MUTED,
            mono=True,
        )
        self.text("SEED " + str(self.seed), (1298, 940), 11, MUTED, mono=True)
        if time.monotonic() < self.toast_until:
            self.panel((225, 14, 807, 30))
            self.text(self.toast, (243, 21), 12, GREEN)
        width, height = self.screen.get_size()
        ratio = min(width / SIZE[0], height / SIZE[1])
        target_size = max(1, round(SIZE[0] * ratio)), max(1, round(SIZE[1] * ratio))
        self.screen.fill(BG)
        surface = (
            self.canvas
            if target_size == SIZE
            else pg.transform.smoothscale(self.canvas, target_size)
        )
        self.screen.blit(surface, ((width - target_size[0]) // 2, (height - target_size[1]) // 2))
        pg.display.flip()

    def loop(self, frames: int | None, screenshot: Path | None, headless: bool):
        try:
            self._loop(frames, screenshot, headless)
        finally:
            self.save_live()

    def _loop(self, frames: int | None, screenshot: Path | None, headless: bool):
        clock = self.pg.time.Clock()
        rendered = 0
        running = True
        while running and (frames is None or rendered < frames):
            dt = 1 / 30 if headless else min(clock.tick(30) / 1000, 0.15)
            running = self.handle_events()
            if not running:
                break
            if not self.paused:
                self.accumulator += dt * self.speeds[self.speed_index]
                while self.accumulator + 1e-10 >= self.world.config.dt:
                    self._tick()
                    self.accumulator -= self.world.config.dt
            self.draw()
            if self.live_dirty and time.monotonic() - self.last_save >= 15:
                try:
                    self.save_live()
                except OSError as error:
                    self.last_save = time.monotonic()
                    self._notify(f"Kayıt yazılamadı: {error}")
            rendered += 1
        if screenshot:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            self.pg.image.save(self.canvas, screenshot)
            print(f"Görüntü kaydedildi: {screenshot}")
