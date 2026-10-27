import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import numpy as np
import pytest

from cockroach.brain import Brain, BrainConfig
from cockroach.checkpoint import load, save
from cockroach.config import WorldConfig
from cockroach.viewer import ARENA, SIZE, Viewer
from cockroach.world import Action, World

pygame = pytest.importorskip("pygame")


@pytest.fixture
def viewer(tmp_path):
    pygame.display.init()
    pygame.font.init()
    app = Viewer(pygame, None, 7, output=tmp_path / "live.npz")
    yield app
    pygame.quit()


def test_all_motor_animations_draw_and_window_resizes(viewer, tmp_path):
    for action in Action:
        viewer.world.begin_actions([action, action])
        for _ in range(4):
            viewer.world.tick()
        viewer.draw()
    assert viewer.canvas.get_size() == SIZE
    viewer.screen = pygame.display.set_mode((900, 600), pygame.RESIZABLE)
    viewer.show_circles = True
    viewer.paused = True
    viewer.draw()
    path = tmp_path / "habitat.png"
    pygame.image.save(viewer.canvas, path)
    assert path.stat().st_size > 1000


def test_mating_observation_fixture_uses_actual_physics(viewer):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F2))
    viewer.handle_events()
    assert viewer.courtship_demo
    for _ in range(110):
        viewer._tick()
    assert viewer.world.pairings == 1
    assert viewer.world.eggs == 8
    assert len(viewer.world.insects) == 2
    viewer.draw()


def test_pause_selection_and_manual_keys(viewer):
    for key in (pygame.K_SPACE, pygame.K_TAB, pygame.K_m, pygame.K_e):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
    viewer.handle_events()
    assert viewer.paused
    assert viewer.selected == 1
    assert viewer.manual
    viewer._choose_actions()
    assert viewer.world.insects[1].action == Action.SNIFF
    assert viewer.pending_action is None


def test_reloading_model_does_not_train_or_modify_checkpoint(viewer, tmp_path):
    path = tmp_path / "model.npz"
    brains = [Brain(1), Brain(2)]
    save(path, brains, WorldConfig(), {})
    original = path.read_bytes()
    viewer.model = path
    viewer.reload()
    for _ in range(50):
        viewer._tick()
    viewer.draw()
    for a, b in zip(brains, viewer.brains, strict=True):
        assert b.steps == 0
        assert b.replay is None
        for key in a.network.params:
            np.testing.assert_array_equal(a.network.params[key], b.network.params[key])
    assert path.read_bytes() == original


def test_natural_habitat_hides_all_overlays_but_preserves_individual_settings(viewer, monkeypatch):
    viewer.show_circles = True
    original_settings = (viewer.show_circles, viewer.show_trails, viewer.show_scent)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1))
    viewer.handle_events()
    assert not viewer.show_annotations
    assert original_settings == (viewer.show_circles, viewer.show_trails, viewer.show_scent)
    calls = []
    monkeypatch.setattr(viewer, "text", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(viewer, "_odor", lambda: pytest.fail("Natural view must hide the odor map"))
    viewer._draw_arena()
    pixels = pygame.surfarray.array3d(viewer.canvas.subsurface(ARENA))
    viewer.selected = 1 - viewer.selected
    viewer.paused = True
    viewer.show_circles = False
    viewer._draw_arena()
    np.testing.assert_array_equal(pixels, pygame.surfarray.array3d(viewer.canvas.subsurface(ARENA)))
    assert calls == []
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1))
    viewer.handle_events()
    assert viewer.show_annotations


def test_enabling_learning_mid_decision_skips_partial_transition_and_saves_on_disable(viewer):
    for _ in range(12):
        viewer._tick()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_o))
    viewer.handle_events()
    assert viewer.learning
    for _ in range(12):
        viewer._tick()
    assert all(b.steps == 0 for b in viewer.brains)
    for _ in range(24):
        viewer._tick()
    assert all(b.steps == 1 for b in viewer.brains)
    viewer.toggle_learning()
    saved, config, state = load(viewer.output)
    assert not viewer.learning
    assert state["decisions"] == 1
    assert state["episodes"] == 0
    assert config.episode_days == 30
    assert all(b.steps == 1 for b in saved)
    for _ in range(48):
        viewer._tick()
    assert all(b.steps == 1 for b in viewer.brains)


def test_notifications_do_not_cover_natural_habitat(viewer):
    viewer.show_annotations = False
    viewer.draw()
    before = pygame.surfarray.array3d(viewer.canvas.subsurface(ARENA))
    viewer._notify("Canlı öğrenme kapalı")
    viewer.draw()
    np.testing.assert_array_equal(before, pygame.surfarray.array3d(viewer.canvas.subsurface(ARENA)))


def test_live_learning_updates_weights_and_exit_saves_to_separate_file(viewer, tmp_path):
    config = BrainConfig(hidden=8, replay_size=16, warmup=2, batch_size=2, train_every=1)
    source = tmp_path / "source.npz"
    save(source, [Brain(1, config), Brain(2, config)], WorldConfig(), {})
    original_file = source.read_bytes()
    viewer.model = source
    viewer.reload()
    initial_weights = viewer.brains[0].network.params["w0"].copy()
    viewer.toggle_learning()
    viewer._choose_actions()
    viewer.loop(frames=72, screenshot=None, headless=True)
    assert all(b.steps == 3 and b.updates == 2 for b in viewer.brains)
    assert not np.array_equal(initial_weights, viewer.brains[0].network.params["w0"])
    assert source.read_bytes() == original_file
    restored, _, state = load(viewer.output)
    assert state["decisions"] == 3
    assert state["mode"] == "live"
    for a, b in zip(restored, viewer.brains, strict=True):
        for key in a.network.params:
            np.testing.assert_array_equal(a.network.params[key], b.network.params[key])


def test_brief_manual_intervention_excludes_entire_transition(viewer):
    viewer.toggle_learning()
    viewer._choose_actions()
    for _ in range(12):
        viewer._tick()
    viewer.toggle_manual()
    viewer.toggle_manual()
    for _ in range(12):
        viewer._tick()
    assert all(b.steps == 0 for b in viewer.brains)
    for _ in range(24):
        viewer._tick()
    assert all(b.steps == 1 for b in viewer.brains)


def test_reproduction_fixture_never_trains_even_after_pairing_finishes(viewer):
    viewer.toggle_learning()
    viewer.mating_demo()
    for _ in range(144):
        viewer._tick()
    assert viewer.world.pairings == 1
    assert not viewer.courtship_demo
    assert viewer.demo_world
    assert all(b.steps == 0 for b in viewer.brains)
    viewer.reset()
    for _ in range(24):
        viewer._tick()
    assert all(b.steps == 1 for b in viewer.brains)


def test_live_learning_records_terminal_transition_and_continues_next_world(viewer):
    viewer.world = World(WorldConfig(day_seconds=0.8, episode_days=1), seed=10)
    viewer.toggle_learning()
    viewer._choose_actions()
    for _ in range(48):
        viewer._tick()
    assert viewer.training["episodes"] == 2
    assert viewer.training["decisions"] == 2
    assert all(np.all(b.replay.dones[:2] == 1) for b in viewer.brains)


def test_pausing_prevents_live_learning_and_empty_save(viewer):
    viewer.toggle_learning()
    viewer._choose_actions()
    viewer.paused = True
    viewer.loop(frames=3, screenshot=None, headless=True)
    assert all(b.steps == 0 for b in viewer.brains)
    assert not viewer.output.exists()


def test_interrupted_window_still_saves_completed_transitions(viewer, monkeypatch):
    viewer.toggle_learning()
    viewer._choose_actions()
    for _ in range(24):
        viewer._tick()

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(viewer, "_loop", interrupt)
    with pytest.raises(KeyboardInterrupt):
        viewer.loop(None, None, True)
    _, _, state = load(viewer.output)
    assert state["decisions"] == 1
