import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import numpy as np
import pytest

from cockroach.brain import Brain
from cockroach.checkpoint import save
from cockroach.config import WorldConfig
from cockroach.viewer import SIZE, Viewer
from cockroach.world import Action

pygame = pytest.importorskip("pygame")


@pytest.fixture
def viewer():
    pygame.display.init()
    pygame.font.init()
    app = Viewer(pygame, None, 7)
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
