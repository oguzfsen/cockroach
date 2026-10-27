import json

import numpy as np
import pytest

from cockroach import checkpoint
from cockroach import training as training_module
from cockroach.brain import Brain, BrainConfig, Network, Replay
from cockroach.config import WorldConfig
from cockroach.training import rollout, train
from cockroach.world import OBS_SIZE, Action, World


def test_manual_backprop_matches_numerical_gradient():
    rng = np.random.default_rng(9)
    network = Network(rng, 8)
    states = rng.normal(size=(5, OBS_SIZE)).astype(np.float32)
    actions = np.array([0, 1, 2, 3, 4])
    targets = network.forward(states)[np.arange(5), actions] + np.array([0.1, -0.4, 2, -2, 0.3])
    _, grads = network.gradients(states, actions, targets)
    for key, index in (("w0", (4, 2)), ("w1", (2, 1)), ("w2", (1, 3)), ("b2", (0,))):
        old = float(network.params[key][index])
        delta = 0.002
        network.params[key][index] = old + delta
        plus, _ = network.gradients(states, actions, targets)
        network.params[key][index] = old - delta
        minus, _ = network.gradients(states, actions, targets)
        network.params[key][index] = old
        assert grads[key][index] == pytest.approx((plus - minus) / (2 * delta), abs=0.002)


def test_brain_actually_learns_a_terminal_reward():
    config = BrainConfig(
        hidden=16, batch_size=16, warmup=16, replay_size=256, learning_rate=0.003, train_every=1
    )
    brain = Brain(5, config)
    state = np.full(OBS_SIZE, 0.2, dtype=np.float32)
    old_error = abs(brain.network.forward(state)[Action.WALK] - 2)
    for _ in range(160):
        brain.remember(state, Action.WALK, 2.0, state, True)
    assert abs(brain.network.forward(state)[Action.WALK] - 2) < old_error * 0.05
    assert brain.updates > 0


def test_replay_replaces_old_experience_without_changing_original_observation():
    replay = Replay(3)
    state = np.ones(OBS_SIZE, dtype=np.float32)
    for i in range(5):
        replay.add(state, Action.WALK, i, state, False)
    state[:] = 99
    assert replay.size == 3
    assert set(replay.rewards) == {2, 3, 4}
    assert np.all(replay.states == 1)


def test_insects_have_independent_parameters_and_targets():
    a, b = Brain(1), Brain(2)
    a.network.params["w0"][:] = 42
    assert not np.any(b.network.params["w0"] == 42)
    assert not np.any(a.target.params["w0"] == 42)


def test_checkpoint_restores_weights_optimizer_and_rng_without_pickle(tmp_path):
    brains = [Brain(1), Brain(2)]
    state = World().observe(0)
    for brain in brains:
        for _ in range(260):
            brain.remember(state, 0, 1.0, state, False)
    path = tmp_path / "model.npz"
    metadata = {"episodes": 4, "decisions": 260}
    checkpoint.save(path, brains, WorldConfig(), metadata)
    loaded, config, data = checkpoint.load(path, training=True)
    assert config == WorldConfig()
    assert data == metadata
    for a, b in zip(brains, loaded, strict=True):
        assert a.steps == b.steps
        assert a.updates == b.updates
        for key in a.network.params:
            np.testing.assert_array_equal(a.network.params[key], b.network.params[key])
            np.testing.assert_array_equal(a.target.params[key], b.target.params[key])
            np.testing.assert_array_equal(a.m[key], b.m[key])
            np.testing.assert_array_equal(a.v[key], b.v[key])
        np.testing.assert_array_equal(a.rng.random(10), b.rng.random(10))
        assert b.replay.size == 0
    assert list(tmp_path.iterdir()) == [path]


def test_inference_and_evaluation_do_not_change_weights_or_saved_file(tmp_path):
    path = tmp_path / "model.npz"
    config = WorldConfig(day_seconds=1.6, episode_days=1)
    checkpoint.save(path, [Brain(1), Brain(2)], config, {})
    original = path.read_bytes()
    brains, _, _ = checkpoint.load(path)
    before = [b.network.params["w0"].copy() for b in brains]
    rollout(config, 10, brains)
    for brain, weights in zip(brains, before, strict=True):
        np.testing.assert_array_equal(brain.network.params["w0"], weights)
        with pytest.raises(RuntimeError):
            brain.remember(np.zeros(OBS_SIZE), 0, 1, np.zeros(OBS_SIZE), True)
    assert path.read_bytes() == original


def test_unknown_checkpoint_version_is_rejected(tmp_path):
    path = tmp_path / "model.npz"
    checkpoint.save(path, [Brain(1), Brain(2)], WorldConfig(), {})
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    meta = json.loads(str(arrays["metadata"]))
    meta["version"] = 900
    arrays["metadata"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="uyumlu"):
        checkpoint.load(path)


def test_legacy_checkpoint_upgrades_environment_without_losing_learned_state(tmp_path):
    path = tmp_path / "legacy.npz"
    brains = [Brain(1), Brain(2)]
    brains[0].steps = 750_000
    state = {"decisions": 750_000, "episodes": 900, "best_validation": {"mean_eaten": 9}}
    checkpoint.save(path, brains, WorldConfig(episode_days=6), state)
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    meta = json.loads(str(arrays["metadata"]))
    meta["version"] = 1
    for field in ("fertility_energy_factor", "mating_reward", "food_speed_ratio"):
        del meta["world"][field]
    arrays["metadata"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)
    original_file = path.read_bytes()
    restored, config, metadata = checkpoint.load(path, training=True)
    assert config.episode_days == 30
    assert config.meal_seconds == 50
    assert config.food_speed_ratio == 0.5
    assert config.mating_reward == 22
    assert metadata["migrated_from"] == 1
    assert metadata["best_validation"] is None
    assert restored[0].steps == 750_000
    for original, loaded in zip(brains, restored, strict=True):
        for key in original.network.params:
            np.testing.assert_array_equal(original.network.params[key], loaded.network.params[key])
            np.testing.assert_array_equal(original.target.params[key], loaded.target.params[key])
        assert original.rng.bit_generator.state == loaded.rng.bit_generator.state
    assert path.read_bytes() == original_file


def test_enabling_live_replay_preserves_weights_and_existing_replay():
    brain = Brain(2, training=False)
    weights = brain.network.params["w0"].copy()
    brain.steps = 1000
    brain.enable_learning()
    replay = brain.replay
    state = np.zeros(OBS_SIZE, dtype=np.float32)
    brain.remember(state, 0, 1, state, False)
    brain.enable_learning()
    assert brain.replay is replay
    assert brain.replay.size == 1
    assert brain.steps == 1001
    np.testing.assert_array_equal(brain.network.params["w0"], weights)


def test_short_training_and_resume_save_progress_and_csv(tmp_path):
    path = tmp_path / "model.npz"
    config = WorldConfig(day_seconds=1.6, episode_days=1)
    train(minutes=None, steps=5, seed=15, output=path, config=config)
    _, loaded_config, state = checkpoint.load(path)
    assert state["decisions"] == 5
    assert state["episodes"] == 2
    assert loaded_config == config
    train(minutes=None, steps=3, seed=999, output=path, resume=path)
    brains, _, state = checkpoint.load(path)
    assert state["decisions"] == 8
    assert state["episodes"] == 3
    assert state["seed"] == 15
    assert brains[0].steps == 8
    assert len(path.with_suffix(".csv").read_text().strip().splitlines()) == 4


def test_best_checkpoint_survives_later_policy_regression(tmp_path, monkeypatch):
    monkeypatch.setattr(training_module, "VALIDATION_EVERY", 2)

    def fake_rollout(config, seed, brains):
        meals = 2 if brains[0].steps <= 2 else 0
        return {"eaten": meals, "score": float(meals)}

    monkeypatch.setattr(training_module, "rollout", fake_rollout)
    path = tmp_path / "model.npz"
    train(
        minutes=None,
        steps=4,
        seed=5,
        output=path,
        config=WorldConfig(day_seconds=1.6, episode_days=1),
    )
    _, _, latest = checkpoint.load(path)
    _, _, best = checkpoint.load(path.with_name("model.best.npz"))
    assert latest["decisions"] == 4
    assert best["decisions"] == 2
    assert best["best_validation"]["seed"] == 1_000_005
    records = [
        json.loads(line) for line in path.with_suffix(".validation.jsonl").read_text().splitlines()
    ]
    assert [r["selected"] for r in records] == [True, False]


def test_interrupt_saves_completed_progress_without_running_validation(tmp_path, monkeypatch):
    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(World, "step", interrupt)
    path = tmp_path / "model.npz"
    train(minutes=5, steps=None, seed=0, output=path)
    _, _, state = checkpoint.load(path)
    assert state["decisions"] == 0
    assert state["episodes"] == 0


def test_failed_atomic_save_preserves_previous_model_and_removes_temporary_file(
    tmp_path, monkeypatch
):
    path = tmp_path / "model.npz"
    brains = [Brain(1), Brain(2)]
    checkpoint.save(path, brains, WorldConfig(), {})
    original = path.read_bytes()

    def fail(file, **kwargs):
        file.write(b"partial checkpoint")
        raise OSError("disk write failed")

    monkeypatch.setattr(checkpoint.np, "savez_compressed", fail)
    with pytest.raises(OSError, match="disk write"):
        checkpoint.save(path, brains, WorldConfig(), {})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
