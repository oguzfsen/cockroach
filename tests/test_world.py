import math

import numpy as np
import pytest

from cockroach.config import WorldConfig
from cockroach.world import FOOD_RADIUS, OBS_NAMES, OBS_SIZE, Action, Food, World, reflect_axis


def test_seed_reproduces_world_and_trajectory():
    a, b = World(seed=41), World(seed=41)
    rng = np.random.default_rng(55)
    for _ in range(40):
        actions = rng.integers(len(Action), size=2)
        obs_a, rewards_a, done_a = a.step(actions)
        obs_b, rewards_b, done_b = b.step(actions)
        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_array_equal(rewards_a, rewards_b)
        assert done_a == done_b
    assert a.food == b.food
    assert a.summary() == b.summary()


def test_live_frames_and_headless_decisions_have_identical_physics_and_senses():
    headless, live = World(seed=15), World(seed=15)
    for actions in (
        [Action.WALK, Action.SNIFF],
        [Action.LEFT, Action.BACK],
        [Action.SNIFF, Action.RIGHT],
        [Action.WALK, Action.WALK],
    ):
        expected = headless.step(actions)
        live.begin_actions(actions)
        for _ in range(live.config.decision_frames):
            live.tick()
        actual = live.finish_actions()
        np.testing.assert_array_equal(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])
        assert headless.insects == live.insects


def test_motor_action_lasts_a_full_decision_without_jitter():
    world = World()
    a, b = world.insects
    a.x, a.y, a.angle = 100, 100, 0
    b.x, b.y = 800, 500
    world.step([Action.WALK, Action.REST])
    assert a.x == pytest.approx(100 + world.config.speed * world.config.decision_seconds)
    assert a.y == 100
    world.step([Action.BACK, Action.REST])
    assert a.x == pytest.approx(100 + world.config.speed * world.config.decision_seconds * 0.4)


@pytest.mark.parametrize("x,y,angle", [(18, 18, math.pi), (942, 622, 0), (100, 18, -math.pi / 2)])
def test_walls_contain_the_collision_circle(x, y, angle):
    world = World()
    bug = world.insects[0]
    bug.x, bug.y, bug.angle = x, y, angle
    world.step([Action.WALK, Action.REST])
    r = world.config.radius
    assert r <= bug.x <= world.config.width - r
    assert r <= bug.y <= world.config.height - r
    assert bug.collision


@pytest.mark.parametrize(
    "positions",
    [
        ((18, 18), (18, 18)),
        ((18, 18), (25, 25)),
        ((942, 622), (942, 622)),
        ((200, 200), (202, 200)),
    ],
)
def test_circles_separate_even_against_a_corner(positions):
    world = World()
    for bug, (x, y) in zip(world.insects, positions, strict=True):
        bug.x, bug.y = x, y
    world.step([Action.REST, Action.REST])
    a, b = world.insects
    assert math.hypot(a.x - b.x, a.y - b.y) >= world.config.radius * 2 - 1e-7
    for bug in world.insects:
        assert 18 - 1e-7 <= bug.x <= 942 + 1e-7
        assert 18 - 1e-7 <= bug.y <= 622 + 1e-7


def test_food_odor_is_monotone_and_finite_range():
    world = World()
    world.food = [Food(300, 300, 0)]
    values = [world.scent_at(300 + d, 300) for d in (0, 50, 100, 300, 420, 500)]
    assert values == sorted(values, reverse=True)
    assert values[0] == 1
    assert values[-2:] == [0, 0]


def test_sniff_extends_antenna_range_and_observations_are_local():
    passive, active = World(), World()
    for world in (passive, active):
        bug = world.insects[0]
        bug.x, bug.y, bug.angle = 100, 100, 0
        world.food = [Food(360, 100, 0)]
    passive.step([Action.REST, Action.REST])
    active.step([Action.SNIFF, Action.REST])
    assert sum(passive.insects[0].scent) == 0
    assert sum(active.insects[0].scent) > 0
    assert not any(name in OBS_NAMES for name in ("food_x", "food_y", "target_angle", "x", "y"))
    # Food outside any antenna's active range has no effect on observations.
    active.food = [Food(900, 600, 0)]
    active.step([Action.SNIFF, Action.REST])
    obs = active.observe(0).copy()
    active.food = [Food(850, 550, 0)]
    np.testing.assert_array_equal(active.observe(0), obs)


def test_observation_shape_dtype_and_finite_bounds():
    world = World()
    for _ in range(100):
        observations, _, _ = world.step([Action.WALK, Action.LEFT])
        for obs in observations:
            assert obs.shape == (OBS_SIZE,)
            assert obs.dtype == np.float32
            assert np.isfinite(obs).all()
            assert np.max(np.abs(obs)) <= 1.00001


def test_two_daily_foods_expire_and_missing_meals_are_penalized_once():
    config = WorldConfig(day_seconds=1.6, episode_days=3)
    world = World(config)
    world.food = [Food(900, 600, 0), Food(850, 600, 0)]
    for bug in world.insects:
        bug.x, bug.y = (100, 100) if bug.sex == "female" else (200, 100)
    old = world.food.copy()
    for _ in range(2):
        world.step([Action.REST, Action.REST])
    assert world.day == 1
    assert len(world.food) == 2
    assert world.food != old
    assert world.food_spawned == 4
    assert all(b.missed == 1 for b in world.insects)
    assert all(b.score == pytest.approx(91.998) for b in world.insects)
    while not world.done:
        world.food.clear()
        world.step([Action.REST, Action.REST])
    assert world.food_spawned == 6  # No extra meal is spawned at a terminal boundary.
    assert all(b.missed == 3 for b in world.insects)
    summary = world.summary()
    world.step([Action.WALK, Action.WALK])
    assert world.summary() == summary


def test_competitive_bug_can_take_both_daily_foods_and_receives_both_rewards():
    world = World()
    a, b = world.insects
    a.x, a.y = 200, 200
    b.x, b.y = 800, 500
    world.food = [Food(200, 200, 0), Food(202, 202, 0)]
    world.step([Action.REST, Action.REST])
    assert a.eaten == 2
    assert not world.food
    assert a.fertile_until == 50
    assert a.fertile(world.now)
    assert a.score > 139
    assert not a.fertile(50)


def test_each_meal_refills_energy_without_daily_eating_lock():
    world = World()
    a, b = world.insects
    a.x, a.y, b.x, b.y = 200, 200, 800, 500
    world.food = [Food(a.x, a.y, 0)]
    world.step([Action.REST, Action.REST])
    first_expiry = a.fertile_until
    world.food = [Food(a.x, a.y, 0)]
    world.step([Action.REST, Action.REST])
    assert a.eaten == 2
    assert a.fertile_until == pytest.approx(first_expiry + world.config.decision_seconds)
    assert a.fertile_until - 0.8 == world.config.day_seconds / 1.2


def test_nearest_bug_wins_shared_food_even_if_it_is_the_male():
    world = World()
    a, b = world.insects
    a.x, a.y, b.x, b.y = 200, 200, 236, 200
    world.food = [Food(220, 200, 0)]
    world.step([Action.REST, Action.REST])
    assert a.eaten == 0
    assert b.eaten == 1
    assert len(world.food) == 0


def test_spawned_worms_keep_half_insect_speed_and_stay_inside_walls():
    world = World(seed=14)
    for _ in range(1000):
        world._move_food()
        for food in world.food:
            assert math.hypot(food.vx, food.vy) == pytest.approx(world.config.speed / 2)
            assert FOOD_RADIUS <= food.x <= world.config.width - FOOD_RADIUS
            assert FOOD_RADIUS <= food.y <= world.config.height - FOOD_RADIUS


@pytest.mark.parametrize(
    "position,velocity,dt,expected_position,expected_velocity",
    [
        (9, 4, 1, 7, -4),
        (1, -4, 1, 3, 4),
        (5, 35, 1, 0, 35),
        (5, -35, 1, 10, -35),
        (9, 1, 1, 10, -1),
        (1, -1, 1, 0, 1),
    ],
)
def test_food_bounces_preserve_speed_and_overshoot(
    position, velocity, dt, expected_position, expected_velocity
):
    x, vx = reflect_axis(position, velocity, dt, 0, 10)
    assert x == pytest.approx(expected_position)
    assert vx == expected_velocity


def test_worm_changes_both_directions_at_a_corner():
    world = World()
    world.food = [Food(953, 633, 0, 30, 30)]
    world._move_food()
    food = world.food[0]
    assert (food.vx, food.vy) == (-30, -30)
    assert (food.x, food.y) == (952, 632)


def test_idle_bug_does_not_earn_odor_reward_from_moving_food():
    world = World()
    a, b = world.insects
    a.x, a.y, b.x, b.y = 100, 100, 800, 500
    world.food = [Food(300, 100, 0, -45, 0)]
    _, rewards, _ = world.step([Action.REST, Action.REST])
    assert world.food[0].x < 300
    np.testing.assert_allclose(rewards, [-0.001, -0.001], atol=1e-7)


def test_default_horizon_extends_past_day_six_and_ends_at_day_thirty():
    world = World()
    assert world.config.episode_days == 30
    world.frame = world.config.day_frames * 6
    world.day = 6
    assert not world.done
    world.frame = world.config.episode_frames - 1
    world.day = 29
    assert not world.done
    world.tick()
    assert world.done
    assert world.day == 30


def mating_world():
    world = World()
    world.food.clear()
    a, b = world.insects
    a.x, a.y, a.angle = 400, 300, math.pi
    b.x, b.y, b.angle = 436, 300, 0
    for bug in world.insects:
        bug.fertile_until = 60
        bug.ate_day = 0
    return world


def test_mating_requires_abdomen_contact_then_completes_and_has_cooldown():
    world = mating_world()
    world.begin_actions([Action.COURT, Action.COURT])
    world.tick()
    assert all(b.mating(world.now) for b in world.insects)
    assert world.eggs == 0
    for _ in range(75):
        world.tick()
    assert world.pairings == 1
    assert world.eggs == 8
    assert all(b.mating_count == 1 for b in world.insects)
    assert all(b.cooldown_until > world.now for b in world.insects)
    assert all(b.score > 121 for b in world.insects)
    for _ in range(75):
        world.tick()
    assert world.pairings == 1
    assert len(world.insects) == 2  # Eggs do not silently create additional agents.


@pytest.mark.parametrize("condition", ["hungry", "heads_touch", "too_far", "no_court", "expiring"])
def test_mating_rejects_invalid_conditions(condition):
    world = mating_world()
    a, b = world.insects
    actions = [Action.COURT, Action.COURT]
    if condition == "hungry":
        a.fertile_until = 0
    elif condition == "heads_touch":
        a.angle, b.angle = 0, math.pi
    elif condition == "too_far":
        b.x += 200
    elif condition == "no_court":
        actions = [Action.REST, Action.REST]
    else:
        a.fertile_until = 1
    world.begin_actions(actions)
    world.tick()
    assert not any(bug.mating(world.now) for bug in world.insects)


def test_court_motor_reflex_can_bring_abdomens_together():
    world = mating_world()
    world.insects[1].x = 490
    for _ in range(6):
        world.step([Action.COURT, Action.COURT])
    assert world.pairings == 1


def test_rewards_add_up_to_individual_scores():
    world = World()
    total = np.zeros(2)
    for _ in range(100):
        _, rewards, _ = world.step([Action.WALK, Action.LEFT])
        total += rewards
    np.testing.assert_allclose(total, [b.score - 100 for b in world.insects], rtol=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps": 0},
        {"day_seconds": -1},
        {"radius": float("nan")},
        {"day_seconds": 0.2},
        {"width": 20},
        {"episode_days": 0},
    ],
)
def test_invalid_world_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        WorldConfig(**kwargs)


def test_invalid_actions_are_rejected_without_advancing_time():
    world = World()
    for actions in ([0], [0, 200]):
        with pytest.raises(ValueError):
            world.step(actions)
        assert world.frame == 0
