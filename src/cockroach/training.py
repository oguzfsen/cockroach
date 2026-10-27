from __future__ import annotations

import csv
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np

from cockroach import checkpoint
from cockroach.brain import EXPLORATION, Brain
from cockroach.config import WorldConfig
from cockroach.world import Action, World

VALIDATION_EVERY = 20_000


def random_action(rng: np.random.Generator) -> int:
    """A motor decision lasts 0.8 s, including in this untrained baseline."""
    return int(rng.choice(len(EXPLORATION), p=EXPLORATION))


def train(
    *,
    minutes: float | None,
    steps: int | None,
    seed: int,
    output: Path,
    resume: Path | None = None,
    config: WorldConfig | None = None,
):
    if resume:
        brains, config, previous = checkpoint.load(resume, training=True)
    else:
        config = config or WorldConfig()
        brains = [Brain(seed * 2 + i) for i in range(2)]
        previous = {"decisions": 0, "episodes": 0, "elapsed_seconds": 0.0, "seed": seed}
    seed = previous.get("seed", seed)
    # Resume begins a new episode; replay and in-progress world are intentionally fresh.
    episodes = previous["episodes"]
    decisions = previous["decisions"]
    world = World(config, seed + episodes)
    observations = world.observations()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".csv")
    best_path = output.with_name(output.stem + ".best.npz")
    validation_path = output.with_suffix(".validation.jsonl")
    best_validation = previous.get("best_validation") if resume and best_path.exists() else None
    last_validated = -1
    validations = 0
    history = deque(maxlen=30)
    start = time.perf_counter()
    last_report = start
    last_save = start
    local_steps = 0
    interrupted = False

    def state():
        return {
            "decisions": decisions,
            "episodes": episodes,
            "seed": seed,
            "elapsed_seconds": previous.get("elapsed_seconds", 0) + time.perf_counter() - start,
            "best_validation": best_validation,
        }

    def validate():
        nonlocal best_validation, last_validated, validations
        if decisions == last_validated:
            return
        # Fixed validation worlds are separate from training AND final evaluation.
        records = [rollout(config, seed + 1_000_000 + i, brains) for i in range(5)]
        candidate = {
            "decisions": decisions,
            "seed": seed + 1_000_000,
            "episodes": 5,
            "mean_eaten": float(np.mean([r["eaten"] for r in records])),
            "mean_score": float(np.mean([r["score"] for r in records])),
        }
        grade = (candidate["mean_eaten"], candidate["mean_score"])
        best_grade = (
            (best_validation["mean_eaten"], best_validation["mean_score"])
            if best_validation
            else (-1, -float("inf"))
        )
        selected = grade > best_grade
        if selected:
            best_validation = candidate
            checkpoint.save(best_path, brains, config, state())
        with validation_path.open("a" if resume or validations else "w", encoding="utf-8") as log:
            log.write(json.dumps({**candidate, "selected": selected}) + "\n")
        coverage = candidate["mean_eaten"] / (config.episode_days * 2)
        print(
            f"  Doğrulama: besin %{coverage * 100:.1f}"
            + (f" · en iyi model → {best_path}" if selected else " · önceki en iyi korundu"),
            flush=True,
        )
        last_validated = decisions
        validations += 1

    print("\n  COCKROACH LAB / PENCERESİZ EĞİTİM")
    print(
        f"  İki bağımsız Double DQN · karar {config.decision_seconds:.2f} s · "
        f"gün {config.day_frames / config.fps:g} s"
    )
    print(f"  Model: {output}   |   Günlük: {log_path}")
    print("  Ctrl+C: ilerlemeyi kaydet ve çık.\n", flush=True)
    fields = [
        "episode",
        "decisions",
        "elapsed_seconds",
        "days",
        "food_spawned",
        "eaten",
        "missed",
        "pairings",
        "eggs",
        "score",
        "epsilon",
        "loss_female",
        "loss_male",
    ]
    append = bool(resume and log_path.exists())
    with log_path.open("a" if append else "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not append:
            writer.writeheader()
        try:
            while True:
                elapsed = time.perf_counter() - start
                if (steps is not None and local_steps >= steps) or (
                    minutes is not None and elapsed >= minutes * 60
                ):
                    break
                actions = [
                    b.act(observations[i], explore=True)
                    if not world.insects[i].dead
                    else Action.REST
                    for i, b in enumerate(brains)
                ]
                alive = [not bug.dead for bug in world.insects]
                next_observations, rewards, done = world.step(actions)
                for i, brain in enumerate(brains):
                    if alive[i]:
                        brain.remember(
                            observations[i],
                            actions[i],
                            rewards[i],
                            next_observations[i],
                            done or world.insects[i].dead,
                        )
                observations = next_observations
                local_steps += 1
                decisions += 1
                if done:
                    episodes += 1
                    summary = world.summary()
                    history.append(summary)
                    writer.writerow(
                        {
                            "episode": episodes,
                            "decisions": decisions,
                            "elapsed_seconds": round(time.perf_counter() - start, 3),
                            **summary,
                            "epsilon": round(brains[0].epsilon, 4),
                            "loss_female": brains[0].last_loss,
                            "loss_male": brains[1].last_loss,
                        }
                    )
                    file.flush()
                    world = World(config, seed + episodes)
                    observations = world.observations()
                now = time.perf_counter()
                if now - last_report >= 2:
                    rate = local_steps * config.decision_seconds / (now - start)
                    eaten = sum(h["eaten"] for h in history)
                    spawned = sum(h["food_spawned"] for h in history)
                    coverage = eaten / max(1, spawned)
                    print(
                        f"  {now - start:6.1f}s | bölüm {episodes:4d} | karar {decisions:7,d} | "
                        f"{rate:5.0f}× | ε {brains[0].epsilon:.2f} | "
                        f"son 30 bölüm besin %{coverage * 100:4.1f}",
                        flush=True,
                    )
                    last_report = now
                if now - last_save >= 15:
                    checkpoint.save(output, brains, config, state())
                    last_save = now
                if decisions % VALIDATION_EVERY == 0:
                    validate()
            validate()
        except KeyboardInterrupt:
            interrupted = True
        finally:
            checkpoint.save(output, brains, config, state())
    elapsed = time.perf_counter() - start
    print(
        f"\n  {'Durduruldu' if interrupted else 'Tamamlandı'}: {local_steps:,} karar, "
        f"{elapsed:.1f} s. Model kaydedildi: {output}"
    )
    print(f"  İzle: cockroach view --model {best_path if best_path.exists() else output}")
    return brains, config, state()


def rollout(
    config: WorldConfig, seed: int, brains: list[Brain] | None, epsilon: float = 0.0
) -> dict:
    world = World(config, seed)
    rng = np.random.default_rng(seed + 1_000_000)
    observations = world.observations()
    while not world.done:
        actions = [
            Action.REST
            if world.insects[i].dead
            else (
                random_action(rng)
                if brains is None or rng.random() < epsilon
                else brains[i].act(observations[i])
            )
            for i in range(2)
        ]
        observations, _, _ = world.step(actions)
    return world.summary()


def evaluate(model: Path, episodes: int, seed: int, output: Path | None, epsilon: float = 0.0):
    brains, config, training = checkpoint.load(model)
    totals = {"random": [], "learned": []}
    start = time.perf_counter()
    print(f"\n  {episodes} eşleştirilmiş bölüm · {config.episode_days} gün/bölüm · ε={epsilon:g}")
    for i in range(episodes):
        for label, policy in (("random", None), ("learned", brains)):
            totals[label].append(rollout(config, seed + i, policy, epsilon))
        print(
            f"  Bölüm {i + 1:3d}/{episodes}: besin "
            f"rastgele {totals['random'][-1]['eaten']:2d} / "
            f"öğrenilmiş {totals['learned'][-1]['eaten']:2d}",
            flush=True,
        )
    report = {
        "model": str(model),
        "seed": seed,
        "episodes": episodes,
        "epsilon": epsilon,
        "training": training,
        "results": {},
    }
    for label, records in totals.items():
        count = sum(r["eaten"] for r in records)
        spawned = sum(r["food_spawned"] for r in records)
        report["results"][label] = {
            "food_coverage": count / spawned,
            "mean_eaten": float(np.mean([r["eaten"] for r in records])),
            "mean_score": float(np.mean([r["score"] for r in records])),
            "pairings": sum(r["pairings"] for r in records),
            "episodes": records,
        }
    # Paired bootstrap resamples complete episodes, not correlated individual frames.
    differences = np.array(
        [b["eaten"] - a["eaten"] for a, b in zip(totals["random"], totals["learned"], strict=False)]
    )
    rng = random.Random(seed)
    means = [float(np.mean(rng.choices(differences.tolist(), k=episodes))) for _ in range(2000)]
    low, high = np.percentile(means, [2.5, 97.5])
    report["paired_food_difference"] = {
        "mean": float(differences.mean()),
        "bootstrap_95_ci": [float(low), float(high)],
    }
    for label, title in (("random", "Rastgele"), ("learned", "Öğrenilmiş")):
        r = report["results"][label]
        print(
            f"  {title:12s} besin %{100 * r['food_coverage']:5.1f} | "
            f"skor {r['mean_score']:7.2f} | çiftleşme {r['pairings']}"
        )
    print(
        f"  Bölüm başına besin farkı: {differences.mean():+.2f} "
        f"(%95 bootstrap aralığı {low:+.2f} … {high:+.2f})"
    )
    print(f"  Süre: {time.perf_counter() - start:.1f} s")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  Rapor: {output}")
    return report
