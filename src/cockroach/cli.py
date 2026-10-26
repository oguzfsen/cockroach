from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

# These networks are tiny: a BLAS thread pool costs more than its matrix products.
# This is set before NumPy imports; library consumers keep their own thread policy.
os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("COCKROACH_BLAS_THREADS", "1")
os.environ["OMP_NUM_THREADS"] = os.environ.get("COCKROACH_BLAS_THREADS", "1")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Pozitif, sonlu bir sayı gerekli.")
    return result


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Pozitif bir tamsayı gerekli.")
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="cockroach", description="İki böcek, iki zeka, tek akvaryum."
    )
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Pencere açmadan hızlı öğrenme")
    duration = train.add_mutually_exclusive_group()
    duration.add_argument(
        "--minutes", type=positive_float, help="Gerçek duvar saati dakikası (varsayılan 5)"
    )
    duration.add_argument("--steps", type=positive_int, help="Tekrarlanabilir sabit karar sayısı")
    train.add_argument("--output", type=Path, default=Path("runs/latest.npz"))
    train.add_argument(
        "--resume", type=Path, help="Bu modelden, yeni bölüm ve boş replay ile devam et"
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--day-seconds", type=positive_float, default=None)
    train.add_argument("--episode-days", type=positive_int, default=None)
    view = commands.add_parser("view", help="Öğrenme yapmadan izle veya elle yönet")
    view.add_argument("--model", type=Path, help="Varsa runs/latest.best.npz otomatik açılır")
    view.add_argument(
        "--random", action="store_true", help="Öğrenilmiş model yerine rastgele davranış"
    )
    view.add_argument("--seed", type=int, default=7)
    view.add_argument("--headless", action="store_true", help="Ekransız görsel kontrol (SDL dummy)")
    view.add_argument("--frames", type=positive_int, help="Bu kadar çizimden sonra çık")
    view.add_argument("--screenshot", type=Path, help="Çıkışta PNG kaydet")
    evaluation = commands.add_parser(
        "evaluate", help="Modeli aynı tohumlardaki rastgele politika ile karşılaştır"
    )
    evaluation.add_argument("--model", type=Path, default=Path("runs/latest.npz"))
    evaluation.add_argument("--episodes", type=positive_int, default=20)
    evaluation.add_argument("--seed", type=int, default=10000)
    evaluation.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    evaluation.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Değerlendirmede keşif olasılığı, varsayılan 0 (açgözlü)",
    )
    return root


def main(argv=None) -> int:
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == "train":
            from cockroach.config import WorldConfig
            from cockroach.training import train

            if args.resume and (args.day_seconds is not None or args.episode_days is not None):
                root.error("--resume ortam parametrelerini modelden okur; gün ayarlarını kaldırın.")
            config = WorldConfig(
                day_seconds=args.day_seconds or 60, episode_days=args.episode_days or 6
            )
            train(
                minutes=args.minutes if args.minutes is not None else (None if args.steps else 5),
                steps=args.steps,
                seed=args.seed,
                output=args.output,
                resume=args.resume,
                config=config,
            )
        elif args.command == "evaluate":
            from cockroach.training import evaluate

            if not 0 <= args.epsilon <= 1:
                root.error("--epsilon 0 ile 1 arasında olmalı.")
            evaluate(args.model, args.episodes, args.seed, args.output, args.epsilon)
        else:
            if args.model and args.random:
                root.error("--model ve --random birlikte kullanılamaz.")
            if args.headless:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
                os.environ["SDL_AUDIODRIVER"] = "dummy"
                if args.frames is None:
                    args.frames = 90
            from cockroach.viewer import run

            model = args.model
            if model is None and not args.random:
                for candidate in (
                    Path("runs/latest.best.npz"),
                    Path("runs/latest.npz"),
                    Path("examples/forager.npz"),
                ):
                    if candidate.exists():
                        model = candidate
                        break
            run(
                model=model,
                seed=args.seed,
                frames=args.frames,
                screenshot=args.screenshot,
                headless=args.headless,
            )
    except (ValueError, OSError) as error:
        print(f"Hata: {error}")
        return 1
    return 0
