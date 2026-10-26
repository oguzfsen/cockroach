from dataclasses import dataclass


@dataclass(frozen=True)
class WorldConfig:
    width: float = 960.0
    height: float = 640.0
    fps: int = 30
    decision_frames: int = 24
    day_seconds: float = 60.0
    episode_days: int = 6
    radius: float = 18.0
    speed: float = 90.0
    turn_speed: float = 2.5
    scent_scale: float = 180.0
    sniff_range: float = 420.0
    passive_range: float = 140.0
    mating_seconds: float = 2.4
    mating_cooldown: float = 15.0

    def __post_init__(self):
        numeric = (
            self.width,
            self.height,
            self.day_seconds,
            self.radius,
            self.speed,
            self.turn_speed,
            self.scent_scale,
            self.sniff_range,
            self.passive_range,
            self.mating_seconds,
            self.mating_cooldown,
        )
        import math

        if any(not math.isfinite(x) or x <= 0 for x in numeric):
            raise ValueError("Dünya parametreleri pozitif ve sonlu olmalı.")
        if min(self.width, self.height) < self.radius * 8:
            raise ValueError("Akvaryum iki böceğin rahatça sığabileceği kadar büyük olmalı.")
        if any(
            type(x) is not int or x < 1 for x in (self.fps, self.decision_frames, self.episode_days)
        ):
            raise ValueError("fps, decision_frames ve episode_days pozitif tamsayı olmalı.")
        if self.day_seconds * self.fps < self.decision_frames:
            raise ValueError("Bir gün en az bir karar süresi kadar olmalı.")

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def decision_seconds(self) -> float:
        return self.decision_frames / self.fps

    @property
    def day_frames(self) -> int:
        return round(self.day_seconds * self.fps)

    @property
    def episode_frames(self) -> int:
        return self.day_frames * self.episode_days
