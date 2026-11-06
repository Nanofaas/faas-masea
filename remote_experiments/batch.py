"""Experiment and Batch value objects: build once, serialize, run later."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Experiment:
  id: str
  suite: str
  algorithm: str
  seed: int
  graph_params: dict
  load_params: dict
  config: dict

  def to_dict(self) -> dict:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict) -> Experiment:
    return cls(**data)


@dataclass(frozen=True)
class Batch:
  suite: str
  experiments: tuple[Experiment, ...]

  def to_dict(self) -> dict:
    return {"suite": self.suite, "experiments": [e.to_dict() for e in self.experiments]}

  @classmethod
  def from_dict(cls, data: dict) -> Batch:
    return cls(
      suite=data["suite"],
      experiments=tuple(Experiment.from_dict(e) for e in data["experiments"]),
    )

  def save(self, path: str | Path) -> None:
    Path(path).write_text(json.dumps(self.to_dict(), indent=2))

  @classmethod
  def load(cls, path: str | Path) -> Batch:
    return cls.from_dict(json.loads(Path(path).read_text()))
