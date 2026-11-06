from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# conductance-matrix column layout: D has shape (Nf, 2 + deg)
LOCAL = 0
REJ = 1


@dataclass(frozen=True)
class PlasmaOptions:
  # Layer A (round period W is the time unit: 1 round = W seconds)
  W: float = 1.0
  rounds_per_step: int = 20
  mu: float = 0.1
  kappa: float = 1.0
  D_init: float = 1.0
  D_min: float = 1e-3
  D_max: float = 1e3
  eps_explore: float = 0.01
  rej_floor: float = 0.01
  ewma: float = 0.3
  lambda_split_threshold: float = 5.0
  rare_function_mode: str = "sampled"  # "sampled" | "unsplittable"
  r_init: str = "spread"  # "spread" | "zero"
  # Layer B (k_sb = 0 disables SB entirely: replicas stay fixed)
  k_sb: int = 10
  n_sb_steps: int = 300
  sb_dt: float = 0.05
  sb_delta: float = 1.0
  sb_c0: float = 0.2
  sb_restarts: int = 8
  a_final: float = 1.0
  A: Optional[float] = None  # None -> auto: 2 * max_f(benefit_f / ram_req_f)
  B: float = 1.0
  C: float = 0.1
  switch_cost: float = 1.0
  z_delta: float = 2.0
  eps_commit: float = 0.05
  n_hyst: int = 2
  p_commit: float = 0.5
  sbm_method: str = "exact"  # "exact" | "dsb"
  # protocol
  hb_latency_rounds: int = 1
  hb_loss: float = 0.0
  staleness_rounds: int = 3

  @classmethod
  def from_config(cls, config: dict) -> "PlasmaOptions":
    return cls(**config.get("solver_options", {}).get("plasma", {}))

  def __post_init__(self) -> None:
    if self.rounds_per_step < 1:
      raise ValueError(f"rounds_per_step must be >= 1, got {self.rounds_per_step}")
    if self.W <= 0.0:
      raise ValueError(f"W must be > 0, got {self.W}")
    if self.hb_latency_rounds > self.staleness_rounds:
      raise ValueError(f"hb_latency_rounds ({self.hb_latency_rounds}) must be <= staleness_rounds ({self.staleness_rounds}), otherwise every neighbor is permanently stale")
    if self.sbm_method not in ("exact", "dsb"):
      raise ValueError(f"sbm_method must be 'exact' or 'dsb', got {self.sbm_method!r}")


@dataclass(frozen=True)
class Heartbeat:
  # the ENTIRE control-plane message: nothing else may cross an edge
  node: int
  seq: int
  spare: Tuple[float, ...]  # per function, floored: max(0, capacity_units - admitted)
  alpha: Tuple[float, ...]  # per function
  pull: Tuple[float, ...]   # per function, offload pressure last window
