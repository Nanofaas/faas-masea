from math import isfinite

import numpy as np


def relative_objective_gap(value: float, reference: float) -> float:
  """Return a nonnegative gap; an undefined comparison cannot mean convergence."""
  if not isfinite(value) or not isfinite(reference):
    return float("inf")
  if reference == 0:
    return 0.0 if value == 0 else float("inf")
  return abs(value - reference) / abs(reference)


def compute_centralized_objective(
    sp_data: dict, sp_x: np.array, sp_y: np.array, sp_z: np.array
  ) -> float:
  Nn = sp_data[None]["Nn"][None]
  Nf = sp_data[None]["Nf"][None]
  # objective function weights
  alpha = np.zeros((Nn,Nf))
  for (n,f), a in sp_data[None]["alpha"].items():
    alpha[n-1,f-1] = a
  beta = np.zeros((Nn,Nn,Nf))
  for (n1,n2,f), b in sp_data[None]["beta"].items():
    beta[n1-1,n2-1,f-1] = b
  gamma = np.zeros((Nn,Nf))
  for (n,f), g in sp_data[None]["gamma"].items():
    gamma[n-1,f-1] = g
  # value
  tot = 0.0
  for n1 in range(Nn):
    for f in range(Nf):
      # A zero-load source has no local, outbound or rejected traffic. Keep
      # every positive load unchanged, including fractional arrival rates.
      load = sp_data[None]["incoming_load"][(n1+1,f+1)] or 1
      tot += alpha[n1,f] * sp_x[n1,f] / load
      tot -= gamma[n1,f] * sp_z[n1,f] / load
      for n2 in range(Nn):
        tot += beta[n1,n2,f] * sp_y[n1,n2,f] / load
  return tot#(alpha * sp_x).sum() + (beta * sp_y).sum() - (gamma * sp_z).sum()
