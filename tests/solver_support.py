"""Probe the licensed solver once; package availability alone is insufficient."""

from functools import lru_cache

import pytest


@lru_cache(maxsize=1)
def gurobi_unavailable_reason():
  try:
    import gurobipy as gp
  except ImportError:
    return "Gurobi Python package is not installed"
  try:
    with gp.Env(empty=True) as env:
      env.setParam("OutputFlag", 0)
      env.start()
      with gp.Model(env=env) as model:
        x = model.addVar(ub=1)
        model.setObjective(x, gp.GRB.MAXIMIZE)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
          return "Gurobi did not solve the license probe"
  except gp.GurobiError as error:
    return f"Gurobi unavailable: {error}"
  return None


def require_gurobi():
  reason = gurobi_unavailable_reason()
  if reason is not None:
    pytest.skip(reason)
