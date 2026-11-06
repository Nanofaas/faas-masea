"""Centralized welfare maximization with each node's local x-gain guarantee."""

import pyomo.environ as pyo
from utils import step_time

from models.model import LoadManagementModel
from models.local_sp import solve_agent_problem
from models.sp import LSP_detailed


class SelfishLoadManagementModel(LoadManagementModel):
  """
  Solve LSP_detailed per node, then protect only its alpha*x/load gain.
  The local solve includes y, z and pi; its total objective is not the floor.
  The central objective and physical constraints are inherited unchanged.
  Use this class's solve() to compute the floors before the global solve.
  Among tied local optima, the reference is the solution returned by the solver
  """

  def __init__(self):
    super().__init__()
    self.name = "SelfishLoadManagementModel"
    self.model.pi = pyo.Param(
      self.model.F, within = pyo.NonNegativeReals, default = 0
    )
    # No default: a direct solver call cannot silently omit the local guarantee.
    self.model.minimum_local_gain = pyo.Param(
      self.model.N, within = pyo.NonNegativeReals, mutable = True
    )
    self.model.local_processing_gain = pyo.Expression(
      self.model.N, rule = self.local_processing_gain
    )
    self.model.protect_local_gain = pyo.Constraint(
      self.model.N, rule = self.protect_local_gain
    )

  @staticmethod
  def local_processing_gain(model, n):
    return sum(
      model.alpha[n, f] * model.x[n, f] / (model.incoming_load[n, f] or 1)
      for f in model.F
    )

  @staticmethod
  def protect_local_gain(model, n):
    return model.local_processing_gain[n] >= model.minimum_local_gain[n]

  def compute_local_gain_floors(
      self, instance, solver_options, solver_name = "glpk", *, deadline = None
    ):
    """Populate per-node floors and return the total local solver runtime."""
    local = LSP_detailed()
    # Copy evaluated central parameters, including defaults (gamma differs
    # between the original central and local classes).
    data = {None: {
      param.name: getattr(instance, param.name).extract_values()
      for param in local.model.component_objects(pyo.Param)
      if hasattr(instance, param.name)
    }}
    local_runtime = 0.0
    for n in instance.N:
      options = dict(solver_options)
      if deadline is not None:
        remaining = deadline - step_time.now()
        if remaining <= 0 or (solver_name in {"glpk", "glpsol"} and remaining < 1):
          raise TimeoutError("Local reference deadline exhausted")
        if solver_name in {"glpk", "glpsol"}:
          remaining = int(remaining)
        options["TimeLimit"] = min(options.get("TimeLimit", remaining), remaining)
      data[None]["whoami"] = {None: n}
      result = solve_agent_problem(local, data, options, solver_name)
      if deadline is not None and (
          step_time.now() >= deadline or result["termination_condition"] == "maxTimeLimit"
        ):
        raise TimeoutError("Local reference deadline exhausted")
      if (
          not result["solution_exists"] or
          result["termination_condition"] != "optimal"
        ):
        raise RuntimeError(
          f"Local reference for node {n} is not optimal: "
          f"{result['termination_condition']}"
        )
      instance.minimum_local_gain[n] = pyo.value(sum(
        instance.alpha[n, f] * result["x"][f - 1] / (
          instance.incoming_load[n, f] or 1
        )
        for f in instance.F
      ))
      local_runtime += result["runtime"]
    return local_runtime

  def solve(
      self,
      instance,
      solver_options,
      solver_name = "glpk",
      initial_solution = None
    ):
    local_runtime = self.compute_local_gain_floors(
      instance, solver_options, solver_name
    )
    solution = super().solve(
      instance, solver_options, solver_name, initial_solution
    )
    solution["local_reference_runtime"] = local_runtime
    solution["global_runtime"] = solution["runtime"]
    solution["runtime"] += local_runtime
    return solution
