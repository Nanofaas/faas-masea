"""The lifecycle shared by the runners: set-up, step loop, outputs.

A runner supplies the decision of one step, ``solve_step(ctx, t, data) -> StepResult``:

  return run_frame.run(config, parallelism, log_on_file=..., disable_plotting=...,
                       method="FaaS-MADiG", solve_step=_solve_step, outputs=OUTPUTS)

The frame creates the run folder and config.json, opens the run log, loads the problem (the
optional fixed reference on first use), then for each step of ``run_steps(config)``: current load and data,
``step_time.start_step()``, ``solve_step``, ``end_step``, decoding of the named solutions,
checkpoints; at the end it writes the solutions, obj.csv, termination_condition.csv, runtime.csv
and the plots, and returns the run folder. The public runner keeps the worker pool
(``parallel_solver_run``); the frame never creates one.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property

import pandas as pd

from utils import communication, run_log, step_time
from utils.compute_backend import compute_session
from utils.common import checkpoint_due, run_steps, run_timestamp

log = run_log.logger(__name__)


@dataclass(frozen=True)
class Output:
  checkpoint: bool = True
  plot_filename: str | None = None  # final plot of this output (sp.png), if any
  check: bool = True                # centralized feasibility check when decoding (not RMP parts)
  plot_max_size: int | None = 10    # plot only when Nn and Nf are at most this (None: any size)
  periodic_plot: bool = False       # also <name>/<t>_plot/<name>.png every plot_interval steps


LSP_OUTPUTS = {  # the local (LSP) and centralized-incumbent (LSPc) solutions of most runners
  "LSP": Output(checkpoint=True, plot_filename="sp.png"),
  "LSPc": Output(checkpoint=True),
}


@dataclass
class StepResult:
  solutions: dict           # output name -> dict of x, y, z, r, xi, rho, U arrays
  objective: float          # the obj.csv value of the step
  termination: str
  runtime: dict | None = None  # centralized only: algorithm_s/workers of the solver call
  # (a centralized result: termination_condition.csv has the model header and no index)
  data: dict | None = None     # the data the solutions refer to (default: the step data)


@dataclass
class RunContext:
  config: dict
  parallelism: int
  folder: str
  log: object
  solver_name: str
  solver_options: dict
  general_options: dict
  time_limit: float
  tolerance: float
  checkpoint_interval: int
  steps: range
  base_data: dict
  traces: dict
  agents: list
  graph: object
  Nn: int
  Nf: int
  neighborhood: object
  latency: object           # sparse network_latency adjacency of the graph
  complete: dict = field(default_factory=dict)    # output name -> complete solution
  objectives: list = field(default_factory=list)  # obj.csv series

  @property
  def max_iterations(self) -> int:
    """Required by the iterative runners only: a missing key fails with its name."""
    return self.config["max_iterations"]

  @property
  def patience(self) -> int:
    return self.config["patience"]

  @cached_property
  def reference(self) -> tuple | None:
    """(solution, replicas, detailed_fwd) of ``opt_solution_folder``, loaded on first use."""
    if "opt_solution_folder" not in self.config:
      return None
    from postprocessing import load_solution
    with step_time.untimed():
      solution, replicas, detailed, _, _ = load_solution(
        self.config["opt_solution_folder"], "LoadManagementModel")
    return solution, replicas, detailed

  @cached_property
  def dense_latency(self):
    """``latency`` as a dense array, built on first use (read it inside ``untimed``)."""
    import numpy as np
    return np.asarray(self.latency.todense())

  def step_data(self, data: dict, t: int) -> dict:
    """A copy of the step data, with the replicas of the fixed reference at ``t`` (r_bar)."""
    step = deepcopy(data)
    if self.reference is not None:
      from utils.centralized import encode_solution
      solution, replicas, detailed = self.reference
      _, _, _, r, _ = encode_solution(self.Nn, self.Nf, solution, detailed, replicas, t)
      step[None]["r_bar"] = {(n + 1, f + 1): int(r[n, f]) for n in range(self.Nn) for f in range(self.Nf)}
    return step


@contextmanager
def open_context(config: dict, parallelism: int, *, log_on_file: bool):
  """Run folder, config.json, run log and problem data of one run."""
  from networkx import adjacency_matrix
  from run_centralized_model import init_problem
  from run_faasmadea import neigh_dict_to_matrix
  folder = f"{config['base_solution_folder']}/{run_timestamp()}"
  os.makedirs(folder, exist_ok=True)
  with open(os.path.join(folder, "config.json"), "w") as ostream:
    ostream.write(json.dumps(config, indent=2))
  with compute_session(config.get("solver_options", {}).get("general", {})) as coverage, \
      run_log.open_run_log(folder if log_on_file else None, config.get("verbose", 0)):
    limits = config["limits"]
    base_data, traces, agents, graph = init_problem(
      limits, limits["load"].get("trace_type", "fixed_sum"), config["max_steps"], config["seed"], folder,
    )
    Nn, Nf = base_data[None]["Nn"][None], base_data[None]["Nf"][None]
    solver_options = config.get("solver_options", {})
    ctx = RunContext(
      config=config, parallelism=parallelism, folder=folder, log=log,
      solver_name=config["solver_name"], solver_options=solver_options,
      general_options=solver_options.get("general", {}),
      time_limit=step_time.algorithm_time_limit(config), tolerance=config.get("tolerance", 1e-6),
      checkpoint_interval=config["checkpoint_interval"], steps=run_steps(config),
      base_data=base_data, traces=traces, agents=agents, graph=graph, Nn=Nn, Nf=Nf,
      neighborhood=neigh_dict_to_matrix(base_data[None]["neighborhood"], Nn),
      latency=adjacency_matrix(graph, weight="network_latency"),
    )
    try:
      yield ctx
    finally:
      if ctx.general_options.get('compute_device', 'cpu') == 'cuda':
        with step_time.untimed():
          with open(os.path.join(folder,'compute_coverage.json'),'w') as stream:
            json.dump(dict(selected_device='cuda',operations=coverage,
                           zero_coverage=not any(r['backend']=='cuda' for r in coverage)),stream,indent=2)


def run(config: dict, parallelism: int, *, log_on_file: bool, disable_plotting: bool, method: str,
        solve_step, outputs: dict) -> str:
  from run_centralized_model import get_current_load, init_complete_solution, save_checkpoint, update_data
  from utils.centralized import check_feasibility, decode_solution
  with open_context(config, parallelism, log_on_file=log_on_file) as ctx, \
      communication.session(config.get('communication_metrics', False), method, ctx.folder):
    ctx.complete = {name: init_complete_solution() for name in outputs}
    terminations, rows, centralized = [], [], False
    for t in ctx.steps:
      communication.begin_step(t)
      log.info("t = %d", t)
      data = update_data(ctx.base_data, {"incoming_load": get_current_load(ctx.traces, ctx.agents, t)})
      clock = step_time.start_step()
      try:
        result = solve_step(ctx, t, data)
      except BaseException:
        step_time.cancel_step(clock)
        raise
      row = step_time.end_step(clock)
      row.update(result.runtime or {})
      centralized = centralized or result.runtime is not None
      rows.append(dict(step=t, **row))
      ctx.objectives.append(result.objective)
      for name, s in result.solutions.items():  # outside the clock: decoding and checks
        ctx.complete[name] = decode_solution(s["x"], s["y"], s["z"], s["r"], s["xi"], s["rho"], s["U"],
                                             ctx.complete[name])
        if outputs[name].check:
          feasible, why = check_feasibility(s["x"], s["y"].sum(axis=1), s["z"], s["r"], s["U"],
                                            result.data if result.data is not None else data)
          if not feasible:
            raise RuntimeError(why)
        if outputs[name].checkpoint and checkpoint_due(t, ctx.steps, ctx.checkpoint_interval):
          save_checkpoint(ctx.complete[name], os.path.join(ctx.folder, name), t)
        if outputs[name].periodic_plot:
          _periodic_plot(ctx, name, outputs[name], t, disable_plotting)
      terminations.append(result.termination)
      log.info("    TOTAL RUNTIME [s] = %s (wallclock: %s)", row["algorithm_s"], row["wall_s"])
    _write_outputs(ctx, outputs, method, terminations, rows, disable_plotting, centralized)
    communication.finish()
    log.info("All solutions saved in: %s", ctx.folder)
    return ctx.folder


def _plottable(ctx, output: Output, disable_plotting: bool) -> bool:
  small = output.plot_max_size is None or max(ctx.Nf, ctx.Nn) <= output.plot_max_size
  return not disable_plotting and small


def _periodic_plot(ctx, name: str, output: Output, t: int, disable_plotting: bool) -> None:
  """Intermediate history of a centralized model, outside the clock."""
  from run_centralized_model import join_complete_solution, plot_history
  config = ctx.config
  interval = config.get("plot_interval", config["max_steps"])
  if not (0 < t < ctx.steps.stop - 1 and t % interval == 0) or not _plottable(ctx, output, disable_plotting):
    return
  solution, offloaded, _ = join_complete_solution(ctx.complete[name])
  folder = os.path.join(ctx.folder, name, f"{t}_plot")
  os.makedirs(folder)
  plot_history(
    ctx.traces, config.get("min_run_time", 0), t, config.get("run_time_step", 1), solution,
    ctx.complete[name]["utilization"], ctx.complete[name]["replicas"], offloaded, ctx.objectives,
    os.path.join(folder, f"{name}.png"),
  )


def _write_outputs(ctx, outputs, method, terminations, rows, disable_plotting, centralized) -> None:
  from run_centralized_model import join_complete_solution, plot_history, save_solution
  config = ctx.config
  for name, output in outputs.items():
    solution, offloaded, detailed = join_complete_solution(ctx.complete[name])
    if output.plot_filename and _plottable(ctx, output, disable_plotting):
      plot_history(
        ctx.traces, config.get("min_run_time", 0), config.get("max_run_time", config["max_steps"]),
        config.get("run_time_step", 1), solution, ctx.complete[name]["utilization"],
        ctx.complete[name]["replicas"], offloaded, ctx.objectives,
        os.path.join(ctx.folder, output.plot_filename),
      )
    save_solution(solution, offloaded, ctx.complete[name], detailed, name, ctx.folder)
  pd.DataFrame(ctx.objectives, columns=[method]).to_csv(os.path.join(ctx.folder, "obj.csv"), index=False)
  if centralized:  # the model's own column, as the centralized results always had
    pd.DataFrame({method: terminations}).to_csv(os.path.join(ctx.folder, "termination_condition.csv"), index=False)
  else:
    pd.DataFrame(terminations).to_csv(os.path.join(ctx.folder, "termination_condition.csv"))
  step_time.write_runtime(ctx.folder, rows)


class Incumbent:
  """Best solution by centralized welfare: strictly greater replaces it, ties keep it."""

  def __init__(self, objective: float, solution: dict | None = None, iteration: int = -1):
    self.objective, self.solution, self.iteration = objective, solution, iteration

  def consider(self, solution: dict, objective: float, iteration: int) -> bool:
    if not objective > self.objective:
      return False
    with step_time.untimed():
      self.objective, self.solution, self.iteration = objective, deepcopy(solution), iteration
    return True


def snapshot(data: dict, *args, **kwargs) -> tuple[dict, float]:
  """Observer: combine the solution (combine_solutions arguments), check it, score it."""
  from run_faasmacro import combine_solutions
  from utils.centralized import check_feasibility
  from utils.faasmacro import compute_centralized_objective
  with step_time.untimed():
    csol = combine_solutions(*args, **kwargs)
    sp = csol["sp"]
    objective = compute_centralized_objective(data, sp["x"], sp["y"], sp["z"])
    feasible, why = check_feasibility(sp["x"], sp["y"].sum(axis=1), sp["z"], sp["r"], sp["U"], data)
    if not feasible:
      raise RuntimeError(why)
  return csol, objective
