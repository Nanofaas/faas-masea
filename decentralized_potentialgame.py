from run_faasmacro import (
  parallel_solver_run,
  solve_subproblem,
)
from run_faasmadea import (
  compute_residual_capacity,
  start_additional_replicas,
)
from utils import communication, run_frame, step_time
from utils.faasmacro import compute_centralized_objective
from utils.common import load_configuration
from models.model import PYO_VAR_TYPE
from models.sp import LSP_detailed, LSP, LSP_fixedr, LSP_pg, LSP_pg_fixedr, LSP_pg_fixedx
from models.local_sp import dp_enabled, try_solve_local

from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, Tuple
import pandas as pd
import numpy as np
import pyomo.environ as pyo
import argparse


def compute_z(x: np.array, y: np.array, sp_data: dict) -> np.array:
  """Cloud-forwarded load derived from flow conservation (same convention as
  combine_solutions): z = load - x - outbound."""
  Nn = sp_data[None]["Nn"][None]
  Nf = sp_data[None]["Nf"][None]
  z = np.zeros((Nn, Nf))
  for n in range(Nn):
    for f in range(Nf):
      load = sp_data[None]["incoming_load"][(n + 1, f + 1)]
      z[n, f] = max(0.0, load - x[n, f] - y[n, :, f].sum())
  return z


def compute_node_utility(
    i: int, x: np.array, y: np.array, z: np.array, sp_data: dict
  ) -> float:
  """Node i's share of the centralized objective. Summing over all nodes
  yields exactly compute_centralized_objective (the exact potential)."""
  values = sp_data[None]
  utility = 0.
  for f in range(x.shape[1]):
    load = values["incoming_load"][i + 1, f + 1] or 1
    utility += (
      values["alpha"].get((i + 1, f + 1), 0) * x[i, f]
      - values["gamma"].get((i + 1, f + 1), 0) * z[i, f]
      + sum(values["beta"].get((i + 1, j + 1, f + 1), 0) * y[i, j, f]
            for j in range(y.shape[1]))
    ) / load
  return float(utility)


def split_omega(
    i: int,
    omega_row: np.array,
    ledger: np.array,
    neighbours: set,
    sp_data: dict,
  ) -> np.array:
  """Fractional-knapsack split of node i's proposed offload across eligible
  sellers by descending beta (exact best response for the linear utility).
  Mutates ledger in place; unplaced residual is left to the Cloud (z)."""
  Nn = sp_data[None]["Nn"][None]
  Nf = sp_data[None]["Nf"][None]
  new_row = np.zeros((Nn, Nf))
  for f in range(Nf):
    if omega_row[f] <= 0:
      continue
    gamma_if = sp_data[None]["gamma"][(i + 1, f + 1)]
    score = {}
    for j in neighbours:
      b = sp_data[None]["beta"][(i + 1, j + 1, f + 1)]
      if b > -gamma_if and ledger[j, f] > 0:
        score[j] = b
    placed = 0.0
    for j in sorted(score, key=lambda k: (-score[k], k)):
      if placed >= omega_row[f]:
        break
      q = min(ledger[j, f], omega_row[f] - placed)
      if q <= 0:
        continue
      new_row[j, f] += q
      ledger[j, f] -= q
      placed += q
  return new_row


def propose_node_move(
    node: int,
    omega_ub_row: np.array,
    y: np.array,
    sp_data: dict,
    model,
    solver_name: str,
    general_solver_options: dict,
    parallelism: int,
  ) -> Tuple[np.array, np.array, np.array, float]:
  """Exact local proposal for one node, with MILP fallback: preserve committed
  inbound flows (y_bar) and the accessible-capacity cap (omega_ub).

  The proposal prices aggregate offloading with delta (as P2 does); the
  caller's epsilon-acceptance on the true beta-aware utility is what
  guarantees potential monotonicity, not proposal optimality."""
  Nn = sp_data[None]["Nn"][None]
  Nf = sp_data[None]["Nf"][None]
  values, owner = sp_data[None], node + 1
  required = [("memory_capacity", owner)]
  required += [("memory_requirement", f) for f in range(1, Nf + 1)]
  required += [(parameter, (owner, f)) for parameter in ("incoming_load", "demand")
               for f in range(1, Nf + 1)]
  if (dp_enabled(general_solver_options) and type(model) in (LSP_pg, LSP_pg_fixedr, LSP_pg_fixedx)
      and all(key in values.get(parameter, {}) for parameter, key in required)):
    # Only own parameters and committed inbound totals enter the local DP.
    # Remapping to one node avoids copying the network and rebuilding Nn² flows.
    local = {
      "Nn": {None: 1}, "Nf": {None: Nf}, "whoami": {None: 1},
      "memory_capacity": {1: values["memory_capacity"][owner]},
      "memory_requirement": {
        f: values["memory_requirement"][f] for f in range(1, Nf + 1)
      },
      "max_utilization": {
        f: values.get("max_utilization", {}).get(f, .8) for f in range(1, Nf + 1)
      },
      "pi": {f: 0. for f in range(1, Nf + 1)},
      "omega_ub": {f + 1: float(omega_ub_row[f]) for f in range(Nf)},
      "y_bar": {
        (1, 1, f + 1): sum(float(max(y[m, node, f], 0.0)) for m in range(Nn))
        for f in range(Nf)
      },
    }
    for parameter in ("incoming_load", "demand"):
      local[parameter] = {
        (1, f): values[parameter][owner, f] for f in range(1, Nf + 1)
      }
    for parameter, default in (("alpha", 1.), ("delta", .9), ("gamma", .1)):
      local[parameter] = {
        (1, f): values.get(parameter, {}).get((owner, f), default)
        for f in range(1, Nf + 1)
      }
    if type(model) is LSP_pg_fixedr:
      local["r_bar"] = {
        (1, f): values.get("r_bar", {}).get((owner, f), 0)
        for f in range(1, Nf + 1)
      }
    if type(model) is LSP_pg_fixedx:
      local["x_bar"] = {(1, f): values["x_bar"][owner, f] for f in range(1, Nf + 1)}
    solution = try_solve_local(model, {None: local})
    if solution is not None:
      return (
        np.array(solution["x"], dtype=float), np.array(solution["r"], dtype=float),
        np.array(solution["omega"], dtype=float), float(solution["runtime"]),
      )
  node_data = deepcopy(sp_data)
  node_data[None]["omega_ub"] = {
    (f + 1): float(omega_ub_row[f]) for f in range(Nf)
  }
  node_data[None]["y_bar"] = {
    (m + 1, n + 1, f + 1): float(max(y[m, n, f], 0.0))
    for m in range(Nn) for n in range(Nn) for f in range(Nf)
  }
  result = solve_subproblem(
    node_data, [node], model, solver_name, general_solver_options, parallelism
  )
  x_row = np.array(result[1][node, :], dtype=float)
  omega_row = np.array(result[4][node, :], dtype=float)
  r_row = np.array(result[5][node, :], dtype=float)
  runtime = float(result[10]["tot"])
  return x_row, r_row, omega_row, runtime


def compute_rho(r: np.array, sp_data: dict) -> np.array:
  """Memory slack per node for the CURRENT committed replicas. Must be
  recomputed whenever r changes (accepted moves re-decide local replicas),
  or the memory market would allocate replicas against stale slack."""
  Nn = sp_data[None]["Nn"][None]
  Nf = sp_data[None]["Nf"][None]
  rho = np.array([
    sp_data[None]["memory_capacity"][j + 1] - sum(
      r[j, f] * sp_data[None]["memory_requirement"][f + 1] for f in range(Nf)
    )
    for j in range(Nn)
  ], dtype=float)
  if not ((rho > -1e-6).all()):
    raise RuntimeError(f"memory budget violated: rho={rho}")
  return np.maximum(rho, 0.0)


def node_move(
    i: int,
    x: np.array,
    y: np.array,
    r: np.array,
    sp_data: dict,
    neighborhood: np.array,
    rho: np.array,
    epsilon: float,
    propose_fn: Callable,
    tolerance: float,
    *, integer_flows: bool = False,
  ) -> Tuple[bool, float, dict, float]:
  """One better-response move. Rules that keep Phi an exact potential:
  the mover keeps serving committed inbound flows (enforced by LSP_pg via
  y_bar) and only claims advertised residual capacity of its neighbours.
  The move is committed iff the TRUE utility (per-pair beta) improves by
  more than epsilon; x, y, r are mutated in place only on acceptance.
  Moves also respect the FRALB no-ping-pong constraint (see inline note)."""
  Nf = sp_data[None]["Nf"][None]
  memory_bids = {"i": [], "j": [], "f": []}
  neighbours = set(int(j) for j in np.nonzero(neighborhood[i, :])[0])
  communication.pg_visit(i, neighbours, Nf)
  z = compute_z(x, y, sp_data)
  u_old = compute_node_utility(i, x, y, z, sp_data)
  # residual capacity visible to i, with its own row released
  y_trial = np.array(y, dtype=float)
  y_trial[i, :, :] = 0.0
  _, ledger, _ = compute_residual_capacity(x, y_trial, r, sp_data)
  if integer_flows:
    ledger = np.floor(ledger + 1e-9)
  # FRALB no-ping-pong (validate_centralized_solution): a node must not both
  # send and receive the same function. Inductively (y starts at 0):
  # a neighbour currently offloading f is not a valid seller for f, and the
  # mover must not offload any f it holds inbound commitments for. Both rules
  # only shrink the move class, so the exact-potential argument is unaffected.
  outbound = y_trial.sum(axis=1)
  ledger[outbound > tolerance] = 0.0
  inbound_row = y_trial[:, i, :].sum(axis=0)
  omega_ub_row = np.zeros(Nf)
  for f in range(Nf):
    if inbound_row[f] > tolerance:
      continue
    gamma_if = sp_data[None]["gamma"][(i + 1, f + 1)]
    omega_ub_row[f] = sum(
      ledger[j, f] for j in neighbours
      if sp_data[None]["beta"][(i + 1, j + 1, f + 1)] > -gamma_if
    )
  x_row, r_row, omega_row, runtime = propose_fn(i, omega_ub_row)
  omega_row = np.minimum(np.array(omega_row, dtype=float), omega_ub_row)
  new_row = split_omega(i, omega_row, ledger, neighbours, sp_data)
  # candidate state (copies: commit only on acceptance)
  x_new = np.array(x, dtype=float)
  y_new = y_trial
  x_new[i, :] = x_row
  y_new[i, :, :] = new_row
  z_new = compute_z(x_new, y_new, sp_data)
  u_new = compute_node_utility(i, x_new, y_new, z_new, sp_data)
  delta_u = u_new - u_old
  accepted = delta_u > epsilon
  if accepted:
    communication.pg_commit(i, y[i, :, :], new_row)
    x[i, :] = x_row
    y[i, :, :] = new_row
    r[i, :] = r_row
  # unplaced appetite signals a capacity shortage: bid on neighbour memory.
  # The proposal's omega is capped by the current ledger, so scarcity must
  # also be read from the Cloud-forwarded residue z (load the node would
  # rather offload than reject, since delta > 0 > -gamma): without the
  # z-signal, a saturated neighbourhood never grows and coordination stalls
  # at the initial residual capacity.
  placed = new_row.sum(axis=0)
  for f in range(Nf):
    if inbound_row[f] > tolerance:
      continue  # no-ping-pong: this node can never offload f
    unmet = max(omega_row[f] - placed[f], z_new[i, f])
    if unmet <= tolerance:
      continue
    gamma_if = sp_data[None]["gamma"][(i + 1, f + 1)]
    memory_requirement = sp_data[None]["memory_requirement"][f + 1]
    for j in sorted(neighbours):
      if (
          rho[j] >= memory_requirement
          and sp_data[None]["beta"][(i + 1, j + 1, f + 1)] > -gamma_if
        ):
        memory_bids["i"].append(i)
        memory_bids["j"].append(j)
        memory_bids["f"].append(f)
  return accepted, float(delta_u), memory_bids, runtime


def potential_game_sweep(
    x: np.array,
    y: np.array,
    r: np.array,
    sp_data: dict,
    neighborhood: np.array,
    rho: np.array,
    epsilon: float,
    tolerance: float,
    order: str,
    rng: np.random.Generator,
    propose_fn: Callable,
  ) -> Tuple[int, float, pd.DataFrame, float]:
  """Gauss-Seidel sweep of better-response moves. Every accepted move raises
  the exact potential Phi by more than epsilon, so sweeps terminate."""
  Nn = sp_data[None]["Nn"][None]
  if order == "random":
    node_order = [int(i) for i in rng.permutation(Nn)]
  elif order == "fixed":
    node_order = list(range(Nn))
  else:
    raise ValueError("order must be one of: fixed, random")
  n_accepted = 0
  delta_phi = 0.0
  proposal_runtime = 0.0
  all_bids = {"i": [], "j": [], "f": []}
  for i in node_order:
    accepted, delta_u, bids, runtime = node_move(
      i, x, y, r, sp_data, neighborhood, rho, epsilon, propose_fn, tolerance,
      integer_flows=PYO_VAR_TYPE is pyo.NonNegativeIntegers,
    )
    proposal_runtime += runtime
    if accepted:
      n_accepted += 1
      delta_phi += delta_u
    for k in all_bids:
      all_bids[k].extend(bids[k])
  return n_accepted, delta_phi, pd.DataFrame(all_bids), proposal_runtime


def check_pg_stopping(
    it: int,
    max_iterations: int,
    n_accepted: int,
    n_new_replicas: int,
    total_runtime: float,
    time_limit: float,
  ) -> Tuple[bool, str]:
  if it >= max_iterations - 1:
    return True, "max iterations reached"
  if total_runtime >= time_limit:
    return True, f"reached time limit: {total_runtime} >= {time_limit}"
  if n_accepted == 0 and n_new_replicas == 0:
    # Proposals optimize delta, not the beta-aware utility used for acceptance.
    return True, "no improving proposal"
  return False, None




@parallel_solver_run
def _run(
    config: dict,
    parallelism: int,
    *,
    order: str,
    method_name: str,
    options_key: str,
    detailed_init: bool = False,
    fixed_x: bool = False,
    log_on_file: bool = False,
    disable_plotting: bool = False,
  ) -> str:
  if fixed_x and not detailed_init:
    raise ValueError("fixed_x requires LSP_detailed initialization")
  if detailed_init:
    if "opt_solution_folder" in config:
      raise ValueError("LSP_detailed initialization does not support opt_solution_folder")
    config = {**config, "algorithm": method_name, "local_reference": "LSP_detailed"}
  if fixed_x:
    config = {**config, "pg_fixed_x": True}
  options = dict(config["solver_options"].get(options_key, {}))
  return run_frame.run(
    config, parallelism, log_on_file=log_on_file, disable_plotting=disable_plotting,
    method=method_name, outputs=run_frame.LSP_OUTPUTS,
    solve_step=partial(_solve_step, detailed_init=detailed_init, fixed_x=fixed_x, epsilon=options.get("epsilon", 1e-6), order=order,
                       rng=np.random.default_rng(config["seed"])),
  )


def _solve_step(ctx, t: int, data: dict, *, epsilon: float, order: str, rng, detailed_init=False, fixed_x=False) -> run_frame.StepResult:
  """One step: sequential sweeps run on the step clock (wall time minus observers);
  the local initialization is a parallel phase."""
  Nn, Nf, tolerance = ctx.Nn, ctx.Nf, ctx.tolerance
  time_limit = ctx.time_limit
  loadt = data[None]["incoming_load"]
  fixed = ctx.reference is not None
  sp_data = ctx.step_data(data, t)
  sp = LSP_detailed() if detailed_init else LSP_fixedr() if fixed else LSP()
  pg_model = LSP_pg_fixedx() if fixed_x else LSP_pg_fixedr() if fixed else LSP_pg()
  (
    sp_data, sp_x, _, _, _, sp_r, sp_rho, _, _, _, sp_runtime
  ) = solve_subproblem(
    sp_data, ctx.agents, sp, ctx.solver_name, ctx.general_options, ctx.parallelism
  )
  x = np.array(sp_x, dtype=float)
  if fixed_x:
    sp_data[None]["x_bar"] = {(i + 1, f + 1): float(x[i, f])
                              for i in range(Nn) for f in range(Nf)}
  r = np.array(sp_r, dtype=float)
  rho = np.array(np.zeros_like(sp_rho) if fixed else sp_rho, dtype=float)
  y = np.zeros((Nn, Nn, Nf))

  def propose_fn(node, omega_ub_row):
    return propose_node_move(
      node, omega_ub_row, y, sp_data, pg_model,
      ctx.solver_name, ctx.general_options, ctx.parallelism,
    )

  it = 0
  stop_searching = False
  with step_time.untimed():  # the potential only validates monotonicity
    phi_prev = compute_centralized_objective(
      sp_data, x, y, compute_z(x, y, sp_data)
    )
  diagnostics = ctx.config.get("pg_diagnostics", False)
  if diagnostics:
    initial_runtime = step_time.current_modelled()
    with step_time.untimed():
      initial_x = x.copy()
      state_path = Path(ctx.folder) / f"pg_initial_state{'' if t == 0 else f'-{t}'}.npz"
      np.savez_compressed(state_path, x=x, r=r)
      trace = [dict(step=t, sweep=0, welfare=phi_prev, accepted_moves=0,
                    new_replicas=0, x_changed_entries=0, x_l1_from_initial=0.,
                    algorithm_s=initial_runtime, termination="initialization")]
  while not stop_searching:
    communication.begin_round(it)
    if not fixed:
      rho = compute_rho(r, sp_data)
    n_accepted, delta_phi, memory_bids, _ = (
      potential_game_sweep(
        x, y, r, sp_data, ctx.neighborhood, rho, epsilon, tolerance,
        order, rng, propose_fn,
      )
    )
    additional_replicas = np.zeros((Nn, Nf))
    communication.requests('memory_request', memory_bids)
    if len(memory_bids) > 0 and not fixed:
      # accepted moves in the sweep re-decide r and consume memory slack:
      # recompute rho before the market or it would allocate stale slack
      rho = compute_rho(r, sp_data)
      if (rho > 0).any():
        additional_replicas, rho = start_additional_replicas(
          memory_bids, r, sp_data, rho
        )
        r += additional_replicas
    with step_time.untimed():
      phi = compute_centralized_objective(
        sp_data, x, y, compute_z(x, y, sp_data)
      )
      if not (phi >= phi_prev - 1e-9):
        raise RuntimeError(f"potential decreased: {phi_prev} -> {phi}")
      phi_prev = phi
    total_runtime = step_time.current_modelled()
    stop_searching, why_stop_searching = check_pg_stopping(
      it, ctx.max_iterations, n_accepted,
      int(additional_replicas.sum()), total_runtime, time_limit,
    )
    if diagnostics:
      with step_time.untimed():
        distance = np.abs(x - initial_x)
        trace.append(dict(step=t, sweep=it + 1, welfare=phi, accepted_moves=n_accepted,
                          new_replicas=int(additional_replicas.sum()),
                          x_changed_entries=int(np.count_nonzero(distance > tolerance)),
                          x_l1_from_initial=float(distance.sum()), algorithm_s=total_runtime,
                          termination=why_stop_searching or ""))
    if not stop_searching:
      it += 1
  if diagnostics:
    with step_time.untimed():
      path = Path(ctx.folder) / "pg_diagnostics.csv"
      pd.DataFrame(trace).to_csv(path, mode="a", header=not path.exists(), index=False)
  csol, objective = run_frame.snapshot(
    sp_data, Nn, Nf, sp_data, loadt, x, r, rho,
    None, y, None, None, None, None
  )
  return run_frame.StepResult(
    solutions={"LSP": csol["sp"], "LSPc": csol["sp"]},
    objective=objective,
    termination=f"{why_stop_searching} (it: {it}; obj. deviation: {None}; best it: {it}; total runtime: {total_runtime})",
    data=sp_data,
  )




def run_pg_r(config, parallelism, log_on_file=False, disable_plotting=False, *, detailed_init=True, fixed_x=True):
  if not detailed_init or not fixed_x:
    raise ValueError("PG uses detailed initialization and fixed local service")
  method = "PG"
  return _run(config, parallelism, order="random",
              method_name=method,
              options_key="pg_r", detailed_init=detailed_init, fixed_x=fixed_x,
              log_on_file=log_on_file, disable_plotting=disable_plotting)
