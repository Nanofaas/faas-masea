from run_faasmacro import (
  parallel_solver_run,
  compute_centralized_objective,
  compute_social_welfare,
  cache_fixed_welfare,
  combine_solutions,
  solve_subproblem
)
from run_faasmadea import (
  check_stopping_criteria,
  compute_residual_capacity,
  define_bids,  # noqa: F401 (default rules, resolved at call time)
  evaluate_bids,  # noqa: F401 (default rules, resolved at call time)
  neigh_dict_to_matrix,  # noqa: F401 (re-exported: hierarchical_auction.runner)
  RULE_NAMES,
  confirm_memory_bid_batch,
  relative_objective_gap,
  start_additional_replicas,  # noqa: F401 (default rules, resolved at call time)
)
from utils.common import load_configuration
from models.sp import LSP_detailed, LSP, LSPr_x

from collections import deque
from functools import partial
import logging
from copy import deepcopy
import pandas as pd
import numpy as np
import argparse
import os
from utils import communication, run_frame, run_log, step_time
from utils.selective_memory import SelectiveMemory

log = run_log.logger(__name__)




@parallel_solver_run
def run(
    config: dict,
    parallelism: int,
    log_on_file: bool = False,
    disable_plotting: bool = False,
    *, refine_welfare: bool = False, rules: dict | None = None,
    detailed_init: bool = True, selective_memory: bool = False,
  ):
  if refine_welfare or rules is not None or not detailed_init:
    raise ValueError("Only the paper FaaS-MASeA variants are supported")
  # Shared auction rules are unchanged from the experimental implementation.
  if config.get('communication_metrics', False) and (refine_welfare or rules is not None):
    raise ValueError('communication metrics unsupported for refinement or custom rules')
  rules = rules if rules is not None else {name: globals()[name] for name in RULE_NAMES}
  if selective_memory:
    detailed_init = True
  method = "FaaS-MASeA(r)" if selective_memory else "FaaS-MASeA"
  if detailed_init:
    if "opt_solution_folder" in config:
      raise ValueError("LSP_detailed initialization does not support opt_solution_folder")
    config = {**config, "algorithm": method, "local_reference": "LSP_detailed"}
  if selective_memory:
    config = {**config, "memory_requests": "selective"}
  refinement = []  # refinement.csv rows of the whole run (PG)
  folder = run_frame.run(
    config, parallelism, log_on_file=log_on_file, disable_plotting=disable_plotting,
    method=method if detailed_init else "One-shot-PG" if refine_welfare else "FaaS-MADeA(1s)", outputs=run_frame.LSP_OUTPUTS,
    solve_step=partial(_solve_step, rules=rules, detailed_init=detailed_init, selective_memory=selective_memory, refinement=refinement if refine_welfare else None),
  )
  if refine_welfare:
    pd.DataFrame(refinement).to_csv(os.path.join(folder, "refinement.csv"), index=False)
  return folder


def _solve_step(ctx, t: int, data: dict, *, rules: dict, refinement, detailed_init=False, selective_memory=False) -> run_frame.StepResult:
  """One-shot auction rounds (a fixed reference is not used); with ``refinement``
  (PG) the bounded potential-game refinement of the centralized incumbent."""
  Nn, Nf, tolerance = ctx.Nn, ctx.Nf, ctx.tolerance
  memory_selector = SelectiveMemory() if selective_memory else None
  auction_options = ctx.solver_options["auction"]
  loadt = data[None]["incoming_load"]
  started_at = step_time.now()
  # -- solve subproblem
  sp = LSP_detailed() if detailed_init else LSP()
  spr = LSPr_x()
  sp_data = deepcopy(data)
  (
    sp_data, sp_x, _, _, sp_omega, sp_r, sp_rho, sp_U, obj, tc, sp_runtime
  ) = solve_subproblem(
    sp_data, ctx.agents, sp, ctx.solver_name, ctx.general_options, ctx.parallelism
  )
  welfare = cache_fixed_welfare(lambda y, omega: compute_social_welfare(
    spr, sp_data, ctx.agents, ctx.solver_name, ctx.general_options,
    y, omega, ctx.parallelism, sp_x,
  ))
  with step_time.untimed():
    log.debug("    sp: DONE  (%s; obj = %s; runtime = %s)", tc["tot"], obj["tot"], sp_runtime["tot"])
  # define target operating point and initial prices
  u0 = np.ones((Nn,Nf)) * 0.9
  p = np.zeros((Nn,Nf))
  # loop over iterations
  it = 0
  stop_searching = False
  best_solution_so_far = None
  best_centralized_solution = None
  best_cost_so_far = np.inf
  best_centralized_cost = -np.inf
  spr_obj = np.inf
  best_it_so_far = -1
  best_centralized_it = -1
  y = np.zeros((Nn,Nn,Nf))
  rmp_omega = np.zeros((Nn,Nf))
  omega = deepcopy(sp_omega)
  fairness = np.zeros((Nn,Nf))
  odev_queue = deque(maxlen=ctx.config.get("patience", 1))
  while not stop_searching:
    communication.begin_round(it)
    communication.announce(ctx.neighborhood, Nf)
    with step_time.untimed():
      log.info("    it = %s", it)
    # compute residual computational capacity
    s = step_time.now()
    capacity, blackboard, ell = compute_residual_capacity(
      sp_x, y, sp_r, sp_data
    )
    e = step_time.now()
    with step_time.untimed():
      if log.isEnabledFor(logging.DEBUG):
        log.debug(
          "        compute_residual_capacity: DONE  (%s; blackboard = %s; ell = %s; runtime = %s)",
          capacity.tolist(), blackboard.tolist(), ell.tolist(), e - s,
        )
    # buyers define their bids
    s = step_time.now()
    bids, memory_bids, n_auctions = rules["define_bids"](
      omega,
      blackboard,
      p,
      sp_data,
      ctx.neighborhood,
      sp_rho,
      auction_options,
      ctx.latency,
      fairness,
      False
    )
    rt = step_time.now() - s
    with step_time.untimed():
      log.debug(
        "        define_bids: DONE; runtime = %s; n_auctions = %s; tot runtime = %s)",
        rt / max(n_auctions, 1), n_auctions, rt,
      )
      if ctx.config.get("verbose", 0) > 2:  # high-volume bid dump
        log.debug("%s", bids)
    # sellers accept/reject bids
    a = np.zeros((Nn,Nf))
    if len(bids) > 0:
      communication.requests('capacity_request', bids)
      s = step_time.now()
      auction_y, p, _, _ = rules["evaluate_bids"](
        bids,
        blackboard,
        data,
        previous_y = y,
        ell = ell,
        p = p,
        total_capacity = capacity,
        u0 = u0,
        auction_options = auction_options,
        tentatively_start_replicas = False,
        may_replace_existing_assignments = False
      )
      e = step_time.now()
      with step_time.untimed():
        log.debug("        evaluate_bids: DONE; runtime = %s)", e - s)
      # update effective load, number of replicas and fairness matrix
      communication.capacity_replies(bids, auction_y)
      y += auction_y
      rmp_omega = np.zeros((Nn,Nf))
      for n in range(Nn):
        for f in range(Nf):
          rmp_omega[n,f] = y[n,:,f].sum()
          if rmp_omega[n,f] > 0:
            fairness[n,f] += 1
      # -- solve "restricted problem"
      spr_sol, spr_obj, spr_tc, spr_runtime = welfare(y, rmp_omega)
      with step_time.untimed():
        log.debug(
          "        solve 'restricted problem': DONE (%s; obj: %s; runtime = %s)",
          spr_tc, spr_obj, spr_runtime,
        )
      # -- update solution
      _, _, _, _, sp_r, sp_rho = spr_sol # x, y, z, omega, r, rho
      for i in range(Nn):
        for f in range(Nf):
          omega[i,f] = sp_omega[i,f] - rmp_omega[i,f]
          if abs(omega[i,f]) < tolerance:
            omega[i,f] = 0.0
      with step_time.untimed():
        if log.isEnabledFor(logging.DEBUG):
          log.debug(
            "        solution updated: DONE (auct_y = %s; omega = %s; x: %s; r = %s; rho = %s)",
            auction_y.tolist(), omega.tolist(), sp_x.tolist(), sp_r.tolist(), sp_rho.tolist(),
          )
    if len(bids) == 0 or memory_selector is not None:
      if memory_selector is not None:
        with step_time.parallel_phase():
          memory_bids = confirm_memory_bid_batch(memory_bids, omega, tolerance=tolerance)
        memory_bids = memory_selector.select(memory_bids, omega, p, sp_data, ctx.latency, auction_options)
      # tentatively start additional replicas
      communication.requests('memory_request', memory_bids)
      s = step_time.now()
      a, sp_rho = rules["start_additional_replicas"](
        memory_bids, sp_r, sp_data, sp_rho
      )
      sp_r += a
      e = step_time.now()
      with step_time.untimed():
        if log.isEnabledFor(logging.INFO):
          log.info(
            "        additional replicas started: DONE (a = %s; rho = %s; runtime = %s)",
            a.tolist(), sp_rho.tolist(), e - s,
          )
    with step_time.untimed():
      # merge solutions and compute the centralized objective value
      csol = combine_solutions(
        Nn, Nf, sp_data, loadt,
        sp_x, sp_r, sp_rho,
        None, y, None, None, None, None
      )
      cobj = compute_centralized_objective(
        sp_data, csol["sp"]["x"], csol["sp"]["y"], csol["sp"]["z"]
      )
      # update best solution so far
      if spr_obj < best_cost_so_far or it == 0:
        best_cost_so_far = spr_obj
        best_solution_so_far = csol
        best_it_so_far = it
        log.info("        best solution updated; obj = %s", spr_obj)
      prev_cobj = best_centralized_cost
      if cobj > best_centralized_cost:
        best_centralized_cost = cobj
        best_centralized_solution = csol
        best_centralized_it = it
        log.info("        best centralized solution updated; obj = %s", cobj)
      odev_queue.append(
        relative_objective_gap(prev_cobj, best_centralized_cost)
      )
    # check termination criteria
    total_runtime = step_time.current_modelled()
    s = step_time.now()
    stop_searching, why_stop_searching = check_stopping_criteria(
      it = it,
      max_iterations = ctx.max_iterations,
      blackboard = blackboard,
      omega = omega,
      rmp_omega = rmp_omega,
      a = a,
      odev_queue = odev_queue,
      bids = bids,
      memory_bids = memory_bids,
      tolerance = tolerance,
      total_runtime = total_runtime,
      time_limit = ctx.time_limit,
      plateau_tolerance = ctx.config.get("plateau_tolerance")
    )
    if memory_selector is not None and memory_selector.retry(it, ctx.max_iterations, total_runtime, ctx.time_limit):
      stop_searching, why_stop_searching = False, None
    e = step_time.now()
    with step_time.untimed():
      log.debug(
        "        check_stopping_criteria: DONE (runtime = %s; total runtime = %s; wallclock: %s) --> stop? %s (%s)",
        e - s, total_runtime, step_time.now() - started_at, stop_searching, why_stop_searching,
      )
    if not stop_searching:
      it += 1
  with step_time.untimed():
    central = best_centralized_solution["sp"]
    objective = compute_centralized_objective(sp_data, central["x"], central["y"], central["z"])
  return run_frame.StepResult(
    solutions={"LSP": best_solution_so_far["sp"], "LSPc": central},
    objective=objective,
    termination=(
      f"{why_stop_searching} "
      f"(it: {it}; obj. deviation: {None}; best it: {best_it_so_far}; "
      f"best centralized it: {best_centralized_it}; "
      f"total runtime: {total_runtime})"
    ),
    data=sp_data,
  )
