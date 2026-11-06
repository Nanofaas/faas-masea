"""Centralized reference (LoadManagementModel) for one sampled instance, step 0.

With ``selfish``, SelfishLoadManagementModel: every node first solves its own
problem (DP, HiGHS where DP does not apply) and the central solve must leave
it at least that local gain.

Solved with HiGHS (appsi) under a wall-clock limit; records the incumbent
welfare (same metric as the decentralized runs, ``compute_centralized_objective``
on a solution checked by ``validate_centralized_solution``), the dual bound
and the termination status.  The gap to the bound tells how far the
incumbent can be from the true optimum when the time limit is hit.
"""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import time

import highspy
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.appsi.solvers import Highs

from remote_experiments.instances import load_materialized_instance
from models.model import LoadManagementModel, _solver_option_name
from models.selfish import SelfishLoadManagementModel
from run_centralized_model import get_current_load, update_data
from utils.centralized import validate_centralized_solution
from utils.faasmacro import compute_centralized_objective
from utils import run_log

log = run_log.logger(__name__)


def _array(var, shape):
  return np.array([pyo.value(var[idx]) for idx in var], dtype=float).reshape(shape)


def solve(path, time_limit, threads=1, gap=1e-6, selfish=False, log_file=None):
  data, traces, agents, graph = load_materialized_instance(path)
  data = update_data(data, {'incoming_load': get_current_load(traces, list(agents), 0)})
  Nn, Nf = data[None]['Nn'][None], data[None]['Nf'][None]
  started = time.perf_counter()
  deadline = started + time_limit
  model = SelfishLoadManagementModel() if selfish else LoadManagementModel()
  instance = model.generate_instance(data)
  local = 0.0
  timed_out = False
  if selfish:
    local_started = time.perf_counter()
    try:
      model.compute_local_gain_floors(
        instance, {'use_dp': True}, 'appsi_highs', deadline=deadline,
      )
    except TimeoutError:
      timed_out = True
    local = time.perf_counter() - local_started
  build = time.perf_counter() - started
  result = None
  if not timed_out and build < time_limit:
    # HiGHS keeps one thread scheduler per process: an earlier solve (e.g. a local floor that fell
    # back to HiGHS) would make this one, with its own thread count, fail
    highspy.Highs.resetGlobalScheduler(True)
    opt = Highs()
    opt.config.time_limit = max(0.0, deadline - time.perf_counter())
    opt.config.load_solution = False
    opt.config.stream_solver = False
    opt.highs_options.update({'threads': threads, 'mip_rel_gap': gap, 'random_seed': 0})
    if log_file:  # the solver's own log, written as it runs: a killed solve keeps its incumbents
      opt.highs_options[_solver_option_name('appsi_highs', 'LogFile')] = log_file
    result = opt.solve(instance)
  wall = time.perf_counter() - started
  row = dict(model=model.name, nodes=Nn, functions=Nf,
             termination=str(result.termination_condition) if result else 'TerminationCondition.maxTimeLimit',
             mip_rel_gap=gap, solver_log=log_file, build_seconds=build, local_reference_seconds=local, wall_seconds=wall,
             bound=result.best_objective_bound if result else None,
             solver_objective=result.best_feasible_objective if result else None,
             input_hash=hashlib.sha256((Path(path) / 'metadata.json').read_bytes()).hexdigest())
  if result is None or result.best_feasible_objective is None:
    return dict(row, status='no_solution', welfare=None, feasible=False)
  result.solution_loader.load_vars()
  x = np.rint(_array(instance.x, (Nn, Nf)))
  y = np.rint(_array(instance.y, (Nn, Nn, Nf)))
  z = np.rint(_array(instance.z, (Nn, Nf)))
  r = np.rint(_array(instance.r, (Nn, Nf)))
  validate_centralized_solution(x, y, z, r, data)
  welfare = compute_centralized_objective(data, x, y, z)
  return dict(row, status='ok', welfare=welfare, feasible=True,
              rel_gap_to_bound=(row['bound'] - welfare) / abs(row['bound']) if row['bound'] else None)
