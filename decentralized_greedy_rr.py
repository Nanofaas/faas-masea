"""Greedy neighbor selection, round-robin service and local replica allocation.

No prices or bids: capacity requests carry (i,j,f,quantity), memory requests
only (i,j,f). A receiver never reads the sender's utility coefficients.
"""
from collections import Counter, defaultdict

import numpy as np
import pyomo.environ as pyo

from models.model import PYO_VAR_TYPE
from models.sp import LSP_detailed
from run_faasmacro import parallel_solver_run, solve_subproblem
from run_faasmadea import compute_residual_capacity, next_replica_batch
from utils import communication, run_frame, step_time


@step_time.parallel
def request_capacity(remaining, capacity, neighborhood, data):
  requests = []
  for i in step_time.timed(range(len(remaining))):
    for f in np.flatnonzero(remaining[i] > 0):
      neighbors = [int(j) for j in np.flatnonzero(neighborhood[i])
                   if j != i and capacity[j, f] >= 1]
      if neighbors:
        j = max(neighbors, key=lambda j: (data[None]['beta'][i+1, j+1, f+1], -j))
        requests.append((i, j, int(f), int(min(remaining[i, f], capacity[j, f]))))
  return requests


@step_time.parallel
def accept_requests(requests, capacity, cursors):
  groups = defaultdict(dict)
  for i, j, f, quantity in requests:
    groups[j, f][i] = quantity
  accepted = []
  for (j, f), demand in step_time.timed(groups.items(), key=lambda item: item[0][0]):
    available = int(capacity[j, f])
    order = sorted(demand, key=lambda i: (i - cursors.get((j, f), 0)) % len(capacity))
    allocated = dict.fromkeys(order, 0)
    active = list(order)
    last = None
    while available > 0 and active:
      # Bulk complete turns; at most one partial turn. Runtime is independent
      # of the number of requests when loads reach millions.
      turns = min(available // len(active), min(demand[i] for i in active))
      if turns:
        for i in active:
          allocated[i] += turns
          demand[i] -= turns
        available -= turns * len(active)
        last = active[-1]
        active = [i for i in active if demand[i] > 0]
      else:
        for i in active[:available]:
          allocated[i] += 1
          last = i
        break
    if last is not None:
      cursors[j, f] = (last + 1) % len(capacity)
    accepted.extend((i, j, f, allocated[i]) for i in order if allocated[i])
  return accepted


@step_time.parallel
def request_memory(remaining, rho, neighborhood):
  requests = []
  for i in step_time.timed(range(len(remaining))):
    for f in np.flatnonzero(remaining[i] > 0):
      for j in np.flatnonzero(neighborhood[i]):
        if j != i and rho[j] > 0:
          requests.append((i, int(j), int(f)))
  return requests


@step_time.parallel
def allocate_memory(requests, r, rho, senders, data):
  groups = defaultdict(set)
  for i, j, f in requests:
    groups[j].add((i, f))
  extra, left = np.zeros_like(r), rho.copy()
  values = data[None]
  for j, incoming in step_time.timed(groups.items(), key=lambda item: item[0]):
    counts = Counter(f for _, f in incoming if not senders[j, f])
    for f in sorted(counts, key=lambda f: (-counts[f], f)):
      capacity = values['max_utilization'][f+1] / values['demand'][j+1, f+1]
      count, _ = next_replica_batch(r[j, f], capacity)
      memory = count * values['memory_requirement'][f+1]
      if memory <= left[j]:
        extra[j, f] = count
        left[j] -= memory
        break  # one useful batch per receiver/round; demand is then re-evaluated
  return extra, left


@parallel_solver_run
def run(config, parallelism, log_on_file=False, disable_plotting=False):
  if 'opt_solution_folder' in config:
    raise ValueError('Greedy-RR does not support opt_solution_folder')
  if PYO_VAR_TYPE is not pyo.NonNegativeIntegers:
    raise ValueError('Greedy-RR requires integer request flows')
  config = {**config, 'algorithm': 'Greedy-RR', 'local_reference': 'LSP_detailed'}
  return run_frame.run(config, parallelism, log_on_file=log_on_file,
    disable_plotting=disable_plotting, method='Greedy-RR',
    outputs=run_frame.LSP_OUTPUTS, solve_step=_solve_step)


def _solve_step(ctx, t, data):
  sp_data, x, _, _, _, r, rho, *_ = solve_subproblem(
    data, ctx.agents, LSP_detailed(), ctx.solver_name, ctx.general_options, ctx.parallelism)
  values = sp_data[None]
  load = np.array([[values['incoming_load'][i+1, f+1] for f in range(ctx.Nf)]
                   for i in range(ctx.Nn)], dtype=float)
  remaining = load - x
  # Roles are fixed for the whole physical step; zero advertised capacity for
  # senders and local rejection of their memory requests prevent ping-pong.
  senders = remaining > ctx.tolerance
  y = np.zeros((ctx.Nn, ctx.Nn, ctx.Nf))
  cursors = {}
  reason, rounds = 'max iterations reached', 0
  for it in range(ctx.max_iterations):
    if step_time.current_modelled() >= ctx.time_limit:
      reason = 'reached time limit'
      break
    rounds = it + 1
    communication.begin_round(it)
    communication.announce(ctx.neighborhood, ctx.Nf)
    _, capacity, _ = compute_residual_capacity(x, y, r, sp_data)
    capacity = np.floor(capacity)
    capacity[senders] = 0
    requests = request_capacity(remaining, capacity, ctx.neighborhood, sp_data)
    communication.requests('capacity_request', requests)
    accepted = accept_requests(requests, capacity, cursors)
    communication.capacity_replies(requests)
    for i, j, f, quantity in accepted:
      y[i, j, f] += quantity
      remaining[i, f] -= quantity
    if not (remaining > ctx.tolerance).any():
      reason = 'all load assigned'
      break
    if step_time.current_modelled() >= ctx.time_limit:
      reason = 'reached time limit'
      break
    if rounds == ctx.max_iterations:
      break  # no unused replicas when there is no next service round
    requests = request_memory(remaining, rho, ctx.neighborhood)
    communication.requests('memory_request', requests)
    extra, rho = allocate_memory(requests, r, rho, senders, sp_data)
    r += extra
    if not accepted and not extra.any():
      reason = 'no feasible capacity or replica addition'
      break
  central, objective = run_frame.snapshot(
    sp_data, ctx.Nn, ctx.Nf, sp_data, values['incoming_load'], x, r, rho,
    None, y, None, None, None, None)
  return run_frame.StepResult(
    solutions={'LSP': central['sp'], 'LSPc': central['sp']}, objective=objective,
    termination=(f'{reason} (it: {max(0, rounds-1)}; obj. deviation: None; best it: {max(0, rounds-1)}; '
                 f'total runtime: {step_time.current_modelled()})'), data=sp_data)
