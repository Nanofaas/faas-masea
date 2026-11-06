"""Exact node-local flow/replica DP, independent of backend selection.

Pyomo remains the oracle and fallback for custom models, continuous flows,
unsupported parameters, infeasible data and DP instances that are too large.
LSP_detailed has no destination capacities: its aggregate offload reward is
max_j beta[i,j,f] over outgoing neighbors, or offloading is disabled if isolated.
"""
from math import ceil, floor, isfinite, gcd
from functools import reduce
from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo

from models import sp as sp_models
from utils import step_time
from plasma.core.sbm import minimize_replica_costs

_SUPPORTED = {
  sp_models.LSP, sp_models.LSP_detailed, sp_models.LSP_v0, sp_models.LSP_fixedr,
  sp_models.LSP_fixedr_v0, sp_models.LSP_capped, sp_models.LSP_capped_fixedr,
  sp_models.LSP_pg, sp_models.LSP_pg_fixedr, sp_models.LSP_pg_fixedx,
  sp_models.LSPr, sp_models.LSPr_v0, sp_models.LSPr_x, sp_models.LSPr_fixedr,
}


@dataclass
class PreparedLocal:
  costs: list
  levels: list
  memory: list
  ram_cap: int
  direct: bool
  detailed: bool
  restricted: bool
  reject: bool
  destinations: list
  nn: int


def _prepare_local_parameters(model, data):
  """Validate and parse local inputs without building level curves."""
  if type(model) not in _SUPPORTED or sp_models.PYO_VAR_TYPE is not pyo.NonNegativeIntegers:
    return None
  name = model.name
  detailed = type(model) is sp_models.LSP_detailed
  restricted = name.startswith('LSPr')
  fixed_x = name in ('LSPr_x', 'LSP_pg_fixedx')
  fixed_r = '_fixedr' in name
  reject = not name.endswith('_v0')
  capped = 'capped' in name or name.startswith('LSP_pg')
  inbound_enabled = restricted or name.startswith('LSP_pg')
  lower_bound = not fixed_r or restricted or not reject
  v = data[None]
  node, nf = v['whoami'][None], v['Nf'][None]
  neighbors = []
  if detailed:
    for j in range(1, v['Nn'][None] + 1):
      edge = v.get('neighborhood', {}).get((node, j), 0)
      if edge not in (0, 1):
        return None
      if edge:
        neighbors.append(j)
  required = {
    'incoming_load': [(node, f) for f in range(1, nf + 1)],
    'demand': [(node, f) for f in range(1, nf + 1)],
    'memory_requirement': list(range(1, nf + 1)),
    'memory_capacity': [node],
  }
  if restricted:
    required['omega_bar'] = [(node, f) for f in range(1, nf + 1)]
    required['y_bar'] = [(m, node, f) for m in range(1, v['Nn'][None] + 1)
                         for f in range(1, nf + 1)]
  if fixed_x:
    required['x_bar'] = [(node, f) for f in range(1, nf + 1)]
  if name == 'LSPr_fixedr':
    required['r_bar'] = [(node, f) for f in range(1, nf + 1)]
  if any(key not in v.get(param, {}) for param, keys in required.items() for key in keys):
    return None
  memory = [v['memory_requirement'][f] for f in range(1, nf + 1)]
  ram = v['memory_capacity'][node]
  if (not nf or not isfinite(ram) or ram < 0 or ram != int(ram)
      or any(not isfinite(m) or m <= 0 or m != int(m) for m in memory)):
    return None
  parameters, destinations, levels_count = [], [], 0
  for f in range(1, nf + 1):
    load = v['incoming_load'][node, f]
    demand = v['demand'][node, f]
    utilization = v.get('max_utilization', {}).get(f, .8)
    alpha = v.get('alpha', {}).get((node, f), 1.)
    delta = v.get('delta', {}).get((node, f), .9)
    if detailed:
      rewards = [v.get('beta', {}).get((node, j, f), .9)
                 for j in range(1, v['Nn'][None] + 1)]
      if not rewards or any(not isfinite(b) or b < 0 for b in rewards):
        return None
      destination = max(neighbors, key=lambda j: rewards[j - 1], default=None)
      delta = rewards[destination - 1] if destination is not None else 0.
      destinations.append(destination)
    gamma = v.get('gamma', {}).get((node, f), .1) if reject else 0.
    pi = 0. if restricted else v.get('pi', {}).get(f, 0.)
    if (any(not isfinite(a) or a < 0 for a in (load, alpha, delta, gamma, pi))
        or (not restricted and load != round(load)) or not isfinite(demand) or demand <= 0
        or not isfinite(utilization) or utilization <= 0):
      return None
    capacity = utilization / demand
    if not isfinite(capacity) or capacity <= 0:
      return None
    inbound = (sum(v.get('y_bar', {}).get((m, node, f), 0.)
                   for m in range(1, v['Nn'][None] + 1)) if inbound_enabled else 0.)
    cap = v.get('omega_ub', {}).get(f, 1e9) if capped else load
    if detailed and not neighbors:
      cap = 0
    if not isfinite(inbound) or inbound < 0 or not isfinite(cap) or cap < 0:
      return None
    ub = min(int(load), floor(cap + 1e-9))
    omega_fixed = None
    if restricted:
      omega_bar = float(v['omega_bar'][node, f])
      if not isfinite(omega_bar) or not -1e-9 <= omega_bar <= load + 1e-9:
        return None
      # coordination arithmetic leaves float noise (e.g. load + 4e-15)
      omega_fixed = min(max(omega_bar, 0.), load)
    x_fixed = v['x_bar'][node, f] if fixed_x else 0.
    if fixed_x and (not isfinite(x_fixed) or x_fixed < 0 or x_fixed != round(x_fixed)):
      return None
    max_x = load - omega_fixed if restricted else load
    hi = (0 if fixed_x else min(int(ram // memory[f - 1]),
                                ceil((max_x + inbound) / capacity) + 1))
    pinned = v.get('r_bar', {}).get((node, f), 0) if fixed_r else None
    if fixed_r and (not isfinite(pinned) or pinned < 0 or pinned != int(pinned)):
      return None
    hi = int(pinned) if fixed_r else hi
    budget = int(ram) // reduce(gcd, map(int, memory))
    if not fixed_r and not fixed_x and (budget + 1) * (levels_count + hi + 1) > 2_000_000:
      return None
    levels_count += 1 if fixed_r or fixed_x else hi + 1
    parameters.append((load, capacity, alpha, delta, gamma, pi, inbound, ub,
                       omega_fixed if restricted else 0., max_x, hi, x_fixed,
                       lower_bound, fixed_x, fixed_r))
  return (PreparedLocal([], [], memory, ram, fixed_x or fixed_r,
                        detailed, restricted, reject, destinations, v['Nn'][None]), parameters)


def prepare_local_problem(model, data):
  """Prepare the original cost curves without solving or mutating inputs."""
  parsed = _prepare_local_parameters(model, data)
  if parsed is None:
    return None
  problem, parameters = parsed
  levels, costs, memory, ram = problem.levels, problem.costs, problem.memory, problem.ram_cap
  restricted, reject = problem.restricted, problem.reject
  fixed_x = model.name in ('LSPr_x', 'LSP_pg_fixedx')
  fixed_r = '_fixedr' in model.name
  for f, row in enumerate(parameters, 1):
    (load, capacity, alpha, delta, gamma, pi, inbound, ub, omega_fixed,
     max_x, hi, x_fixed, lower_bound, _, _) = row
    v, node = data[None], data[None]['whoami'][None]
    denominator = load or 1

    def allocation(x, r):
      omega = (omega_fixed if restricted else load - x if not reject
               else min(load - x, ub) if delta - pi + gamma > 0 else 0)
      z = load - x - omega
      if z < -1e-9 or (reject and not restricted and abs(z - round(z)) > 1e-9):
        return None
      cost = -(alpha * x + (delta - pi) * omega - gamma * z) / denominator
      return cost, (x, omega, z, r)

    if fixed_x:
      x = v['x_bar'][node, f]
      if not isfinite(x) or x < 0 or x != round(x):
        return None
      r = max(0, ceil((x + inbound) / capacity - 1e-9))
      candidate = allocation(x, r)
      if candidate is None:
        return None
      levels.append([candidate[1]])
      costs.append([candidate[0]])
      continue
    choices, curve = [], []
    for r in ([hi] if fixed_r else range(hi + 1)):
      lo_x = max(0, ceil((r - 1) * capacity - inbound - 1e-9)) if lower_bound else 0
      hi_x = min(floor(max_x + 1e-9), floor(r * capacity - inbound + 1e-9))
      candidates = {lo_x, hi_x}
      if not restricted:
        candidates.add(max(lo_x, min(hi_x, load - ub)))
      best = None
      # Linear pieces: extrema and the offload-cap breakpoint are sufficient.
      for x in sorted(candidates, reverse=True):
        candidate = allocation(x, r) if lo_x <= x <= hi_x else None
        if candidate is not None and (best is None or candidate[0] < best[0]):
          best = candidate
      choices.append(best[1] if best else None)
      curve.append(best[0] if best else np.inf)
    levels.append(choices)
    costs.append(curve)
  if fixed_x or fixed_r:
    chosen = [a[0] for a in levels]
    if (any(a is None for a in chosen)
        or sum(a[3] * m for a, m in zip(chosen, memory)) > ram):
      return None
  return problem


def finish_local_solution(problem, choices, *, runtime):
  """Decode valid alternative indices using the common CPU flow reconstruction."""
  chosen = [problem.levels[f][r] for f, r in enumerate(choices)]
  objective = sum(problem.costs[f][r] for f, r in enumerate(choices))
  detailed, restricted, reject = problem.detailed, problem.restricted, problem.reject
  destinations, nf = problem.destinations, len(problem.costs)
  x, omega, z, r = map(list, zip(*chosen))
  result = dict(x=x, r=r, obj=objective, solver_status='ok', solution_exists=True,
                termination_condition='optimal', runtime=runtime)
  if detailed:
    result['obj'] = -objective
    result['y'] = [omega[f] if j == destinations[f] else 0
                   for j in range(1, problem.nn + 1) for f in range(nf)]
  elif not restricted:
    result['omega'] = omega
  if reject:
    result['z'] = z
  return result


def try_solve_local(model, data):
  """Return an exact local solution, or None to retain the Pyomo solve path."""
  started = step_time.now()
  problem = prepare_local_problem(model, data)
  if problem is None:
    return None
  if problem.direct:
    choices = [0] * len(problem.costs)
  else:
    try:
      choices = minimize_replica_costs(problem.costs, problem.memory, problem.ram_cap)
    except ValueError:
      return None
  result = finish_local_solution(problem, choices, runtime=0.)
  result['runtime'] = step_time.now() - started
  return result
