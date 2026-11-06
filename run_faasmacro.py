from run_centralized_model import (
  compute_utilization,
  extract_solution,
  join_complete_solution
)
from utils.centralized import (
  validate_centralized_solution,
  encode_solution,
)
from utils.faasmacro import (
  compute_centralized_objective,
  relative_objective_gap
)
from utils.common import load_configuration
from postprocessing import plot_history

from models.local_sp import solve_agent_problem, solve_agent_batch, dp_device
from models.sp import (
  SPAbstractModel,
  LSP,
  LSPr,
  LSP_fixedr,
  LSP_v0,
  LSPr_v0,
  LSP_fixedr_v0
)
from utils import run_frame, run_log, step_time

import multiprocessing as mpp
from contextlib import contextmanager
from contextvars import ContextVar
from collections import deque
from copy import deepcopy
from functools import partial, wraps
from typing import Tuple
import numpy as np
import argparse
import os


log = run_log.logger(__name__)

# Globals for each parallel worker
_sp_data = None
_solver_options = None
_solver_name = None
_sp = None
_parallel_session: ContextVar[dict | None] = ContextVar('parallel_sp_session', default=None)


@contextmanager
def parallel_solver_session():
  """Own one lazy process pool; nested runners reuse it until the outer run ends."""
  current = _parallel_session.get()
  if current is not None:
    yield current
    return
  state: dict = {'pool': None, 'workers': 0}
  token = _parallel_session.set(state)
  try:
    yield state
  except BaseException:
    if state['pool'] is not None:
      state['pool'].terminate()
    raise
  else:
    if state['pool'] is not None:
      state['pool'].close()
  finally:
    if state['pool'] is not None:
      state['pool'].join()
    _parallel_session.reset(token)


def parallel_solver_run(runner):
  """Keep local-solver workers warm across all batches and timesteps of a run."""
  @wraps(runner)
  def wrapped(*args, **kwargs):
    with parallel_solver_session():
      return runner(*args, **kwargs)
  return wrapped


def _available_cpu_count():
  try:
    return max(1, len(os.sched_getaffinity(0)))
  except (AttributeError, OSError):
    return os.cpu_count() or 1


def _solve_parallel_batch(task):
  data, options, solver, model, agents, prices = task
  # Each task carries the current snapshot, even if this worker ran another model.
  init_parallel_worker(data, options, solver, model)
  out = []
  for agent in agents:
    started = step_time.now()  # the node's own computation, without transfer
    agent_id, result = solve_single_agent(agent, prices)
    out.append((agent_id, result, step_time.now() - started))
  return out


def _solve_agents_parallel(data, agents, model, solver, options, parallelism, prices=None):
  agents = list(agents)
  if not agents:
    return {}
  state = _parallel_session.get()
  if state is None:
    with parallel_solver_session():
      return _solve_agents_parallel(data, agents, model, solver, options, parallelism, prices)
  if state['pool'] is None:
    available = _available_cpu_count() if parallelism < 0 else parallelism
    state['workers'] = max(1, min(available, len(agents)))
    state['pool'] = mpp.Pool(processes=state['workers'])
  count = min(state['workers'], len(agents))
  # One fresh snapshot per chunk, rather than serializing the network per node.
  tasks = [(data, options, solver, model, agents[i::count], prices) for i in range(count)]
  with step_time.offloaded(state['workers']):
    results = state['pool'].map(_solve_parallel_batch, tasks, chunksize=1)
    for batch in results:
      for agent, _, seconds in batch:
        step_time.record(agent, seconds)
  by_agent = {agent: result for batch in results for agent, result, _ in batch}
  return {agent: by_agent[agent] for agent in agents}




def check_stopping_criteria(
    it: int,
    max_iterations: int,
    sp_omega: np.array,
    rmp_omega: np.array,
    pi_queue: deque,
    dev_queue: deque,
    sw_queue: deque,
    current_sw_queue: deque,
    odev_queue: deque,
    psi: float,
    tolerance: float,
    total_runtime: float,
    time_limit: float
  ) -> Tuple[bool, str]:
  stop = False
  why_stopping = None
  if it >= max_iterations - 1:
    stop = True
    why_stopping = "max iterations reached"
  elif (abs(sp_omega - rmp_omega) <= tolerance).all():
    stop = True
    why_stopping = "SP and RMP solutions are equal"
  elif psi < 1e-12:
    stop = True
    why_stopping = "psi < tol"
  elif (dev_queue[-1] < tolerance).all():
    stop = True
    why_stopping = "dev < tol"
  elif total_runtime >= time_limit:
    stop = True
    why_stopping = f"reached time limit: {total_runtime} >= {time_limit}"
  if not stop and len(odev_queue) >= odev_queue.maxlen:
    stop = True
    why_stopping = "UB/LB diff < tol"
    for odev in odev_queue:
      if odev >= tolerance:
        stop = False
        break
  #if not stop and len(current_sw_queue) >= current_sw_queue.maxlen:
  #  last_sw = None
  #  for sw in current_sw_queue:
  #    if last_sw is not None and sw < last_sw:
  #      stop = True
  #      why_stopping = "SW starts decreasing"
  #      break
  #    last_sw = sw
  if not stop and len(sw_queue) >= sw_queue.maxlen:
    stop = True
    why_stopping = "SW deviation < tol"
    last_sw = None
    for sw in sw_queue:
      if last_sw is not None and abs(sw - last_sw) >= tolerance:
        stop = False
        break
      last_sw = sw
  if not stop:
    if len(pi_queue) >= pi_queue.maxlen and len(dev_queue) >= dev_queue.maxlen:
      stop = True
      # check whether all prices in pi keep increasing
      last_pi = None
      for pi in pi_queue:
        if last_pi is not None:
          for k, v in last_pi.items():
            if pi[k] < v:
              stop = False
              break
        last_pi = pi
      # check whether the deviation is constantly >= 0
      if stop:
        for dev in dev_queue:
          if (dev < 0).any():
            stop = False
            break
      if stop:
        why_stopping = "no pi/dev improvements in the last iterations"
  # update psi if it is not time for stopping
  new_psi = psi
  if not stop and len(current_sw_queue) == current_sw_queue.maxlen:
    last_sw = None
    discount_factor = 0.5
    for sw in current_sw_queue:
      if last_sw is not None:
        if sw < last_sw:
          discount_factor = 1
          break
      last_sw = sw
    new_psi = psi * discount_factor
    if new_psi < psi:
      current_sw_queue.clear()
  return stop, why_stopping, new_psi


def compute_deviation(
    rmp_data: dict,
    sp_x: np.array,
    sp_omega: np.array,
    rmp_r: np.array,
    rmp_omega: np.array
  ) -> Tuple[np.array, list, np.array]:
  Nn = rmp_data[None]["Nn"][None]
  Nf = rmp_data[None]["Nf"][None]
  dev = np.array([sp_omega[:,f].sum() for f in range(Nf)])
  detailed_dev = np.zeros((Nn,Nf))
  Nfthr = []
  for f in range(Nf):
    n_f = 0
    for n in range(Nn):
      d = rmp_data[None]["demand"][n+1,f+1]
      u = rmp_data[None]["max_utilization"][f+1]
      n_f += (u * rmp_r[n,f] / d - sp_x[n,f])
      detailed_dev[n,f] = sp_omega[n,f] - rmp_omega[n,f]
    # deviation
    dev[f] -= n_f
    # if abs(dev[f]) < tolerance:
    #   dev[f] = 0
    Nfthr.append(n_f)
  return dev, Nfthr, detailed_dev


def compute_lower_bound(
    sp_obj: float, pi: dict, Nfthr: list, loadt: dict, Nn: int, sp_omega: np.array
  ) -> float:
  highest_cost = 0
  for f, pi_f in pi.items():
    highest_cost += pi_f# * tot_workload[f]#sp_omega[:,f-1].sum()#Nfthr[f-1]#
  return (sp_obj - highest_cost)


def cache_fixed_welfare(solve):
  """Cache the last optimal solve(y, omega) for one step with fixed data and x.

  Create a new wrapper whenever the local problem changes. Copy both the key
  and the solution: auction rounds mutate flows and tentative replica counts.
  """
  cached = None

  def evaluate(y, omega):
    nonlocal cached
    if (cached is not None and np.array_equal(y, cached[0])
        and np.array_equal(omega, cached[1])):
      solution, objective, condition = deepcopy(cached[2])
      return solution, objective, condition, 0.0
    cached = None
    result = solve(y, omega)
    if set(str(result[2]).split('-')) == {'optimal'}:
      cached = (y.copy(), omega.copy(), deepcopy(result[:3]))
    return result

  return evaluate


def compute_social_welfare(
    spr: SPAbstractModel,
    data: dict,
    agents: list,
    solver_name: str,
    solver_options: dict,
    rmp_y: np.array,
    rmp_omega: np.array,
    parallelism: int,
    sp_x: np.array = None
  ) -> Tuple[list, float, list]:
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  # solve restricted SP
  spr_data = deepcopy(data)
  spr_data[None]["y_bar"] = {
    (n1+1,n2+1,f+1): max(rmp_y[n1,n2,f], 0) \
      for n1 in range(Nn) \
        for n2 in range(Nn) \
          for f in range(Nf)
  }
  spr_data[None]["omega_bar"] = {
    (n+1,f+1): max(rmp_omega[n,f], 0) for n in range(Nn) for f in range(Nf)
  }
  if sp_x is not None:
    spr_data[None]["x_bar"] = {
      (n+1,f+1): max(sp_x[n,f], 0) for n in range(Nn) for f in range(Nf)
    }
  # solve for all agents
  agents_sol = {}
  with step_time.parallel_phase():
    from models.local_sp import compute_batch_enabled
    if dp_device(solver_options) == 'cuda' or compute_batch_enabled(spr, solver_options):
      agents_sol = solve_agent_batch(spr_data, agents, spr, solver_options, solver_name)
    elif parallelism != 0:
      agents_sol = _solve_agents_parallel(
        spr_data, agents, spr, solver_name, solver_options, parallelism,
      )
    else:
      for agent in agents:
        with step_time.task(agent):
          spr_data[None]["whoami"] = {None: agent + 1}
          agents_sol[agent] = solve_agent_problem(
            spr, spr_data, solver_options, solver_name
          )
  # merge solutions
  spr_sol = merge_agents_solutions(
    spr_data, agents_sol
  )
  return (
    list(spr_sol[:-3]), # x, y, z, omega, r, rho
    spr_sol[-3]["tot"], # obj
    spr_sol[-2]["tot"], # termination condition
    spr_sol[-1]["tot"]  # runtime
  )


def combine_solutions(
    Nn: int, Nf: int, sp_data: dict, loadt: dict,
    sp_x: np.array, spr_r: np.array, sp_rho: np.array,
    rmp_x: np.array, rmp_y: np.array, rmp_z: np.array,
    rmp_r: np.array, rmp_xi: np.array, rmp_rho: np.array
  ):
  sp_y = rmp_y.copy()
  # -- compute xi
  sp_xi = np.zeros((Nn,Nn,Nf))
  for n1 in range(Nn):
    for n2 in range(Nn):
      for f in range(Nf):
        sp_xi[n2,n1,f] = sp_y[n1,n2,f]
  # -- tighten replicas to the centralized replica equilibrium
  # r = ceil(utilization / max_utilization) for the realized served load
  # (local processing + received offload). A heuristic may greedily over-acquire
  # replicas to be able to receive offload that neighbour/no-ping-pong
  # restrictions then keep from arriving; r does not enter the welfare objective,
  # so tightening it leaves the objective unchanged while making the solution
  # centrally feasible (utilization_equilibrium / utilization_equilibrium2).
  demand = np.array([
    [sp_data[None]["demand"][(n + 1, f + 1)] for f in range(Nf)]
    for n in range(Nn)
  ])
  max_utilization = np.array([
    sp_data[None]["max_utilization"][f + 1] for f in range(Nf)
  ])
  served = sp_x + sp_y.sum(axis = 0)
  with np.errstate(divide = "ignore", invalid = "ignore"):
    spr_r = np.ceil(demand * served / max_utilization - 1e-9).astype(int)
  spr_r = np.maximum(spr_r, 0)
  # -- compute utilization
  # These are total replicas, even when sp_data contains fixed r_bar inputs.
  spr_U = np.divide(
    demand * served, spr_r, out=np.zeros_like(served, dtype=float), where=spr_r > 0,
  )
  memory = np.array([sp_data[None]["memory_requirement"][f + 1] for f in range(Nf)])
  capacity = np.array([sp_data[None]["memory_capacity"][n + 1] for n in range(Nn)])
  sp_rho = capacity - spr_r @ memory
  rmp_U = None
  if rmp_x is not None and rmp_r is not None:
    rmp_U = np.divide(
      demand * (rmp_x + sp_y.sum(axis=0)), rmp_r,
      out=np.zeros_like(served, dtype=float), where=rmp_r > 0,
    )
  # -- compute rejections
  sp_z = np.zeros((Nn,Nf))
  for (n,f), l in loadt.items():
    sp_z[n-1,f-1] = (
      l - sp_x[n-1,f-1] - sp_y[n-1,:,f-1].sum()
    )
  solution = {
    "sp": {
      "x":    sp_x,
      "y":    sp_y,
      "z":    sp_z,
      "r":    spr_r,
      "xi":   sp_xi,
      "rho":  sp_rho,
      "U":    spr_U
    },
    "rmp": {
      "x":    rmp_x,
      "y":    sp_y,
      "z":    rmp_z,
      "r":    rmp_r,
      "xi":   rmp_xi,
      "rho":  rmp_rho,
      "U":    rmp_U
    }
  }
  validate_centralized_solution(
    solution["sp"]["x"], solution["sp"]["y"], solution["sp"]["z"],
    solution["sp"]["r"], sp_data,
  )
  return solution


def init_parallel_worker(
    sp_data: dict, solver_options: dict, solver_name: str, sp: SPAbstractModel
  ):
  """Initializer for each worker: store common data as globals."""
  global _sp_data, _solver_options, _solver_name, _sp
  # Each worker process will get its own copy (once)
  _sp_data = sp_data
  _solver_options = solver_options
  _solver_name = solver_name
  _sp = sp


def merge_agents_solutions(
    data: dict, agents_sol: dict, approx_tol: float = 1e-6
  ) -> Tuple[
    np.array, np.array, np.array, np.array, np.array, np.array, dict, dict
  ]:
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  x = np.zeros((Nn,Nf))
  y = np.zeros((Nn,Nn,Nf))
  z = np.zeros((Nn,Nf))
  omega = np.zeros((Nn,Nf))
  r = np.zeros((Nn,Nf))
  rho = np.zeros((Nn,))
  temp_data = {
    None: {
      "Nn": {None: 1},
      "Nf": data[None]["Nf"],
      "memory_capacity": data[None]["memory_capacity"],
      "memory_requirement": data[None]["memory_requirement"]
    }
  }
  obj_dict = {}
  tc_dict = {}
  runtime_dict = {}
  # loop over all agents
  for agent, agent_solution in agents_sol.items():
    temp_data["indices"] = [agent]
    # -- variables
    a_x, a_y, a_z, a_r, _, a_omega, a_rho, a_obj = extract_solution(
      temp_data, agent_solution, approx_tol = approx_tol
    )
    if a_x is not None:
      x[agent,:] = a_x
    if a_y is not None:
      y[agent,:,:] = a_y
    if a_z is not None:
      z[agent,:] = a_z
    if a_r is not None:
      r[agent,:] = a_r
    if a_omega is not None:
      omega[agent,:] = a_omega
    if a_rho is not None:
      rho[agent] = a_rho[0]
    # -- termination condition
    a_tc = agent_solution["termination_condition"]
    tc_dict[agent] = a_tc
    # -- objective function value
    obj_dict[agent] = a_obj
    # -- runtime
    runtime_dict[agent] = agent_solution["runtime"]
  # "total" objective function value, termination condition and runtime
  # (NOTE: the total runtime is the average runtime among agents)
  obj_dict["tot"] = sum(list(obj_dict.values()))
  tc_dict["tot"] = "-".join(tc_dict.values())
  runtime_dict["tot"] = sum(list(runtime_dict.values())) / len(agents_sol)
  return x, y, z, omega, r, rho, obj_dict, tc_dict, runtime_dict






def solve_single_agent(agent: int, detailed_pi=None):
  """Function run in each worker, uses the global data initialized above."""
  # Make a local copy of sp_data to modify safely
  local_data = _sp_data.copy()
  local_data[None] = _sp_data[None].copy()
  local_data[None]["whoami"] = {None: agent + 1}
  if detailed_pi is not None:
    local_data[None]["pi"] = {
      f+1: price for f, price in enumerate(detailed_pi[agent])
    }
  result = solve_agent_problem(_sp, local_data, _solver_options, _solver_name)
  return agent, result


def solve_subproblem(
    base_instance_data: dict,
    agents: list,
    sp: SPAbstractModel,
    solver_name: str,
    solver_options: dict,
    parallelism: int,
    pi: dict = None,
    detailed_pi: np.array = None
  ):
  Nf = base_instance_data[None]["Nf"][None]
  # update data
  sp_data = deepcopy(base_instance_data)
  if pi is not None:
    sp_data[None]["pi"] = pi
  else:
    sp_data[None]["pi"] = {f+1: 0 for f in range(Nf)}
  # solve for all agents
  agents_sol = {}
  with step_time.parallel_phase():
    from models.local_sp import compute_batch_enabled
    if dp_device(solver_options) == 'cuda' or compute_batch_enabled(sp, solver_options):
      agents_sol = solve_agent_batch(
        sp_data, agents, sp, solver_options, solver_name, detailed_pi,
      )
    elif parallelism != 0:
      agents_sol = _solve_agents_parallel(
        sp_data, agents, sp, solver_name, solver_options, parallelism, detailed_pi,
      )
    else:
      for agent in agents:
        with step_time.task(agent):
          agent_data = sp_data.copy()
          agent_data[None] = sp_data[None].copy()
          agent_data[None]["whoami"] = {None: agent + 1}
          if detailed_pi is not None:
            agent_data[None]["pi"] = {f+1: detailed_pi[agent,f] for f in range(Nf)}
          agents_sol[agent] = solve_agent_problem(
            sp, agent_data, solver_options, solver_name
          )
  # merge solutions
  (
    sp_x,
    sp_y,
    sp_z,
    sp_omega,
    sp_r,
    sp_rho,
    obj,
    tc,
    runtime
  ) = merge_agents_solutions(
    sp_data, agents_sol, approx_tol=solver_options.get("FeasibilityTol", 1e-6)
  )
  sp_solution = {
    "x": sp_x,
    "y": sp_y,
    "r": sp_r,
    "obj": obj["tot"]
  }
  sp_U = compute_utilization(sp_data, sp_solution)
  return (
    sp_data, sp_x, sp_y, sp_z, sp_omega, sp_r, sp_rho, sp_U, obj, tc, runtime
  )






OUTPUTS = {
  "LSP": run_frame.Output(checkpoint=True, plot_filename="sp.png"),
  "LRMP": run_frame.Output(checkpoint=False, check=False),
  "LSPc": run_frame.Output(checkpoint=True),
  "LRMPc": run_frame.Output(checkpoint=False, check=False),
}
