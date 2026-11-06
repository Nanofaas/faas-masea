from dataclasses import dataclass
from functools import partial
import logging

from run_faasmacro import (
  parallel_solver_run,
  combine_solutions,
  compute_social_welfare,
  solve_subproblem
)
from utils.centralized import check_feasibility
from utils.faasmacro import (
  compute_centralized_objective,
  relative_objective_gap
)
from utils.common import load_configuration
from models.sp import LSP_detailed, LSP, LSP_fixedr, LSPr_x
from models.model import PYO_VAR_TYPE

from collections import deque
import pyomo.environ as pyo
from copy import deepcopy
from typing import Tuple
import pandas as pd
import numpy as np
import argparse
import os
from utils import run_frame, run_log, step_time
from utils.selective_memory import SelectiveMemory
from utils import communication

log = run_log.logger(__name__)


VAR_TYPE = int if PYO_VAR_TYPE == pyo.NonNegativeIntegers else float




def check_ls_pr_feasibility_from_fixed_y(sp_data, y, tol=1e-9):
  Nn = sp_data[None]["Nn"][None]
  Nf = sp_data[None]["Nf"][None]
  bad = []
  # loop over nodes
  for n in range(Nn):
    mem_cap = sp_data[None]["memory_capacity"][n+1]
    req_mem = 0
    details = []
    # loop over functions
    for f in range(Nf):
      fixed_in = y[:, n, f].sum()
      d = sp_data[None]["demand"][(n+1, f+1)]
      u = sp_data[None]["max_utilization"][f+1]
      ram = sp_data[None]["memory_requirement"][f+1]
      # check the number of replicas required for the current function
      # and the corresponding memory requirement
      req_r = int(np.ceil((d * fixed_in - tol) / u)) if fixed_in > tol else 0
      req_mem += req_r * ram
      details.append({
        "f": f,
        "fixed_in": float(fixed_in),
        "req_r": req_r,
        "ram": ram,
        "mem_for_f": req_r * ram
      })
    # track nodes for which the memory capacity is exceeded
    if req_mem > mem_cap:
      bad.append({
        "node": n,
        "required_memory": int(req_mem),
        "memory_capacity": int(mem_cap),
        "details": details
      })
  return bad


def check_stopping_criteria(
    it: int,
    max_iterations: int,
    blackboard: np.array,
    omega: np.array = None,
    rmp_omega: np.array = None,
    odev_queue: deque = None,
    a: np.array = None,
    bids: pd.DataFrame = None,
    memory_bids: pd.DataFrame = None,
    tolerance: float = 1e-6,
    total_runtime: float = 0.0,
    time_limit: float = float("inf"),
    sp_omega: np.array = None,
    sp_y: np.array = None,
    plateau_tolerance: float = None
  ) -> Tuple[bool, str]:
  """plateau_tolerance (config, optional): stop once the best objective improved less than it, in
  all, over the patience window; without it every single iteration must improve less than tolerance
  (one more request per iteration on a large instance resets that window)."""
  if omega is None:
    omega = sp_omega
  if a is None:
    a = np.zeros_like(rmp_omega) if rmp_omega is not None else None
  # Newly created capacity needs an auction before declaring a plateau or exhaustion.
  if (a is not None and (a > tolerance).any() and (omega > tolerance).any()
      and it < max_iterations - 1 and total_runtime < time_limit):
    return False, None
  stop = False
  why_stopping = None
  if it >= max_iterations - 1:
    stop = True
    why_stopping = "max iterations reached"
  elif (blackboard <= tolerance).all():
    stop = True
    why_stopping = "no capacity left"
  elif (omega <= tolerance).all():
    stop = True
    why_stopping = "all load assigned"
  elif (
      rmp_omega is not None and a is not None and
        (rmp_omega <= tolerance).all() and (a <= tolerance).all()
    ):
    stop = True
    why_stopping = "load cannot be assigned"
  elif (
      bids is not None and memory_bids is not None and
        len(bids) == 0 and len(memory_bids) == 0
    ):
    stop = True
    why_stopping = "no available or convenient sellers"
  elif total_runtime >= time_limit:
    stop = True
    why_stopping = f"reached time limit: {total_runtime} >= {time_limit}"
  elif (
      sp_omega is not None and rmp_omega is not None and sp_y is not None and
        (np.abs(sp_omega - rmp_omega) <= tolerance).all()
    ):
    stop = True
    why_stopping = "feasible solution found"
  if not stop and odev_queue is not None and len(odev_queue) >= odev_queue.maxlen:
    if plateau_tolerance is not None:
      if sum(odev_queue) < plateau_tolerance:  # relative gains over the window, compounded ~ summed
        stop = True
        why_stopping = f"best improved < {plateau_tolerance} over {odev_queue.maxlen} iterations"
    elif all(odev < tolerance for odev in odev_queue):
      stop = True
      why_stopping = "UB/LB diff < tol"
  return stop, why_stopping


@step_time.parallel
def compute_residual_capacity(
    x: np.array, y: np.array, r: np.array, data: dict
  ) -> Tuple[np.array, np.array, np.array]:
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  from utils.compute_backend import cuda_compute_enabled
  if (cuda_compute_enabled() and Nn >= 100 and Nn * Nf >= 1000
      and 8 * (Nn*Nn*Nf + 6*Nn*Nf + Nf) <= 64*1024*1024
      and all(a.dtype == np.float64 for a in (x,y,r))
      and all(type(v) in (float,np.float64) or (type(v) is int and abs(v)<=2**53)
              for v in (*data[None]['demand'].values(), *data[None]['max_utilization'].values()))):
    step_time.mark_shared_device()
    from models.numeric_gpu import residual_capacity_cuda
    demand = np.array([[data[None]['demand'][(n+1,f+1)] for f in range(Nf)] for n in range(Nn)])
    utilization = np.array([data[None]['max_utilization'][f+1] for f in range(Nf)])
    return residual_capacity_cuda(x, y, r, demand, utilization)
  if cuda_compute_enabled():
    from utils.compute_backend import record_operation
    record_operation('residual_capacity','cpu',Nn,shape=[Nn,Nf],
                     reason='outside validated float64/scalar dtype domain, below 100 nodes/1000 entries or above 64 MiB')
  # loop over nodes and functions
  cap = np.zeros((Nn,Nf))
  residual_capacity = np.zeros((Nn,Nf))
  ell = np.zeros((Nn,Nf))
  for n in step_time.timed(range(Nn)):
    for f in range(Nf):
      # number of enqueued requests
      ell[n,f] = x[n,f] + y[:,n,f].sum()
      # computational capacity
      cap[n,f] = r[n,f] * (
        data[None]["max_utilization"][f+1] / data[None]["demand"][(n+1,f+1)]
      )
      residual_capacity[n,f] = max(0.0, cap[n,f] - ell[n,f])
  return cap, residual_capacity, ell


@step_time.parallel
def define_bids(
    omega: np.array,
    blackboard: np.array,
    p: np.array,
    data: dict,
    neighborhood: np.array,
    rho: np.array,
    auction_options: dict,
    latency: np.array,
    fairness: np.array,
    force_memory_bids: bool,
    memory_bid_value=None,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Bids of the buyers. ``memory_bid_value(i, j, f)`` (MADEA2): memory bids
  also carry the uncovered quantity "d" and that value "v"."""
  from utils.compute_backend import record_operation
  record_operation('define_bids','cpu',len(omega),shape=list(omega.shape),
                   reason='full CUDA adapter failed phase performance gate; custom callbacks also require ordered host execution')
  use_internal_value = auction_options.get("use_internal_value", False)
  # loop over agents and functions
  potential_buyers, functions_to_share = np.nonzero(omega)
  bids = {
    "i": [], "j": [], "f": [], "d": [], "b": [], "utility": []
  }
  memory_bids = {
    "i": [], "j": [], "f": []
  }
  if memory_bid_value is not None:
    memory_bids.update(d=[], v=[])
  for i,f in step_time.timed(zip(potential_buyers, functions_to_share), key=lambda item: item[0]):
    # identify potential sellers
    potential_sellers = set(np.nonzero(neighborhood[i,:])[0])
    # -- capacity sellers are neighbors with residual computing capacity
    # for function f
    potential_capacity_sellers = potential_sellers.intersection(
      set(np.where(blackboard[:,f]>=1)[0])
    )
    # -- memory sellers are neighbors with residual memory capacity to
    # instantiate new replicas
    potential_memory_sellers = potential_sellers.intersection(
      set(np.nonzero(rho)[0])
    )
    # -- loop over potential sellers
    utility = []
    candidate_sellers = []
    candidate_values = []
    for j in potential_capacity_sellers:
      # -- compute utility
      value = data[None]["beta"][(i+1,j+1,f+1)]
      if use_internal_value:
        # Match the welfare normalization, including its zero-load convention.
        load = data[None]["incoming_load"][(i+1,f+1)] or 1
        value = (value + data[None]["gamma"][(i+1,f+1)]) / load
      ut = (
        value -
        p[j,f] -
        auction_options["latency_weight"] * latency[i,j] -
        auction_options["fairness_weight"] * fairness[i,f]
      )
      reservation_utility = 0.0 if use_internal_value else -data[None]["gamma"][(i+1,f+1)]
      if ut > reservation_utility:
        utility.append(ut)
        candidate_sellers.append(j)
        candidate_values.append(value)
    # compute weights and define bids
    assigned = 0
    if len(utility) > 0:
      utility = np.array(utility)
      sellers_order = np.argsort(utility)[::-1]
      idx = 0
      while idx < len(sellers_order) and assigned < omega[i,f]:
        j = candidate_sellers[sellers_order[idx]]
        delta = 0.0
        if idx < len(sellers_order) - 1:
          delta = utility[sellers_order[idx]] - utility[sellers_order[idx+1]]
        b = p[j,f] + auction_options["epsilon"] + delta
        if use_internal_value:
          b = min(b, candidate_values[sellers_order[idx]])
        if auction_options["unit_bids"]:
          d = 1
          while (d < int(min(blackboard[j,f], omega[i,f])) + 1) and (
              assigned < omega[i,f]
            ):
            bids["i"].append(i)
            bids["f"].append(f)
            bids["j"].append(j)
            bids["d"].append(1)
            bids["b"].append(b)
            bids["utility"].append(utility[sellers_order[idx]])
            assigned += 1
            d += 1
        else:
          d = VAR_TYPE(min(blackboard[j,f], (omega[i,f] - assigned)))
          bids["i"].append(i)
          bids["f"].append(f)
          bids["j"].append(j)
          bids["d"].append(d)
          bids["b"].append(b)
          bids["utility"].append(utility[sellers_order[idx]])
          assigned += d
        idx += 1
      # -- if you could not bid for everything, ask also for new replicas
      if assigned < omega[i,f]:
        for idx in sellers_order:
          j = candidate_sellers[idx]
          if j in potential_memory_sellers:
            memory_bids["i"].append(i)
            memory_bids["j"].append(j)
            memory_bids["f"].append(f)
            if memory_bid_value is not None:
              memory_bids["d"].append(float(omega[i,f] - assigned))
              memory_bids["v"].append(memory_bid_value(i, j, f))
    # if you could not bid for all, ask for new replicas
    if assigned < omega[i,f] or force_memory_bids:
      for j in potential_memory_sellers - potential_capacity_sellers:
        memory_bids["i"].append(i)
        memory_bids["j"].append(j)
        memory_bids["f"].append(f)
        if memory_bid_value is not None:
          memory_bids["d"].append(float(max(omega[i,f] - assigned, 0.0)))
          memory_bids["v"].append(memory_bid_value(i, j, f))
  return pd.DataFrame(bids), pd.DataFrame(memory_bids), len(potential_buyers)


def ensure_memory_sellers(
    potential_sellers, functions_to_share,
    potential_memory_sellers, Nf
  ):
  extra_sellers = np.repeat(potential_memory_sellers, Nf)
  extra_funcs = np.tile(np.arange(Nf), len(potential_memory_sellers))
  sellers_all = np.concatenate((potential_sellers, extra_sellers))
  funcs_all = np.concatenate((functions_to_share, extra_funcs))
  pairs = np.unique(np.column_stack((sellers_all, funcs_all)), axis=0)
  return pairs[:,0], pairs[:,1]


@step_time.parallel
def evaluate_bids(
    bids: pd.DataFrame,
    blackboard: np.array,
    data: dict,
    previous_y: np.array = None,
    ell: np.array = None,
    p: np.array = None,
    total_capacity: np.array = None,
    u0: np.array = None,
    auction_options: dict = None,
    initial_rho: np.array = None,
    r: np.array = None,
    tentatively_start_replicas: bool = False,
    it: int = 0,
    may_replace_existing_assignments: bool = True,
    residual_capacity: np.array = None,
    replica_value_per_mb: bool = False,
    congestion_weight: float = 0.0,
  ) -> np.array:
  """Seller-side evaluation. MADEA2 rules: ``replica_value_per_mb`` serves the
  pending bids by value per MB of a new replica when replicas start here;
  ``congestion_weight`` raises the price by the offered demand left unserved."""
  from utils.compute_backend import record_operation
  record_operation('evaluate_bids','cpu',data[None]['Nn'][None],shape=list(blackboard.shape),
                   reason='ordered vendors update shared sender/receiver roles, including incumbent role release; per-vendor price writes stay at that boundary')
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  last_y = np.zeros((Nn,Nn,Nf)) if previous_y is None else previous_y.copy()
  if ell is None:
    ell = np.zeros((Nn,Nf))
  if p is None:
    p = np.zeros((Nn,Nf))
  if total_capacity is None:
    total_capacity = np.ones((Nn,Nf))
  if u0 is None:
    u0 = np.zeros((Nn,Nf))
  if auction_options is None:
    auction_options = {"eta": 0.0, "zeta": 0.0}
  if initial_rho is None:
    initial_rho = np.zeros((Nn,))
  if r is None:
    r = np.zeros((Nn,Nf))
  # eta may be a per-iteration schedule (e.g. [0.5, 0.3, 0.15]); clamp to the
  # last value once the schedule is exhausted, accept a plain scalar too
  eta = auction_options["eta"]
  eta = eta[min(it, len(eta) - 1)] if isinstance(eta, (list, tuple)) else eta
  # no_ping_pong (validate_centralized_solution): a node must not both send
  # and receive the same function across the accumulated y. Track realized
  # roles and refuse any bid that would give a node a second, conflicting
  # role. Mirrors the guard in decentralized_auction.evaluate_bids.
  sending = last_y.sum(axis = 1) > 1e-10
  receiving = last_y.sum(axis = 0) > 1e-10
  # loop over agents and functions
  potential_sellers, functions_to_share = [None, None]
  potential_sellers, functions_to_share = np.nonzero(blackboard)
  if tentatively_start_replicas:
    potential_sellers, functions_to_share = ensure_memory_sellers(
      potential_sellers,
      functions_to_share,
      np.nonzero(initial_rho)[0],
      Nf
    )
  y = np.zeros((Nn,Nn,Nf))
  additional_replicas = np.zeros((Nn,Nf))
  rho = deepcopy(initial_rho)
  unique_potential_sellers = np.unique(potential_sellers)
  for j in step_time.timed(unique_potential_sellers):
    # extract bids for the current node
    all_bids_for_j = bids[(bids["j"] == j)].sort_values(
      by = "b", ascending = False
    )
    bid_rows = all_bids_for_j.to_records(index=False)
    remaining_capacities = None
    if may_replace_existing_assignments:
      remaining_capacities = residual_capacity[j,:].astype(int).copy()
    else:
      remaining_capacities = blackboard[j,:].astype(int).copy()
    all_min_b = all_bids_for_j.groupby("f")["b"].max()
    pending_bids = []
    # Fill free capacity first; keep each bid's unserved quantity for its function.
    for bid in bid_rows:
      i = int(bid["i"])
      f = int(bid["f"])
      if receiving[i,f] or sending[j,f]:
        continue
      q = min(remaining_capacities[f], bid["d"])
      if q > 0:
        y[i,j,f] += q
        remaining_capacities[f] -= q
        all_min_b[f] = min(all_min_b[f], bid["b"])
        sending[i,f] = True
        receiving[j,f] = True
      if bid["d"] > q:
        pending_bids.append((i, f, bid["d"] - q, bid["b"]))

    if tentatively_start_replicas and replica_value_per_mb:
      # replicas started here compete for the same memory: serve first the
      # pending bids that bring the most value per MB of a new replica
      pending_bids.sort(key=lambda t: -t[3] * (
        data[None]["max_utilization"][t[1]+1] / data[None]["demand"][(j+1,t[1]+1)]
      ) / data[None]["memory_requirement"][t[1]+1])
    for i, f, remaining_d, b in pending_bids:
      if receiving[i,f] or sending[j,f]:
        continue
      if tentatively_start_replicas:
        current_a = int(additional_replicas[j,f])
        max_a = current_a + int(rho[j]/data[None]["memory_requirement"][f+1])
        # Reuse capacity from a replica already started, even if memory is exhausted.
        a = max(current_a, 1)
        while a <= max_a and remaining_d > 0:
          capacity = (r[j,f] + a) * (
            data[None]["max_utilization"][f+1] / data[None]["demand"][(j+1,f+1)]
          ) - ell[j,f] - y[:,j,f].sum()
          q = min(remaining_d, VAR_TYPE(max(0.0, capacity)))
          if q > 0:
            y[i,j,f] += q
            remaining_d -= q
            all_min_b[f] = min(all_min_b[f], b)
            sending[i,f] = True
            receiving[j,f] = True
            if current_a < a:
              rho[j] -= (a - current_a) * data[None]["memory_requirement"][f+1]
              additional_replicas[j,f] = a
              current_a = a
          a += 1
      if remaining_d <= 0 or not may_replace_existing_assignments:
        continue
      if b <= p[j,f]:
        continue
      # Only prior-round assignments are replaceable; a bid may span incumbents.
      for previous_buyer in np.nonzero(last_y[:,j,f])[0]:
        if previous_buyer == i:
          continue
        q = min(remaining_d, last_y[previous_buyer,j,f])
        y[previous_buyer,j,f] -= q
        y[i,j,f] += q
        last_y[previous_buyer,j,f] -= q
        remaining_d -= q
        all_min_b[f] = min(all_min_b[f], b)
        sending[i,f] = True
        receiving[j,f] = True
        sending[previous_buyer,f] = (
          previous_y[previous_buyer,:,f] + y[previous_buyer,:,f]
        ).sum() > 1e-10
        if remaining_d <= 0:
          break
    # compute utilization and update prices
    for f,b in all_min_b.items():
      u = (ell[j,f] + y[:,j,f].sum()) / total_capacity[j,f]
      p[j,f] = b + eta * (u - u0[j,f])
      if congestion_weight > 0:
        # demand the seller saw but could not serve, relative to its
        # capacity: local information, no extra message
        requested = all_bids_for_j.loc[all_bids_for_j["f"] == f, "d"].sum()
        excess = max(0.0, float(requested - y[:,j,f].sum()))
        p[j,f] += congestion_weight * excess / max(float(total_capacity[j,f]), 1.0)
    for f in set(functions_to_share) - set(all_min_b.index):
      p[j,f] *= (1 - auction_options["zeta"])
  return y, p, additional_replicas, len(unique_potential_sellers)


def neigh_dict_to_matrix(neighborhood_dict: dict, Nn: int) -> np.array:
  neighborhood = np.zeros((Nn,Nn))
  for n1 in range(Nn):
    for n2 in range(Nn):
      if n1 != n2 and neighborhood_dict[(n1+1,n2+1)]:
        neighborhood[n1,n2] = 1
  return neighborhood


def _replicas_for_capacity(capacity: float, target: int) -> int:
  """Invert capacity using the same strict boundary as the bidding rules."""
  count = max(0, int(np.ceil(target / capacity)))
  # Division can round across an integer even when multiplication is exact.
  while count and (count - 1) * capacity >= target:
    count -= 1
  while count * capacity < target:
    count += 1
  return count


def next_replica_batch(replicas: float, capacity: float) -> Tuple[int, float]:
  """Smallest replica batch that increases usable capacity in the request domain."""
  if VAR_TYPE is not int:
    return 1, capacity
  served = int(np.floor(replicas * capacity))
  count = max(1, _replicas_for_capacity(capacity, served + 1) - int(replicas))
  gain = int(np.floor((replicas + count) * capacity)) - served
  return count, gain


def confirm_memory_bids(
    candidates: pd.DataFrame, residual_demand: np.ndarray, *, tolerance: float = 1e-6,
  ) -> pd.DataFrame:
  """Confirm one buyer's candidates using only its current residual demand."""
  if candidates.empty:
    return candidates.copy()
  demand = residual_demand[candidates["f"].to_numpy(dtype=int)]
  confirmed = candidates.loc[demand > tolerance].copy()
  if "d" in confirmed.columns:
    confirmed["d"] = demand[demand > tolerance]
  return confirmed


_BUILTIN_CONFIRM_MEMORY_BIDS = confirm_memory_bids


def confirm_memory_bid_batch(candidates,omega,*,tolerance=1e-6,confirm=None):
  """Confirm independent buyers in the caller's existing parallel phase."""
  from utils.compute_backend import cuda_compute_enabled,record_operation
  confirm=confirm_memory_bids if confirm is None else confirm
  if candidates.empty:return candidates.copy()
  if cuda_compute_enabled():
    count=int(candidates['i'].nunique())
    ids=candidates['i'].to_numpy();functions=candidates['f'].to_numpy()
    native=all(isinstance(candidates[key].dtype,np.dtype) for key in ('i','f'))
    bounds=(native and ids.dtype.kind in 'iu' and functions.dtype.kind in 'iu'
            and ids.min()>=0 and ids.max()<len(omega) and functions.min()>=0 and functions.max()<omega.shape[1])
    eligible=(confirm is _BUILTIN_CONFIRM_MEMORY_BIDS and count>=30 and len(candidates)>=1000
              and omega.dtype==np.float64 and bounds and omega.nbytes+25*len(candidates)<=64*1024*1024)
    if eligible:
      step_time.mark_shared_device()
      from models.numeric_gpu import confirm_memory_rows_cuda
      mask,demand=confirm_memory_rows_cuda(ids,functions,omega,tolerance)
      order=np.argsort(ids,kind='stable');kept=order[mask[order]]
      result=candidates.iloc[kept].copy()
      if 'd' in result.columns:result['d']=demand[kept]
      return result
    reason='custom callback requires ordered host execution' if confirm is not _BUILTIN_CONFIRM_MEMORY_BIDS else 'performance/index/buffer gate: >=30 buyers, >=1000 rows, native integer indices, float64 omega, <=64 MiB'
    record_operation('memory_confirmation','cpu',count,shape=[len(candidates),*omega.shape],reason=reason)
  confirmed=[]
  for i,group in step_time.timed(candidates.groupby('i'),key=lambda item:item[0]):
    confirmed.append(confirm(group,omega[int(i)],tolerance=tolerance))
  return pd.concat(confirmed)


@step_time.parallel
def start_additional_replicas(
    memory_bids: pd.DataFrame,
    r: np.array,
    data: dict,
    rho: np.array
  ) -> Tuple[np.array, np.array]:
  from utils.compute_backend import cuda_compute_enabled,record_operation
  if cuda_compute_enabled():
    from models.replicas_gpu import replica_batch_eligible,start_additional_replicas_cuda
    eligible,reason,count=replica_batch_eligible(memory_bids,r,rho,value_policy=False)
    if eligible:
      step_time.mark_shared_device()
      return start_additional_replicas_cuda(memory_bids,r,data,rho)
    record_operation('replicas_proportional','cpu',count,shape=list(r.shape),reason=reason)
  # loop over sellers
  additional_replicas = np.zeros(r.shape)
  residual_capacity = deepcopy(rho)
  for j, bids_for_j in step_time.timed(memory_bids.groupby("j"), key=lambda item: item[0]):
    if rho[j] > 0:
      seller_capacity = float(rho[j])
      # count the fraction that each function requires
      fractions = bids_for_j["f"].value_counts(normalize = True).sort_index()
      allocated_memory = {}
      capacities = {
        f: data[None]["max_utilization"][f+1] / data[None]["demand"][(j+1,f+1)]
        for f in fractions.index
      }
      # assign new replicas proportionally to this fraction
      for f, frac in fractions.items():
        # -- check memory requirement
        ram_f = data[None]["memory_requirement"][f+1]
        # -- determine the maximum number of replicas that fit in the
        # assignable fraction of the residual memory capacity
        a = int((seller_capacity * frac) // ram_f)
        if VAR_TYPE is int:
          # Trim replicas that add no whole request, including a useless tail.
          capacity = capacities[f]
          before = int(np.floor(r[j,f] * capacity))
          after = int(np.floor((r[j,f] + a) * capacity))
          needed = max(0, _replicas_for_capacity(capacity, after) - int(r[j,f]))
          a = min(a, needed) if after > before else 0
        # -- update
        residual_capacity[j] -= int(ram_f * a)
        additional_replicas[j,f] = a
        allocated_memory[f] = ram_f * a
      while True:
        candidates = []
        for f, frac in fractions.items():
          ram_f = data[None]["memory_requirement"][f+1]
          deficit = seller_capacity * frac - allocated_memory[f]
          count, _ = next_replica_batch(r[j,f] + additional_replicas[j,f], capacities[f])
          memory = count * ram_f
          # A function may use another's unspendable share once its own is full.
          if memory <= residual_capacity[j] and (VAR_TYPE is int or deficit > 0):
            candidates.append((deficit, frac, -int(f), int(f), count, memory))
        if not candidates:
          break
        _, _, _, f, count, memory = max(candidates)
        additional_replicas[j, f] += count
        allocated_memory[f] += memory
        residual_capacity[j] -= memory
  return additional_replicas, residual_capacity




def compute_offloaded_demand(y: np.ndarray) -> np.ndarray:
  return y.sum(axis=1)


RULE_NAMES = ("define_bids", "evaluate_bids", "start_additional_replicas")
