"""Greedy-RR uses capacity requests, local resource data and no auction values."""
import numpy as np

from decentralized_greedy_rr import request_capacity, accept_requests, request_memory, allocate_memory


def test_requests_use_only_sender_beta_and_advertised_capacity():
  remaining = np.array([[7.], [0.], [0.]])
  capacity = np.array([[0.], [2.], [4.]])
  neighbors = np.ones((3, 3))-np.eye(3)
  data = {None: {'beta': {(1, 2, 1): 8., (1, 3, 1): 5.}}}
  assert request_capacity(remaining, capacity, neighbors, data) == [(0, 1, 0, 2)]
  capacity[1, 0] = 0
  assert request_capacity(remaining, capacity, neighbors, data) == [(0, 2, 0, 4)]
  neighbors[0, 2] = 0
  assert request_capacity(remaining, capacity, neighbors, data) == []
  assert request_memory(remaining, np.array([0., 2., 3.]), neighbors) == [(0, 1, 0)]


def test_round_robin_shares_capacity_and_remembers_next_sender():
  capacity = np.zeros((4, 1)); capacity[3, 0] = 5
  requests = [(0, 3, 0, 10), (1, 3, 0, 10), (2, 3, 0, 10)]
  cursors = {}
  assert accept_requests(requests, capacity, cursors) == [(0, 3, 0, 2), (1, 3, 0, 2), (2, 3, 0, 1)]
  capacity[3, 0] = 1
  assert accept_requests(requests, capacity, cursors) == [(2, 3, 0, 1)]
  # Batched turns must not iterate once per request on large loads.
  capacity[3, 0] = 3_000_000
  got = accept_requests([(i, 3, 0, 2_000_000) for i in range(3)], capacity, {})
  assert [q for _, _, _, q in got] == [1_000_000]*3
  capacity[3, 0] = 7
  got = accept_requests([(0, 3, 0, 1), (1, 3, 0, 10)], capacity, {})
  assert got == [(0, 3, 0, 1), (1, 3, 0, 6)]


def test_replica_choice_counts_senders_and_respects_roles_and_useful_batches():
  # Receiver 2 knows its own demand/RAM and no sender beta, alpha or gamma.
  data = {None: {'demand': {(3, 1): 2., (3, 2): 1.},
                 'memory_requirement': {1: 1, 2: 1}, 'max_utilization': {1: 1., 2: 1.}}}
  r = np.zeros((3, 2)); rho = np.array([0., 0., 2.]); senders = np.zeros((3, 2), dtype=bool)
  requests = [(0, 2, 0), (1, 2, 0), (0, 2, 1)]
  extra, left = allocate_memory(requests, r, rho, senders, data)
  assert extra[2].tolist() == [2, 0]  # two half-capacity replicas are needed for one request
  assert left.tolist() == [0, 0, 0]
  assert r.sum() == 0 and rho[2] == 2  # inputs are not mutated
  # A sender of f0 cannot simultaneously become its receiver.
  senders[2, 0] = True
  extra, left = allocate_memory(requests, r, rho, senders, data)
  assert extra[2].tolist() == [0, 1]
  # Unaffordable most-requested function: choose a feasible one instead.
  senders[:] = False; rho[2] = 1
  extra, left = allocate_memory(requests, r, rho, senders, data)
  assert extra[2].tolist() == [0, 1]


def test_complete_step_falls_back_to_other_neighbors_and_keeps_cloud_residue():
  from types import SimpleNamespace
  from copy import deepcopy
  from decentralized_greedy_rr import _solve_step
  from utils import step_time
  from utils.centralized import validate_centralized_solution
  data = {None: {
    'Nn': {None: 3}, 'Nf': {None: 1},
    'incoming_load': {(n, 1): 5 if n == 1 else 0 for n in (1, 2, 3)},
    'demand': {(n, 1): 1. for n in (1, 2, 3)},
    'alpha': {(n, 1): 2. for n in (1, 2, 3)},
    'gamma': {(n, 1): .1 for n in (1, 2, 3)},
    'beta': {(n, j, 1): (.9 if j == 2 else .8) if n != j else 0.
             for n in (1, 2, 3) for j in (1, 2, 3)},
    'neighborhood': {(n, j): int(n != j) for n in (1, 2, 3) for j in (1, 2, 3)},
    'memory_capacity': {1: 0, 2: 1, 3: 3},
    'memory_requirement': {1: 1}, 'max_utilization': {1: 1.},
  }}
  original = deepcopy(data)
  ctx = SimpleNamespace(Nn=3, Nf=1, agents=list(range(3)), solver_name='missing_solver',
    general_options={'use_dp': True}, parallelism=0, max_iterations=10, time_limit=60,
    tolerance=1e-6, neighborhood=np.ones((3, 3))-np.eye(3))
  for limit, iterations, expected in [(60, 10, 4), (1e-12, 10, 0), (60, 1, 0)]:
    ctx.time_limit, ctx.max_iterations = limit, iterations
    clock = step_time.start_step()
    try:
      result = _solve_step(ctx, 0, data)
    finally:
      step_time.end_step(clock)
    solution = result.solutions['LSPc']
    assert solution['y'].sum() == expected
    assert solution['z'].sum() == 5-expected
    assert solution['x'].sum() == 0
    validate_centralized_solution(solution['x'], solution['y'], solution['z'], solution['r'], data)
    assert data == original
