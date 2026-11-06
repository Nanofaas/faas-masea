"""Detailed offloading must use outgoing edges, even when forbidden routes pay more."""
from copy import deepcopy

import pyomo.environ as pyo
import pytest

from models.local_sp import solve_agent_problem
from models.sp import LSP_detailed
from solver_support import require_gurobi


def _data():
  return {None: {
    'Nn': {None: 3}, 'Nf': {None: 1}, 'whoami': {None: 1},
    'incoming_load': {(1, 1): 5}, 'demand': {(1, 1): 1.},
    'max_utilization': {1: 1.}, 'memory_requirement': {1: 1},
    'memory_capacity': {1: 0}, 'gamma': {(1, 1): 1.},
    'neighborhood': {(1, 2): 1},
    'beta': {(1, 1, 1): 0., (1, 2, 1): 1., (1, 3, 1): 10.},
  }}


@pytest.mark.parametrize('use_dp', [False, True], ids=['gurobi', 'dp'])
@pytest.mark.parametrize('case,expected_y,expected_x,expected_z,expected_obj', [
  ('nonneighbor', [0, 5, 0], 0, 0, 1.),
  ('self', [0, 5, 0], 0, 0, 1.),
  ('incoming_edge_only', [0, 0, 0], 0, 5, -1.),
  ('missing_topology', [0, 0, 0], 0, 5, -1.),
  ('isolated_with_ram', [0, 0, 0], 2, 3, -.2),
  ('zero_reward_nonedge', [0, 0, 0], 0, 5, -1.),
  ('price_favors_rejection', [0, 0, 0], 0, 5, -1.),
  ('zero_load', [0, 0, 0], 0, 0, 0.),
])
def test_detailed_solution_respects_topology(
    use_dp, case, expected_y, expected_x, expected_z, expected_obj,
  ):
  if not use_dp:
    require_gurobi()
  data = _data()
  v = data[None]
  if case == 'self':
    v['beta'][1, 1, 1] = 20.
  elif case == 'incoming_edge_only':
    v['neighborhood'] = {(2, 1): 1}
  elif case == 'missing_topology':
    del v['neighborhood']
  elif case == 'isolated_with_ram':
    v['neighborhood'] = {}
    v['memory_capacity'][1] = 2
  elif case == 'zero_reward_nonedge':
    v['neighborhood'] = {}
    v['beta'] = {(1, j, 1): 0. for j in range(1, 4)}
  elif case == 'price_favors_rejection':
    v['pi'] = {1: 3.}
  elif case == 'zero_load':
    v['incoming_load'][1, 1] = 0
  before = deepcopy(data)
  result = solve_agent_problem(
    LSP_detailed(), data, {'use_dp': use_dp, 'OutputFlag': 0, 'MIPGap': 0},
    'missing_solver' if use_dp else 'gurobi',
  )
  assert result['termination_condition'] == 'optimal'
  assert result['y'] == pytest.approx(expected_y)
  assert result['x'] == pytest.approx([expected_x])
  assert result['z'] == pytest.approx([expected_z])
  assert result['obj'] == pytest.approx(expected_obj)
  assert data == before
  for j, forwarded in enumerate(result['y'], 1):
    assert forwarded <= v['incoming_load'][1, 1] * v.get('neighborhood', {}).get((1, j), 0)


@pytest.mark.parametrize('use_dp', [False, True], ids=['gurobi', 'dp'])
def test_selfish_floor_cannot_use_a_profitable_nonedge(use_dp):
  if not use_dp:
    require_gurobi()
  from models.selfish import SelfishLoadManagementModel
  data = _data()
  data[None]['memory_capacity'] = {1: 5, 2: 0, 3: 0}
  data[None]['alpha'] = {(1, 1): 2.}
  for n in (2, 3):
    data[None]['incoming_load'][n, 1] = 0
    data[None]['demand'][n, 1] = 1.
  model = SelfishLoadManagementModel()
  instance = model.generate_instance(data)
  model.compute_local_gain_floors(
    instance, {'use_dp': use_dp, 'OutputFlag': 0, 'MIPGap': 0},
    'missing_solver' if use_dp else 'gurobi',
  )
  assert pyo.value(instance.minimum_local_gain[1]) == pytest.approx(2.)
