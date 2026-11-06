"""Local gain guarantees use x only, after optimizing the full local problem."""

import pyomo.environ as pyo
import pytest

from models.model import LoadManagementModel
from models.sp import LSP_detailed
from solver_support import require_gurobi


def _data():
  return {None: {
    "Nn": {None: 2}, "Nf": {None: 1},
    "incoming_load": {(1, 1): 1, (2, 1): 10},
    "demand": {(1, 1): 1, (2, 1): 1},
    "max_utilization": {1: 1},
    "memory_requirement": {1: 1}, "memory_capacity": {1: 1, 2: 0},
    "alpha": {(1, 1): 1, (2, 1): 1},
    "beta": {(1, 1, 1): 0, (1, 2, 1): 0.1,
             (2, 1, 1): 20, (2, 2, 1): 0},
    "gamma": {(1, 1): 0.1, (2, 1): 0.1},
    "neighborhood": {(1, 1): 0, (1, 2): 1, (2, 1): 1, (2, 2): 0},
  }}


def test_detailed_local_objective_maximizes_gain_with_offload_price():
  data = _data()
  data[None]["incoming_load"][(1, 1)] = 4
  data[None]["alpha"][(1, 1)] = 2
  data[None]["beta"][(1, 2, 1)] = 0.5
  data[None]["gamma"][(1, 1)] = 0.25
  data[None]["pi"] = {1: 0.1}
  instance = LSP_detailed().generate_instance(data)
  instance.x[1] = 1
  instance.y[1, 1] = 0
  instance.y[2, 1] = 2
  instance.z[1] = 1
  assert instance.OBJ.sense == pyo.maximize
  assert pyo.value(instance.OBJ) == pytest.approx(0.6375)


@pytest.mark.parametrize("offload_reward,floor,global_gain,local_x", [
  (0.1, 1, 0.9, 1),
  (2, 0, 1.81, 0),
])
def test_selfish_model_protects_x_gain_without_requiring_local_offloading(
    offload_reward, floor, global_gain, local_x,
  ):
  require_gurobi()
  from models.selfish import SelfishLoadManagementModel

  options = {"OutputFlag": 0, "MIPGap": 0}
  data = _data()
  data[None]["beta"][(1, 2, 1)] = offload_reward
  original = LoadManagementModel()
  original_solution = original.solve(original.generate_instance(data), options, "gurobi")
  model = SelfishLoadManagementModel()
  instance = model.generate_instance(data)
  solution = model.solve(instance, options, "gurobi")

  assert original_solution["termination_condition"] == "optimal"
  assert original_solution["obj"] == pytest.approx(1.81)
  assert original_solution["x"][0] == pytest.approx(0)
  assert solution["termination_condition"] == "optimal"
  assert solution["obj"] == pytest.approx(global_gain)
  assert solution["x"] == pytest.approx([local_x, 0])
  # Node 2's local optimum forwards everything and has total gain 20, but
  # its x-only guarantee is zero. Including y would make this infeasible.
  # With reward 2, the full local objective chooses y instead of x, hence
  # floor 0. Maximizing x alone would incorrectly impose floor 1.
  assert pyo.value(instance.minimum_local_gain[1]) == pytest.approx(floor)
  assert pyo.value(instance.minimum_local_gain[2]) == pytest.approx(0)
  for n in instance.N:
    assert pyo.value(instance.local_processing_gain[n]) >= (
      pyo.value(instance.minimum_local_gain[n]) - 1e-7
    )
  assert solution["runtime"] >= solution["local_reference_runtime"]


def test_selfish_model_handles_zero_load_and_copies_centralized_defaults():
  require_gurobi()
  from models.selfish import SelfishLoadManagementModel

  data = _data()
  data[None]["incoming_load"] = {(1, 1): 0, (2, 1): 0}
  del data[None]["gamma"]
  model = SelfishLoadManagementModel()
  instance = model.generate_instance(data)
  solution = model.solve(instance, {"OutputFlag": 0, "MIPGap": 0}, "gurobi")
  assert solution["termination_condition"] == "optimal"
  assert solution["obj"] == pytest.approx(0)
  assert all(pyo.value(instance.minimum_local_gain[n]) == 0 for n in instance.N)
