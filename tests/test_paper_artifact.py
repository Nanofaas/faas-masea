"""Regression checks for decisions, objectives and logical messages."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import methods
from catalog_centralized import solve
from postprocessing import load_solution
from remote_experiments.instances import load_materialized_instance
from run_centralized_model import get_current_load, update_data
from utils.centralized import encode_solution, validate_centralized_solution

ROOT = Path(__file__).resolve().parents[1]
DECENTRALIZED = {**methods.HARMONIZED, **methods.SEQUENTIAL_METHODS}


def test_only_paper_methods_are_selectable():
  names = set(DECENTRALIZED) | set(methods.CENTRALIZED)
  assert names == {'FaaS-MASeA', 'FaaS-MASeA(r)', 'PG', 'Greedy-RR', 'centralized', 'selfish-centralized'}
  result = subprocess.run([sys.executable, str(ROOT/'run.py'), '--list'], capture_output=True, text=True, check=True)
  assert set(json.loads(result.stdout)) == names


@pytest.mark.parametrize('name', list(DECENTRALIZED))
@pytest.mark.parametrize('workers', [0, 2])
def test_decisions_objectives_and_messages_match_expected_results(name, workers, tmp_path):
  reference = json.loads((ROOT/'tests/expected-results.json').read_text())[name]
  config = json.loads((ROOT/'config/paper.json').read_text())
  source = ROOT/'examples/tiny'
  config.update(base_solution_folder=str(tmp_path), max_steps=2, max_run_time=2, pg_diagnostics=True)
  config['limits'] = {'instance_type':'materialized','path':str(source),'load':{}}
  original = deepcopy(config)
  folder = Path(DECENTRALIZED[name](config, workers, log_on_file=True, disable_plotting=True))
  assert config == original
  saved = json.loads((folder/'config.json').read_text())
  assert saved.get('algorithm', name) == name
  values = pd.read_csv(folder/'obj.csv')
  assert list(values.columns) == [name]
  np.testing.assert_allclose(values.iloc[:,-1], reference['welfare'], atol=1e-10, rtol=0)
  solution, replicas, detailed, *_ = load_solution(str(folder), 'LSPc')
  data,traces,agents,_ = load_materialized_instance(source)
  for t in range(2):
    arrays = encode_solution(len(agents),data[None]['Nf'][None],solution,detailed,replicas,t)[:4]
    current = update_data(data, {'incoming_load':get_current_load(traces,list(agents),t)})
    validate_centralized_solution(*arrays,current)
    for got, expected in zip(arrays,reference['arrays'][t]):
      np.testing.assert_allclose(got,expected,atol=1e-10,rtol=0)
  metrics = json.loads((folder/'communication.json').read_text())
  assert metrics['complete'] and metrics['method'] == name
  for key in ('data_messages','control_messages','messages','items','request_units'):
    assert metrics[key] == reference['communication'][key]
  if name == 'PG':
    assert (pd.read_csv(folder/'pg_diagnostics.csv')['x_changed_entries'] == 0).all()


@pytest.mark.parametrize('name', list(methods.CENTRALIZED))
def test_centralized_references_match_expected_results(name):
  reference = json.loads((ROOT/'tests/expected-results.json').read_text())[name]
  row = solve(ROOT/'examples/tiny',30,threads=1,gap=1e-6,**methods.CENTRALIZED[name])
  assert row['status'] == reference['status'] == 'ok'
  assert row['feasible']
  assert row['welfare'] == pytest.approx(reference['welfare'],abs=1e-10)


def test_cli_runs_and_refuses_overwrite(tmp_path):
  command = [sys.executable,str(ROOT/'run.py'),'--algorithm','FaaS-MASeA(r)',
             '--instance',str(ROOT/'examples/tiny'),'--output',str(tmp_path/'run')]
  subprocess.run(command,check=True,capture_output=True,text=True)
  result = tmp_path/'run/result.json'
  before = result.read_bytes()
  assert json.loads(before)['status'] == 'ok'
  assert subprocess.run(command,capture_output=True).returncode != 0
  assert result.read_bytes() == before


@pytest.mark.parametrize('name', ['FaaS-MASeA','FaaS-MASeA(r)','PG'])
def test_excluded_variants_cannot_be_enabled(name):
  kwargs = {'fixed_x':False} if name == 'PG' else {'refine_welfare':True}
  with pytest.raises(ValueError):
    DECENTRALIZED[name]({},0,**kwargs)


def test_obsolete_runner_entry_points_are_removed():
  import run_centralized_model
  import run_faasmacro
  import run_faasmadea
  for module in (run_centralized_model, run_faasmacro, run_faasmadea):
    assert not hasattr(module, 'run')


@pytest.mark.parametrize('name', list(methods.CENTRALIZED))
def test_centralized_cli_records_reference_status(name, tmp_path):
  command = [sys.executable, str(ROOT/'run.py'), '--algorithm', name,
             '--instance', str(ROOT/'examples/tiny'), '--output', str(tmp_path/'run'),
             '--time-limit', '30']
  subprocess.run(command, check=True, capture_output=True, text=True)
  result = json.loads((tmp_path/'run/result.json').read_text())
  assert result['algorithm'] == name
  assert result['status'] == 'ok'
  assert result['details']['feasible']
  expected = json.loads((ROOT/'tests/expected-results.json').read_text())[name]['welfare']
  assert result['welfare'] == pytest.approx([expected], abs=1e-10)


def test_replica_allocation_matches_exhaustive_search():
  from itertools import product
  from models.dp_solver import minimize_replica_costs
  costs = [[0., -2., -3.], [0., -4., -5.]]
  memory = np.array([2, 3])
  for budget in range(9):
    result = minimize_replica_costs(costs, memory, budget)
    feasible = [r for r in product(range(3), repeat=2) if np.dot(r, memory) <= budget]
    optimum = min(sum(costs[f][r[f]] for f in range(2)) for r in feasible)
    assert np.dot(result, memory) <= budget
    assert sum(costs[f][result[f]] for f in range(2)) == optimum
  with pytest.raises(ValueError, match='positive integer'):
    minimize_replica_costs(costs, np.array([0, 3]), 6)
  with pytest.raises(ValueError, match='No feasible'):
    minimize_replica_costs([[float('inf'), 0.]], np.array([2]), 1)
