"""Logical protocol accounting: batches, records and request quantities differ."""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils import communication, step_time


def test_batched_messages_include_rejections_and_displaced_buyers(tmp_path):
  graph = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
  bids = pd.DataFrame({'i': [0, 0, 0, 2], 'j': [1, 1, 1, 1],
                       'f': [0, 0, 1, 0], 'd': [2, 3, 1, 4]})
  delta = np.zeros((3, 3, 2))
  delta[2, 0, 0] = -1  # displaced incumbent, even without a new bid
  with communication.session(True, 'FaaS-MASeA(r)', tmp_path):
    communication.begin_step(0)
    communication.begin_round(0)
    communication.announce(graph, 2)
    communication.requests('capacity_request', bids)
    communication.capacity_replies(bids, delta)
    communication.requests('memory_request', [(0, 1, 0), (0, 1, 1)])
    communication.finish()
  rows = {r['kind']: r for r in csv.DictReader((tmp_path/'communication.csv').open())}
  assert [(rows[k]['messages'], rows[k]['items'], float(rows[k]['request_units'])) for k in rows] == [
    ('4', '12', 0.), ('2', '3', 10.), ('3', '4', 0.), ('1', '2', 0.)]
  meta = json.loads((tmp_path/'communication.json').read_text())
  assert meta['complete'] and meta['messages'] == 10 and meta['items'] == 21
  assert meta['model'] == 'dfaas-logical-batched-v1'


def test_unit_bids_are_compact_but_distinct_prices_and_rounds_remain(tmp_path):
  bids = pd.DataFrame({'i': [0]*1001, 'j': [1]*1001, 'f': [0]*1001,
                       'd': [1]*1001, 'b': [2.]*1000+[3.], 'utility': [4.]*1001})
  with communication.session(True, 'FaaS-MASeA', tmp_path):
    for round_index in range(2):
      communication.begin_round(round_index)
      communication.requests('capacity_request', bids)
    communication.finish()
  rows = list(csv.DictReader((tmp_path/'communication.csv').open()))
  assert len(rows) == 2
  assert all((r['messages'], r['items'], float(r['request_units'])) == ('1', '2', 1001.)
             for r in rows)
  meta = json.loads((tmp_path/'communication.json').read_text())
  assert meta['messages'] == 2 and meta['items'] == 4 and meta['request_units'] == 2002.




def test_pg_counts_turn_queries_and_only_committed_updates(tmp_path):
  with communication.session(True, 'PG', tmp_path):
    communication.begin_step(3)
    communication.begin_round(2)
    communication.pg_visit(0, {1, 2}, 2)
    old = np.zeros((3, 2))
    new = old.copy(); new[1] = [2, 1]
    communication.pg_commit(0, old, new)
    communication.pg_commit(0, new, new)
    communication.finish()
  rows = {r['kind']: r for r in csv.DictReader((tmp_path/'communication.csv').open())}
  assert {k: int(v['messages']) for k,v in rows.items()} == {
    'pg_turn': 1, 'pg_state_query': 2, 'pg_state_reply': 2, 'pg_commit': 1}
  assert rows['pg_commit']['items'] == '2'
  assert rows['pg_commit']['request_units'] == '3.0'
  assert all(r['step']=='3' and r['round']=='2' for r in rows.values())
  meta = json.loads((tmp_path/'communication.json').read_text())
  assert meta['data_messages'] == 5 and meta['control_messages'] == 1


def test_observer_time_is_excluded_and_failure_does_not_leak(tmp_path, monkeypatch):
  now = [0.]
  monkeypatch.setattr(step_time, 'perf_counter', lambda: now[0])
  def slow_rows():
    now[0] += 10
    yield (0, 1, 0, 2)
  with pytest.raises(RuntimeError, match='failed'):
    with communication.session(True, 'Greedy-RR', tmp_path):
      communication.begin_step(0)
      clock = step_time.start_step()
      communication.requests('capacity_request', slow_rows())
      assert step_time.end_step(clock)['algorithm_s'] == 0
      raise RuntimeError('failed')
  assert not json.loads((tmp_path/'communication.json').read_text())['complete']
  # Disabled/no active session must not even consume the payload.
  def forbidden():
    raise AssertionError('observer executed while disabled')
    yield
  communication.requests('capacity_request', forbidden())
  with communication.session(False, 'unsupported', tmp_path/'disabled'):
    communication.requests('capacity_request', forbidden())
  assert not (tmp_path/'disabled').exists()
  with pytest.raises(ValueError, match='unsupported'):
    with communication.session(True, 'unsupported', tmp_path/'unsupported'):
      pass
