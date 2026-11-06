"""Observer-only accounting under the documented logical batched protocol v1.

These are modelled endpoint deliveries, not measured network packets or bytes.
Only explicitly supported runners may publish complete communication metrics.
"""
import csv
import json
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

import numpy as np

from utils import step_time

MODEL = 'dfaas-logical-batched-v1'
SUPPORTED = {'FaaS-MASeA', 'FaaS-MASeA(r)', 'PG', 'Greedy-RR'}
COLUMNS = ('step', 'round', 'kind', 'messages', 'items', 'request_units')
_session = ContextVar('communication_session', default=None)


@contextmanager
def session(enabled, method, folder):
  if type(enabled) is not bool:
    raise ValueError('communication_metrics must be a boolean')
  if enabled and method not in SUPPORTED:
    raise ValueError(f'communication metrics unsupported for {method}')
  state = dict(step=0, round=0, rows={}, complete=False) if enabled else None
  token = _session.set(state)
  try:
    yield
  finally:
    _session.reset(token)
    if state is not None:
      with step_time.untimed():
        rows = list(state['rows'].values())
        with (Path(folder)/'communication.csv').open('w') as stream:
          writer = csv.DictWriter(stream, fieldnames=COLUMNS)
          writer.writeheader()
          writer.writerows(rows)
        totals = {key: sum(r[key] for r in rows) for key in COLUMNS[3:]}
        control = sum(r['messages'] for r in rows if r['kind'] == 'pg_turn')
        (Path(folder)/'communication.json').write_text(json.dumps(dict(
          schema_version=1, model=MODEL, method=method, complete=state['complete'],
          data_messages=totals['messages']-control, control_messages=control,
          **totals), indent=2))


def observer(function):
  @wraps(function)
  def wrapped(*args, **kwargs):
    if _session.get() is not None:
      with step_time.untimed():
        return function(*args, **kwargs)
  return wrapped


@observer
def begin_step(step):
  _session.get().update(step=step, round=0)


@observer
def begin_round(round_index):
  _session.get()['round'] = round_index


@observer
def finish():
  _session.get()['complete'] = True


def _add(kind, messages, items, units=0.):
  if not messages:
    return
  state = _session.get()
  key = (state['step'], state['round'], kind)
  row = state['rows'].setdefault(key, dict(zip(COLUMNS, (*key, 0, 0, 0.))))
  row['messages'] += int(messages)
  row['items'] += int(items)
  row['request_units'] += float(units)


def _rows(payload):
  if hasattr(payload, 'columns'):
    columns = ['i', 'j', 'f'] + (['d'] if 'd' in payload.columns else [])
    return payload[columns].itertuples(index=False, name=None)
  return iter(payload)


@observer
def announce(neighborhood, functions):
  # One state vector from each destination to each eligible source.
  messages = np.count_nonzero(neighborhood) - np.count_nonzero(np.diag(neighborhood))
  _add('state_advertisement', messages, messages * (functions + 1))


@observer
def requests(kind, payload):
  pairs, records, items, units = set(), set(), 0, 0.
  for row in _rows(payload):
    if row[0] == row[1]:
      continue
    pairs.add((int(row[0]), int(row[1])))
    records.add(tuple(row[:3]))
    items += 1
    units += float(row[3]) if len(row) > 3 else 0.
  if kind == 'capacity_request':
    # Unit offers with identical terms travel as one quantity-bearing record.
    items = len(records)
    if hasattr(payload, 'columns'):
      columns = ['i', 'j', 'f'] + [c for c in ('b', 'utility') if c in payload.columns]
      items = len(payload.loc[payload.i != payload.j, columns].drop_duplicates())
  _add(kind, len(pairs), items, units)


@observer
def capacity_replies(bids, delta=None):
  # Reply to every bidder, including rejections; notify displaced incumbents.
  records = {(int(i), int(j), int(f)) for i, j, f, *_ in _rows(bids) if i != j}
  if delta is not None:
    records.update((int(i), int(j), int(f)) for i, j, f in zip(*np.nonzero(delta)) if i != j)
  _add('capacity_reply', len({(j, i) for i, j, _ in records}), len(records))


@observer
def pg_visit(node, neighbors, functions):
  n = len(set(neighbors) - {node})
  _add('pg_turn', 1, 1)
  _add('pg_state_query', n, n)
  _add('pg_state_reply', n, n * (functions + 1))


@observer
def pg_commit(node, old, new):
  changed = new != old
  changed[node, :] = False
  _add('pg_commit', np.count_nonzero(changed.any(axis=1)), np.count_nonzero(changed),
       np.abs(new - old)[changed].sum())
