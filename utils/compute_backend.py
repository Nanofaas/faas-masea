"""Run-scoped device selection. Opening a session never imports CUDA."""
import os
from contextlib import contextmanager
from contextvars import ContextVar

_session = ContextVar('compute_session', default=None)
_coverage = ContextVar('compute_coverage', default=None)
_dp_selection = ContextVar('compute_dp_selection', default='cpu')


def compute_device(options: dict) -> str:
  device = options.get('compute_device', 'cpu')
  if device not in ('cpu', 'cuda'):
    raise ValueError('compute_device must be cpu or cuda')
  return device


@contextmanager
def compute_session(options: dict):
  # Validate dp_device independently, including its use_dp requirement.
  from models.local_sp import dp_device
  selected_dp = dp_device(options)
  token = _session.set((os.getpid(), compute_device(options)))
  dp_token = _dp_selection.set(selected_dp)
  coverage = []
  coverage_token = _coverage.set(coverage)
  try:
    yield coverage
  finally:
    _coverage.reset(coverage_token)
    _session.reset(token)
    _dp_selection.reset(dp_token)


def cuda_dp_compute_enabled() -> bool:
  """Extended DP adapters require both validated options in the owning process."""
  return cuda_compute_enabled() and _dp_selection.get() == 'cuda'


def cuda_compute_enabled() -> bool:
  session = _session.get()
  return session is not None and session == (os.getpid(), 'cuda')


def record_operation(operation, backend, count, *, shape=None, reason='', device_bytes=0):
  """Observer-only coverage; no synchronization or device queries."""
  if not cuda_compute_enabled():
    return
  from utils import step_time
  with step_time.untimed():
    row = dict(operation=operation,backend=backend,count=count,shape=shape,reason=reason)
    if device_bytes:
      row['device_bytes'] = device_bytes
    _coverage.get().append(row)
