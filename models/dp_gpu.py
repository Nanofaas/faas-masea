"""Optional numeric CUDA knapsack batches. Importing this module never touches CUDA."""
from functools import lru_cache, reduce
from math import gcd, isfinite
from time import perf_counter

import numpy as np

_MAX_DEVICE_BYTES = 64 * 1024 * 1024
_INT_MAX = np.iinfo(np.int32).max


def _shape(costs, memory, ram_cap):
  if (not len(costs) or len(costs) != len(memory) or not isfinite(ram_cap)
      or ram_cap < 0 or ram_cap != int(ram_cap)
      or any(not isfinite(m) or m <= 0 or m != int(m) for m in memory)):
    raise ValueError('GPU DP requires positive integer memory and nonnegative integer RAM')
  scale = reduce(gcd, map(int, memory))
  budget = int(ram_cap) // scale
  nf, nl = len(costs), max(map(len, costs))
  if (not nl or any(not len(c) for c in costs) or max(nf, nl, budget + 1) > _INT_MAX
      or any(int(m) // scale > _INT_MAX for m in memory)):
    raise ValueError('GPU DP dimensions exceed int32')
  return nf, nl, budget + 1


def _buffer_bytes(count, nf, nl, nb):
  # float64 costs + int32 memory/lengths/output + budget/nf/status + DP buffers/choices
  return count * (8 * nf * nl + 12 * nf + 12 + 16 * nb + 4 * nf * nb)


def _padded_bytes(items):
  shapes = [_shape(*item) for item in items]
  return _buffer_bytes(len(items), *map(max, zip(*shapes)))


def gpu_problem_fits(costs, memory, ram_cap):
  """Driver-free policy for a single problem's index and device memory bounds."""
  try:
    return _padded_bytes([(costs, memory, ram_cap)]) <= _MAX_DEVICE_BYTES
  except (ValueError, OverflowError):
    return False


def _chunks(items):
  chunk, shape = [], (0, 0, 0)
  for item in items:
    item_shape = _shape(*item)
    if _buffer_bytes(1, *item_shape) > _MAX_DEVICE_BYTES:
      raise ValueError('GPU DP problem exceeds device buffer/index limits')
    padded = tuple(map(max, zip(shape, item_shape)))
    if chunk and _buffer_bytes(len(chunk) + 1, *padded) > _MAX_DEVICE_BYTES:
      yield chunk
      chunk, padded = [], item_shape
    chunk.append(item)
    shape = padded
  if chunk:
    yield chunk


@lru_cache(maxsize=1)
def _get_kernel():
  # Deliberately lazy: CPU paths and singleton/incompatible batches need no CUDA package.
  from numba import cuda
  # cuda must be a module global for Numba's CUDA intrinsics (also in simulator mode).
  globals()['cuda'] = cuda

  @cuda.jit(fastmath=False)
  def kernel(costs, memory, lengths, budgets, nfs, best, other, choices, replicas, valid):
    n = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    nb = budgets[n] + 1
    for b in range(tid, nb, cuda.blockDim.x):
      best[n, b] = 0.
    cuda.syncthreads()
    for f in range(nfs[n]):
      for b in range(tid, nb, cuda.blockDim.x):
        value = float('inf')
        selected = 0
        hi = min(lengths[n, f] - 1, b // memory[n, f])
        for k in range(hi + 1):
          candidate = best[n, b - k * memory[n, f]] + costs[n, f, k]
          if candidate < value:
            value = candidate
            selected = k
        other[n, b] = value
        choices[n, f, b] = selected
      cuda.syncthreads()
      for b in range(tid, nb, cuda.blockDim.x):
        best[n, b] = other[n, b]
      cuda.syncthreads()
    if tid == 0:
      value = float('inf')
      budget = 0
      has_finite = False
      for b in range(nb):
        if best[n, b] > -float('inf') and best[n, b] < float('inf'):
          has_finite = True
        if best[n, b] < value:
          value = best[n, b]
          budget = b
      valid[n] = has_finite
      if valid[n]:
        for f in range(nfs[n] - 1, -1, -1):
          k = choices[n, f, budget]
          replicas[n, f] = k
          budget -= k * memory[n, f]
  return cuda, kernel


def _run_chunk(items, *, threads=128, profile=None):
  started = perf_counter()
  shapes = [_shape(*item) for item in items]
  nf, nl, nb = map(max, zip(*shapes))
  count = len(items)
  costs = np.full((count, nf, nl), np.inf, dtype=np.float64)
  memory = np.ones((count, nf), dtype=np.int32)
  lengths = np.zeros((count, nf), dtype=np.int32)
  budgets = np.empty(count, dtype=np.int32)
  nfs = np.empty(count, dtype=np.int32)
  for n, ((curves, mem, cap), (fs, _, bs)) in enumerate(zip(items, shapes)):
    scale = reduce(gcd, map(int, mem))
    budgets[n], nfs[n] = bs - 1, fs
    memory[n, :fs] = [int(m) // scale for m in mem]
    for f, curve in enumerate(curves):
      lengths[n, f] = len(curve)
      costs[n, f, :len(curve)] = curve
  packed = perf_counter()
  cuda, kernel = _get_kernel()
  # Initialization, JIT and allocations remain inside the synchronized batch wall time.
  inputs = [cuda.to_device(a) for a in (costs, memory, lengths, budgets, nfs)]
  best = cuda.device_array((count, nb), dtype=np.float64)
  other = cuda.device_array_like(best)
  choices = cuda.device_array((count, nf, nb), dtype=np.int32)
  replicas = cuda.device_array((count, nf), dtype=np.int32)
  valid = cuda.device_array(count, dtype=np.int32)
  cuda.synchronize()
  copied = perf_counter()
  kernel[count, threads](*inputs, best, other, choices, replicas, valid)
  cuda.synchronize()
  executed = perf_counter()
  result, feasible = replicas.copy_to_host(), valid.copy_to_host()
  cuda.synchronize()
  finished = perf_counter()
  if profile is not None:
    profile.append(dict(packing_s=packed-started, upload_alloc_s=copied-packed,
                        kernel_s=executed-copied, download_s=finished-executed,
                        device_bytes=_padded_bytes(items), threads=threads, problems=count))
  return [result[n, :fs].copy() if feasible[n] else None
          for n, (fs, _, _) in enumerate(shapes)]


def _minimize_batch(costs_batch, memory_batch, ram_caps, *, threads=128, profile=None):
  if not (len(costs_batch) == len(memory_batch) == len(ram_caps)):
    raise ValueError('GPU DP batch lengths differ')
  result = []
  planning_started = perf_counter()
  for chunk in _chunks(list(zip(costs_batch, memory_batch, ram_caps))):
    planning_s = perf_counter() - planning_started
    result.extend(_run_chunk(chunk, threads=threads, profile=profile))
    if profile is not None:
      profile[-1]['packing_s'] += planning_s
    planning_started = perf_counter()
  return result


def minimize_replica_costs_batch(costs_batch, memory_batch, ram_caps):
  """Return choices or per-problem None for infeasibility; CUDA errors propagate."""
  return _minimize_batch(costs_batch, memory_batch, ram_caps)


def _launch_device_dp(costs, memory, lengths, budgets, nfs, count, nf, nb):
  """Use the existing recurrence with curves already resident on the device.

  The adapter owns chunk limits, synchronization, and the complete phase clock.
  Returned allocations stay alive through their references until the launch ends.
  """
  cuda, kernel = _get_kernel()
  best = cuda.device_array((count, nb), dtype=np.float64)
  other = cuda.device_array_like(best)
  choices = cuda.device_array((count, nf, nb), dtype=np.int32)
  replicas = cuda.device_array((count, nf), dtype=np.int32)
  valid = cuda.device_array(count, dtype=np.int32)
  kernel[count,128](costs,memory,lengths,budgets,nfs,best,other,choices,replicas,valid)
  return replicas, valid, (best,other,choices)
