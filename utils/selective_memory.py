"""Sender-local memory requests; no additional fields are sent to a receiver."""
from utils import step_time


class SelectiveMemory:
  """Per-step attempts for each (buyer, function); reset after demand changes."""

  def __init__(self):
    self.attempts = {}
    self.pending = False

  @step_time.parallel
  def select(self, bids, omega, prices, data, latency, options):
    self.pending = False
    chosen = []
    # Confirmed bids omit satisfied buyers entirely. Observe those changes too,
    # so a later displacement of accepted traffic starts a fresh attempt cycle.
    for key in step_time.timed(list(self.attempts), key=lambda key: key[0]):
      if self.attempts[key][0] != float(omega[key]):
        del self.attempts[key]
    groups = bids.groupby(['i', 'f'], sort=False)
    for (i, f), group in step_time.timed(groups, key=lambda item: item[0][0]):
      demand = float(omega[i, f])
      if demand <= 1e-6:
        self.attempts.pop((i, f), None)
        continue
      previous, tried = self.attempts.get((i, f), (None, set()))
      if previous != demand:
        tried = set()
      candidates = set(int(j) for j in group.j) - tried
      if candidates:
        # Only own outgoing beta and already advertised prices/latency are read.
        j = max(sorted(candidates), key=lambda j:
          data[None]['beta'][i+1, j+1, f+1] - prices[j, f]
          - options.get('latency_weight', 0.) * latency[i, j])
        chosen.append(group.index[group.j == j][0])
        tried.add(j)
        self.pending |= len(candidates) > 1
      self.attempts[i, f] = (demand, tried)
    return bids.loc[chosen].reset_index(drop=True)

  def retry(self, it, max_iterations, runtime, time_limit):
    """Permit a next neighbor after a soft stop, never beyond the run's budget."""
    return self.pending and it < max_iterations - 1 and runtime < time_limit
