import matplotlib.pyplot as plt
import numpy as np


def rescale(
    val: float,
    in_min: float,
    in_max: float,
    out_min: float,
    out_max: float
  ) -> float:
  """
  Rescale value from the original interval [in_min, in_max] to the new
  range [out_min, out_max]
  """
  if in_min == in_max:
    return out_min
  return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))


class LoadGenerator:
  def __init__(
      self,
      # default values are calculated to match the average mean of the real
      # traces
      average_requests: int = 50,
      amplitude_requests: int = 100,
      noise_ratio: float = 0.1,
      unique_periods: int = 3
    ) -> None:
    self.average_requests = average_requests
    self.amplitude_requests = amplitude_requests
    self.noise_ratio = noise_ratio
    self.unique_periods = unique_periods

  def _impose_system_workload(
      self, total_workload, input_requests: dict
    ) -> dict:
    # Allocate whole requests (or thousandths for real-valued traces) by largest
    # remainder. Clip the source before normalizing so the later nonnegative
    # clamp cannot change the requested system total.
    base_signals = np.array(list(input_requests.values()))
    integer = np.issubdtype(base_signals.dtype, np.integer)
    base_signals = np.maximum(base_signals, 0)
    num_agents, num_timesteps = base_signals.shape
    totals = np.broadcast_to(total_workload, (num_timesteps,))
    scale = 1 if integer else 1000
    workloads = np.zeros(base_signals.shape, dtype=int if integer else float)
    for t, requested in enumerate(totals):
      target = int(round(max(0, requested) * scale))
      total = base_signals[:, t].sum()
      shares = base_signals[:, t] / total if total else np.full(num_agents, 1 / num_agents)
      quotas = shares * target
      units = np.floor(quotas).astype(int)
      remainder = target - units.sum()
      order = np.argsort(-(quotas - units), kind="stable")
      units[order[:remainder]] += 1
      workloads[:, t] = units if integer else units / scale
    return dict(zip(input_requests, workloads))

  def _synthetic_sinusoidal_input_requests(
      self,
      max_steps: int,
      agents: list,
      rng: np.random.Generator,
      only_integer_values: bool = False
    ) -> dict:
    """
    Generates the input requests for the given agents with the given length,
    clipping the values within the given bounds and using the given rng to
    generate the synthesized data.

    limits must be a dictionary whose keys are the agent ids, and each agent has
    two sub-keys: "min" for the minimum value and "max" for the maximum value.

    Returns a dictionary whose keys are the agent IDs and whose value is an
    np.ndarray containing the input requests for each step.
    """
    # generate
    input_requests = {}
    steps = np.arange(max_steps)
    for agent in agents:
      # Note: with default max_stes, the period changes every 96 steps
      # (max_steps = 288). We first generate the periods and expand the array
      # to match the max_steps. If max_steps is not a multiple of 96, some
      # elements must be appended at the end, hence the resize call.
      repeats = max(1, max_steps // self.unique_periods)
      periods = rng.uniform(15, high = 100, size = self.unique_periods)
      periods = np.repeat(periods, repeats)  # Expand the single values.
      periods = np.resize(periods, periods.size + max_steps - periods.size)
      # base + noise
      base_input = self.average_requests + self.amplitude_requests * np.sin(
        2 * np.pi * steps / periods
      )
      noisy_input = base_input + self.noise_ratio * rng.normal(
        0, self.amplitude_requests, size=max_steps
      )
      requests = None
      if only_integer_values:
        requests = np.asarray(noisy_input, dtype = np.int32)
      else:
        requests = np.asarray(noisy_input)
      # save
      input_requests[agent] = requests
    return input_requests

  def generate_traces(
      self,
      max_steps: int,
      limits: dict,
      rng: np.random.Generator,
      trace_type: str = "clipped",
      only_integer_values: bool = False
    ):
    input_requests = self._synthetic_sinusoidal_input_requests(
      max_steps, limits.keys(), rng, only_integer_values
    )
    if trace_type in ["clipped", "sinusoidal"]:
      # Ensure the number of requests stays in [min, max]
      for agent, requests in input_requests.items():
        minr = limits[agent]["min"]
        maxr = limits[agent]["max"]
        if trace_type == "clipped":
          # Clip the excess values respecting the minimum and maximum values
          # for the input requests observation.
          clipped = np.clip(requests.astype(float), minr, maxr)
          requests = clipped.astype(np.int32) if only_integer_values else clipped
        elif trace_type == "sinusoidal":
          # Rescale
          in_min = requests.min()
          in_max = requests.max()
          if only_integer_values:
            requests = np.array([
              int(rescale(r, in_min, in_max, minr, maxr)) for r in requests
            ])
          else:
            requests = np.array([
              round(rescale(r, in_min, in_max, minr, maxr), 3) for r in requests
            ])
        input_requests[agent] = requests
    elif trace_type.startswith("fixed_sum"):
      total_workload = 0.0
      values = list(limits.values())
      numeric_values = [v["max"] if isinstance(v, dict) else v for v in values]
      if trace_type == "fixed_sum":
        total_workload = sum(numeric_values) / len(numeric_values)
      else:
        total_workload = min(numeric_values) if trace_type.endswith(
          "min"
        ) else max(numeric_values)
      input_requests = self._impose_system_workload(
        total_workload, input_requests
      )
    elif trace_type == "flat":
      # Match the integer truncation used by clipped and sinusoidal traces.
      input_requests = {
        agent: np.full(max_steps, limits[agent], dtype=int if only_integer_values else float)
        for agent in limits
      }
    else:
      raise KeyError(f"Trace type `{trace_type}` is not supported")
    # ensure that everything, anyway, stays above zero
    for agent in input_requests:
      input_requests[agent] = np.array([
        max(0, r) for r in input_requests[agent]
      ])
    return input_requests

  @staticmethod
  def plot_input_load(
      input_requests: dict, current_step: int = 0, plot_filename: str = None
    ):
    _, ax = plt.subplots()
    for agent, incoming_load in input_requests.items():
      max_steps = len(incoming_load)
      ax.plot(
        range(max_steps),
        incoming_load,
        ".-",
        label = f"agent {agent}"
      )
    # highlight current time
    ax.axvline(
      x = current_step,
      color = "k",
      linestyle = "dashed"
    )
    # axis properties
    ax.set_xlabel("Control time period $t$", fontsize = 14)
    ax.set_ylabel("Load [req/s]", fontsize = 14)
    ax.legend(fontsize = 14)
    plt.grid()
    if plot_filename is not None:
      plt.savefig(
        plot_filename, dpi = 300, format = "png", bbox_inches = "tight"
      )
      plt.close()
    else:
      plt.show()
