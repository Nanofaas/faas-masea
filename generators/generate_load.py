from utils.common import load_requests_traces, NpEncoder
from generators.load_generator import LoadGenerator

import numpy as np
import json
import os


def generate_load_traces(
    limits: dict,
    max_steps: int = 100,
    seed: int = 4850,
    trace_type: str = "clipped",
    solution_folder: str = None,
    enable_plotting: bool = True
  ) -> dict:
  input_requests_traces = {}
  if trace_type == "load_existing" or trace_type == "load_filtered":
    input_requests_traces = load_requests_traces(
      limits["load_existing"], trace_type == "load_filtered"
    )[0]
  else:
    LG = LoadGenerator(average_requests = 100, amplitude_requests = 50)
    rng = np.random.default_rng(seed = seed)
    # generate trace for all request classes
    input_requests_traces = {}
    for function, function_limits in limits.items():
      new_trace_type = trace_type
      if trace_type == "fixed_sum_minmax" and function%2 == 0: # min for even
        new_trace_type = "fixed_sum_min"
      elif trace_type == "fixed_sum_minmax" and function%2 != 0: # max for odd
        new_trace_type = "fixed_sum_max"
      input_requests_traces[function] = LG.generate_traces(
        max_steps = max_steps,
        limits = function_limits,
        rng = rng,
        trace_type = new_trace_type, #f"manual{function}"#
        only_integer_values = True
      )
      # plot trace (if required)
      if (len(limits)<=10 and len(function_limits)<=10 and enable_plotting):
        plot_filename = None
        if solution_folder is not None:
          plot_filename = os.path.join(solution_folder, "load")
          os.makedirs(plot_filename, exist_ok = True)
          plot_filename = os.path.join(plot_filename, f"f{function}.png")
        LG.plot_input_load(
          input_requests_traces[function], plot_filename = plot_filename
        )
  # save traces (if required)
  if solution_folder is not None:
    with open(
      os.path.join(solution_folder, "input_requests_traces.json"), "w"
    ) as istream:
      istream.write(
        json.dumps(input_requests_traces, indent = 2, cls = NpEncoder)
      )
  return input_requests_traces
