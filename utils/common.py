from scipy.stats import truncnorm
from typing import Tuple
import numpy as np
import json
import ast
import os
from datetime import datetime


class NpEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.floating):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return super(NpEncoder, self).default(obj)


def delete_tuples(_dict: dict):
  """Delete tuple keys recursively from all of the dictionaries"""
  serializable_dict = {}
  for key, value in _dict.items():
    new_key = str(key) if isinstance(key, tuple) or key is None else key
    if isinstance(value, dict):
      serializable_dict[new_key] = delete_tuples(value)
    else:
      serializable_dict[new_key] = value
  return serializable_dict


def float_to_int(fval: float, approx_tol: float) -> int:
  ival = None
  if np.floor(fval) > 0 and ((fval / np.floor(fval) - 1) > approx_tol):
    ival = int(np.ceil(fval))
  else:
    if int(np.floor(fval)) == 0 and (fval > approx_tol):
      ival = int(np.ceil(fval))
    else:
      ival = int(fval)
  return ival


def generate_random_float(
    rng: np.random.Generator, limits: dict
  ) -> float:
  val = 0
  if "avg" in limits:
    val = generate_truncated_normal(
      rng, limits["avg"], limits["std"], limits["min"], limits["max"]
    )
  else:
    if "min" in limits and "max" in limits:
      if limits["max"] > limits["min"]:
        val = round(rng.uniform(limits["min"], limits["max"]), 3)
      else:
        val = limits["min"]
    elif "values_from" in limits:
      val = rng.choice(limits["values_from"])
    else:
      raise ValueError("Missing values to define limits")
  return val


def generate_random_int(
    rng: np.random.Generator, limits: dict
  ) -> int:
  val = 0
  if "min" in limits and "max" in limits:
    val = int(rng.integers(limits["min"], limits["max"], endpoint = True))
  elif "values_from" in limits:
    val = rng.choice(limits["values_from"])
  else:
    raise ValueError("Missing values to define limits")
  return val


def generate_truncated_normal(
    rng: np.random.Generator,
    mean: float = 0.0,
    sd: float = 1.0,
    low: float = 0.0,
    upp: float = 10.0
  ) -> float:
  gen = truncnorm(
    (low - mean) / sd,
    (upp - mean) / sd,
    loc = mean,
    scale = sd
  )
  return gen.rvs(random_state = rng)


def int_keys_decoder(pairs: dict) -> dict:
  return {int(k): v for k, v in pairs}


def load_base_instance(folder: str) -> Tuple[dict, dict]:
  # base instance
  base_instance = {}
  with open(
    os.path.join(folder, "base_instance_data.json"), "r"
  ) as istream:
    data = json.load(istream)
    base_instance = restore_types(data)
  # load limits
  load_limits = {}
  with open(
    os.path.join(folder, "load_limits.json"), "r"
  ) as istream:
    data = json.load(istream)
    load_limits = restore_types(data)
  return base_instance, load_limits


def load_configuration(config_file: str) -> dict:
  """Load configuration file"""
  config = {}
  with open(config_file, "r") as istream:
    config = json.load(istream)
  return config


def load_requests_traces(
    folder: str, load_filtered: bool = False
  ) -> Tuple[dict, int, int, int]:
  # load time info
  mt = 0
  Mt = 1
  ts = 1
  if os.path.exists(os.path.join(folder, "config.json")):
    config = load_configuration(os.path.join(folder, "config.json"))
    mt = config.get("min_run_time", 0)
    Mt = config.get("max_run_time", config["max_steps"])
    ts = config.get("run_time_step", 1)
  # load requests
  requests = {}
  filename = "input_requests_traces.json" if not load_filtered else (
    f"load/norej_trace-{mt}_{Mt}_{ts}.json"
  )
  with open(os.path.join(folder, filename), "r") as ist:
    requests = {
      int(f): {
        int(a): np.array(r) for a,r in v.items()
      } for f,v in json.load(ist).items()
    }
  return requests, mt, Mt, ts


def reconcile_paths(base_solution_folder: str, fpath: str) -> str:
  f_complete_path = None
  if fpath is not None:
    if fpath.startswith("/"):
      f_complete_path = fpath
    elif "solutions/" in base_solution_folder and "solutions/" in fpath:
      common_base = base_solution_folder.split("solutions/")[0]
      common_end = fpath.split("solutions/")[1]
      f_complete_path = os.path.join(common_base, "solutions", common_end)
    else:
      f_complete_path = fpath
  return f_complete_path


def restore_types(serialized_dict: dict):
  """Restore the original types"""
  _dict = {}
  for key, value in serialized_dict.items():
    new_key = key
    try:
      new_key = ast.literal_eval(key)
    except ValueError:
      pass
    if isinstance(value, dict):
      _dict[new_key] = restore_types(value)
    else:
      _dict[new_key] = value
  return _dict


def run_steps(config: dict) -> range:
  """Simulated steps: min_run_time to max_run_time (excluded) by run_time_step; equal bounds run that step."""
  min_run_time = config.get("min_run_time", 0)
  max_run_time = config.get("max_run_time", config["max_steps"])
  run_time_step = config.get("run_time_step", 1)
  stop = max_run_time + run_time_step if max_run_time == min_run_time else max_run_time
  return range(min_run_time, stop, run_time_step)


def checkpoint_due(t: int, steps: range, interval: int) -> bool:
  """Every `interval` steps and at the last simulated step."""
  return t % interval == 0 or t == steps[-1]


def run_timestamp() -> str:
  """Name of a run folder: the only calendar-clock reading of the core modules."""
  return datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
