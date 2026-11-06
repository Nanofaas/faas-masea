from utils.common import (
  delete_tuples,
  float_to_int,
  load_configuration,
  NpEncoder
)
from functools import partial

from utils import run_frame, run_log, step_time
from utils.centralized import get_current_load  # noqa: F401 (re-exported)
from generators.generate_data import generate_data, update_data  # noqa: F401 (update_data re-exported)
from generators.generate_load import generate_load_traces
from remote_experiments.instances import (
  PAYLOAD_FILES,
  load_materialized_instance,
)
from postprocessing import plot_history  # noqa: F401 (re-exported)
from models.model import (
  BaseLoadManagementModel,
  LoadManagementModel,
  TightLoadManagementModel,
  PYO_VAR_TYPE
)
from models.selfish import SelfishLoadManagementModel

from networkx import draw_networkx, kamada_kawai_layout, Graph, is_planar
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from typing import Tuple
import pandas as pd
import numpy as np
import argparse
import json
import shutil
import os


VAR_TYPE = int if PYO_VAR_TYPE == pyo.NonNegativeIntegers else float


log = run_log.logger(__name__)




def compute_residual_capacity(data: dict, r: np.array) -> np.array:
  Nn = data[None]["Nn"][None]
  rho = np.zeros((Nn,))
  indices = data.get("indices", range(Nn))
  for n in indices:
    memory_capacity = data[None]["memory_capacity"][n+1]
    rho_idx = n if len(indices) > 1 else 0
    rho[rho_idx] = memory_capacity
    for f, memory_requirement in data[None]["memory_requirement"].items():
      rho[rho_idx] -= (memory_requirement * r[rho_idx,f-1])
  return rho


def compute_utilization(data: dict, solution: dict) -> np.array:
  x, y, _, r, xi, _, _, _ = extract_solution(data, solution)
  demand = data[None]["demand"]
  utilization = np.zeros(shape = x.shape)
  for (n,f), d in demand.items():
    if r[n-1, f-1] > 0:
      if xi is None or ((xi == 0).all() and (y != 0).any()):
        utilization[n-1, f-1] = d * (
          x[n-1, f-1] + y[:, n-1, f-1].sum()
        ) / r[n-1, f-1]
      else:
        utilization[n-1, f-1] = d * (
          x[n-1, f-1] + xi[n-1, :, f-1].sum()
        ) / r[n-1, f-1]
  return utilization


def extract_solution(
    data: dict, solution: dict, approx_tol: float = 1e-6
  ) -> Tuple[
    np.array, np.array, np.array, np.array, np.array, np.array, float
  ]:
  Nn = data[None]["Nn"][None]
  Nf = data[None]["Nf"][None]
  # extract values from solution dictionary
  x = np.zeros((Nn,Nf))
  omega = np.zeros((Nn,Nf))
  r = np.zeros((Nn,Nf))
  y = np.zeros((Nn,Nn,Nf))
  z = np.zeros((Nn,Nf))
  # -- local processing
  if "x" in solution:
    x = np.array(solution["x"], dtype = VAR_TYPE).reshape((Nn, Nf))
  elif "x_bar" in data[None]:
    x = np.array(
      list(data[None]["x_bar"].values()), dtype = VAR_TYPE
    ).reshape((Nn, Nf))
  # -- (detailed) offloading
  if "y" in solution:
    y = np.array(solution["y"], dtype = VAR_TYPE).reshape((Nn, -1, Nf))
  elif "y_bar" in data[None]:
    y = np.array(
      list(data[None]["y_bar"].values()), dtype = VAR_TYPE
    ).reshape((Nn, -1, Nf))
  elif "d" in solution:
    y = np.array(solution["d"], dtype = VAR_TYPE).reshape((Nn, -1, Nf))
  # -- rejections
  if "z" in solution:
    z = np.array(solution["z"], dtype = VAR_TYPE).reshape((Nn, Nf))
  elif "z_bar" in data[None]:
    z = np.array(
      list(data[None]["z_bar"].values()), dtype = VAR_TYPE
    ).reshape((Nn, Nf))
  # -- number of function replicas
  if "r" in solution:
    r = np.array(
      [
        float_to_int(
          rval, approx_tol
        ) for rval in np.array(solution["r"]).flatten()
      ],
      dtype = int
    ).reshape((Nn, Nf))
    if "r_bar" in data[None]:
      r_bar = np.array(
        [
          float_to_int(
            rval, approx_tol
          ) for rval in data[None]["r_bar"].values()
        ],
        dtype = int
      ).reshape((Nn, Nf))
      r += r_bar
  elif "r_bar" in data[None]:
    r = np.array(
      list(data[None]["r_bar"].values()), dtype = int
    ).reshape((Nn, Nf))
  # -- offloading
  if "omega" in solution:
    omega = np.array(solution["omega"], dtype = VAR_TYPE).reshape((Nn, -1))
  else:
    if "y" in solution or "d" in solution:
      for n in range(Nn):
        for f in range(Nf):
          omega[n,f] = y[n,:,f].sum()
    elif "omega_bar" in data[None]:
      omega = np.array(
        list(data[None]["omega_bar"].values()), dtype = VAR_TYPE
      ).reshape((Nn, -1))
  # compute xi
  xi = np.zeros((Nn,Nn,Nf))
  for n1 in range(Nn):
    for n2 in range(Nn):
      for f in range(Nf):
        xi[n2,n1,f] = y[n1,n2,f]
  # compute residual capacity
  rho = compute_residual_capacity(data, r)
  return x, y, z, r, xi, omega, rho, solution.get("obj", np.nan)


def init_complete_solution():
  return {
    "local_processing": pd.DataFrame(),
    "offloading": pd.DataFrame(),
    "detailed_offloading": pd.DataFrame(),
    "rejections": pd.DataFrame(),
    "replicas": pd.DataFrame(),
    "utilization": pd.DataFrame(),
    "offloaded_processing": pd.DataFrame(),
    "detailed_offloaded_processing": pd.DataFrame(),
    "residual_capacity": pd.DataFrame()
  }


def init_empty_solution(Nn: int, Nf: int) -> Tuple:
  x = np.zeros((Nn,Nf))
  y = np.zeros((Nn,Nn,Nf))
  z = np.zeros((Nn,Nf))
  r = np.zeros((Nn,Nf))
  xi = np.zeros((Nn,Nn,Nf))
  omega = np.zeros((Nn,Nf))
  rho = np.zeros((Nn,))
  obj = None
  U = np.zeros((Nn,Nf))
  return x, y, z, r, xi, omega, rho, obj, U


def init_problem(
    limits: dict,
    trace_type: str,
    max_steps: int,
    seed: int,
    solution_folder: str
  ) -> Tuple[dict, dict, list, Graph]:
  if limits.get("instance_type") == "materialized":
    instance_path = limits["path"]
    problem = load_materialized_instance(instance_path)
    for filename in (*PAYLOAD_FILES, "metadata.json"):
      shutil.copy2(
        os.path.join(instance_path, filename),
        os.path.join(solution_folder, filename),
      )
    return problem
  # generate base instance data
  rng = np.random.default_rng(seed = seed)
  base_instance_data, load_limits, graph = generate_data(
    limits.get("instance_type", "random"), rng = rng, limits = limits
  )
  with open(
    os.path.join(solution_folder, "base_instance_data.json"), "w"
  ) as istream:
    istream.write(
      json.dumps(
        delete_tuples(base_instance_data), indent = 2, cls = NpEncoder
      )
    )
  with open(
    os.path.join(solution_folder, "load_limits.json"), "w"
  ) as istream:
    istream.write(json.dumps(load_limits, indent = 2, cls = NpEncoder))
  # generate input load traces
  input_requests_traces = generate_load_traces(
    load_limits, max_steps, seed, trace_type, solution_folder
  )
  # draw graph
  os.makedirs(os.path.join(solution_folder, "graph"), exist_ok = True)
  planar = is_planar(graph)
  if planar:
    with open(os.path.join(solution_folder, "graph/PLANAR"), "wb"):
      pass
  draw_networkx(
    graph,
    pos = kamada_kawai_layout(graph, weight = "network_latency"),
    node_color = mcolors.CSS4_COLORS["lightskyblue"],
    with_labels = False,
    node_size = 30,
    width = 0.1
  )
  plt.savefig(
    os.path.join(solution_folder, "graph", "graph.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()
  return base_instance_data,input_requests_traces,load_limits[0].keys(),graph


def join_complete_solution(
    complete_solution: dict
  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  solution = complete_solution["local_processing"].join(
    complete_solution["offloading"], lsuffix = "_loc", rsuffix = "_fwd"
  ).join(complete_solution["rejections"])
  offloaded = complete_solution["offloaded_processing"]
  detailed_fwd_solution = complete_solution["detailed_offloading"].join(
    complete_solution["detailed_offloaded_processing"],
    lsuffix = "_tot",
    rsuffix = "_accepted"
  )
  return solution, offloaded, detailed_fwd_solution


def solver_threads(solver_name: str, solver_options: dict) -> int:
  """Threads of the centralized solve: an explicit positive setting, 1 for a
  single-threaded backend, 0 when automatic or unknown."""
  for key in ("Threads", "threads"):
    value = solver_options.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
      return value
  return 1 if solver_name in {"glpk", "glpsol"} else 0


def solve_instance(
    M: BaseLoadManagementModel,
    data: dict,
    solver_name: str,
    solver_options: dict
  ) -> Tuple[
    np.array,
    np.array,
    np.array,
    np.array,
    np.array,
    np.array,
    np.array,
    float,
    float,
    str
  ]:
  instance = M.generate_instance(data)
  # instance.pprint()
  #
  solution = M.solve(instance, solver_options, solver_name)
  tc = solution["termination_condition"]
  x, y, z, r, xi, omega, rho, obj, U = init_empty_solution(
    data[None]["Nn"][None], data[None]["Nf"][None]
  )
  if solution["solution_exists"]:
    x, y, z, r, xi, omega, rho, obj = extract_solution(
      data, solution, approx_tol = solver_options.get("FeasibilityTol", 1e-6)
    )
    U = compute_utilization(data, solution)
  return x, y, z, r, xi, omega, rho, U, obj, solution["runtime"], tc


def save_checkpoint(complete_solution: dict, solution_folder: str, t: int):
  checkpoint_folder = os.path.join(solution_folder, str(t))
  os.makedirs(checkpoint_folder, exist_ok = True)
  for key, val in complete_solution.items():
    checkpoint_path = os.path.join(checkpoint_folder, f"{key}.csv")
    val.to_csv(checkpoint_path, index = False)


def load_checkpoint(solution_folder: str, t: int):
  checkpoint_folder = os.path.join(solution_folder, str(t))
  complete_solution = {}
  for filename in os.listdir(checkpoint_folder):
    if filename.endswith(".csv"):
      df = pd.read_csv(os.path.join(checkpoint_folder, filename))
      complete_solution[filename.split(".")[0]] = df
  return complete_solution


def save_solution(
    solution: pd.DataFrame,
    offloaded: pd.DataFrame,
    complete_solution: dict,
    detailed_fwd_solution: pd.DataFrame,
    model_name: str,
    solution_folder: str
  ):
  solution.to_csv(
    os.path.join(solution_folder, f"{model_name}_solution.csv"),
    index = False
  )
  offloaded.to_csv(
    os.path.join(solution_folder, f"{model_name}_offloaded.csv"),
    index = False
  )
  complete_solution["utilization"].to_csv(
    os.path.join(solution_folder, f"{model_name}_utilization.csv"),
    index = False
  )
  complete_solution["replicas"].to_csv(
    os.path.join(solution_folder, f"{model_name}_replicas.csv"),
    index = False
  )
  detailed_fwd_solution.to_csv(
    os.path.join(solution_folder, f"{model_name}_detailed_fwd_solution.csv"),
    index = False
  )
  complete_solution["residual_capacity"].to_csv(
    os.path.join(solution_folder, f"{model_name}_residual_capacity.csv"),
    index = False
  )
