from typing import Tuple
import pandas as pd
import numpy as np


def count_offloaded_processing(
    detailed_offloading: pd.DataFrame, Nn: int, Nf: int
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  offloaded_processing = pd.DataFrame()
  detailed_offloaded_processing = pd.DataFrame()
  for n in range(Nn):
    for f in range(Nf):
      offloaded_processing[f"n{n}_f{f}_accepted"] = detailed_offloading.loc[
        :,detailed_offloading.columns.str.endswith(f"_f{f}_n{n}")
      ].sum(axis = "columns")
  return offloaded_processing, detailed_offloaded_processing


def decode_solution(
    x: np.array,
    y: np.array,
    z: np.array,
    r: np.array,
    xi: np.array,
    rho: np.array,
    U: np.array,
    complete_solution: dict,
  ) -> dict:
  Nn, Nf = x.shape
  # local processing
  complete_solution["local_processing"] = update_2d_variables(
    x, complete_solution["local_processing"]
  )
  # offloading
  (
    complete_solution["offloading"], complete_solution["detailed_offloading"]
  ) = update_3d_variables(
    y, complete_solution["offloading"], complete_solution["detailed_offloading"]
  )
  # rejections
  complete_solution["rejections"] = update_2d_variables(
    z, complete_solution["rejections"]
  )
  # number of reserved instances
  complete_solution["replicas"] = update_2d_variables(
    r, complete_solution["replicas"]
  )
  # utilization
  complete_solution["utilization"] = update_2d_variables(
    U, complete_solution["utilization"]
  )
  # received offloading
  if xi is not None:
    (
      complete_solution["offloaded_processing"],
      complete_solution["detailed_offloaded_processing"]
    ) = update_3d_variables(
      xi,
      complete_solution["offloaded_processing"],
      complete_solution["detailed_offloaded_processing"]
    )
  else:
    (
      complete_solution["offloaded_processing"],
      complete_solution["detailed_offloaded_processing"]
    ) = count_offloaded_processing(
      complete_solution["detailed_offloading"], Nn, Nf
    )
  # residual capacity
  complete_solution["residual_capacity"] = update_1d_variables(
    rho, complete_solution["residual_capacity"]
  )
  return complete_solution


def encode_solution(
    Nn: int, Nf: int,
    solution: pd.DataFrame,
    detailed_fwd_solution: pd.DataFrame,
    replicas: pd.DataFrame,
    t: int
  ) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
  x = np.zeros((Nn,Nf))
  y = np.zeros((Nn,Nn,Nf))
  z = np.zeros((Nn,Nf))
  r = np.zeros((Nn,Nf))
  # check whether xi and zeta are part of the solution
  xi_exist = detailed_fwd_solution.columns.str.endswith("tot").any()
  xi = np.zeros((Nn,Nn,Nf)) if xi_exist else None
  for n in range(Nn):
    for f in range(Nf):
      basename = f"n{n}_f{f}"
      x[n,f] = solution.loc[t,f"{basename}_loc"]
      z[n,f] = solution.loc[t,basename]
      r[n,f] = replicas.loc[t,basename]
      for m in range(Nn):
        endname = "_tot" if xi_exist else ""
        if m != n:
          y[n,m,f] = detailed_fwd_solution.loc[t,f"{basename}_n{m}{endname}"]
          if xi_exist:
            xi[n,m,f] = detailed_fwd_solution.loc[t,f"{basename}_n{m}_accepted"]
  return x, y, z, r, xi


def ping_pong_forbidden_hosts(omega, y, tolerance = 1e-6) -> np.array:
  """(Nn, Nf) boolean mask of (node, function) pairs that must NOT host f.

  A node is forbidden as host (receiver) of f when it is already a *sender* of
  f: either it still has residual demand to offload (``omega``) or it has
  already forwarded some load for f (``y`` summed over receivers, axis 1).
  Excluding these nodes from the seller side each round is what keeps the
  decentralized methods that accumulate ``y`` inside the FRALB no_ping_pong
  constraint enforced by validate_centralized_solution. Mirrors the guards
  already inlined in decentralized_auction / _gcaa / _potentialgame.
  """
  omega = np.asarray(omega)
  y = np.asarray(y)
  return (omega > tolerance) | (y.sum(axis = 1) > tolerance)


def validate_centralized_solution(x, y, z, r, data, tolerance = 1e-6) -> None:
  try:
    tolerance_value = np.asarray(tolerance)
  except (TypeError, ValueError, OverflowError):
    raise ValueError(f"tolerance domain NonNegativeReal: {tolerance!r}")
  if tolerance_value.shape or tolerance_value.dtype.kind not in "buif":
    raise ValueError(f"tolerance domain NonNegativeReal: {tolerance!r}")
  tolerance = float(tolerance_value)
  if not np.isfinite(tolerance) or tolerance < 0:
    raise ValueError(f"tolerance domain NonNegativeReal: {tolerance!r}")

  values = data[None]
  Nn, Nf = values["Nn"][None], values["Nf"][None]
  arrays = {
    "x": np.asarray(x), "y": np.asarray(y),
    "z": np.asarray(z), "r": np.asarray(r),
  }
  expected_shapes = {
    "x": (Nn, Nf), "y": (Nn, Nn, Nf),
    "z": (Nn, Nf), "r": (Nn, Nf),
  }
  for name, array in arrays.items():
    if array.shape != expected_shapes[name]:
      raise ValueError(
        f"{name} shape: {array.shape} != {expected_shapes[name]}"
      )

  for name, array in arrays.items():
    domain = "NonNegativeIntegers" if name == "r" else "NonNegativeReals"
    if array.dtype.kind not in "buif":
      converted = np.empty(array.shape)
      for index, value in np.ndenumerate(array):
        scalar = np.asarray(value)
        invalid_scalar = (
          bool(scalar.shape)
          or scalar.dtype.kind not in "buifc"
          or (scalar.dtype.kind == "c" and scalar.imag != 0)
        )
        if invalid_scalar:
          shown = ",".join(str(i + 1) for i in index)
          raise ValueError(f"{name} domain {domain} ({shown}): {value!r}")
        converted[index] = scalar.real
      array = arrays[name] = converted
    invalid = ~np.isfinite(array) | (array < -tolerance)
    if invalid.any():
      index = tuple(np.argwhere(invalid)[0])
      shown = ",".join(str(i + 1) for i in index)
      raise ValueError(
        f"{name} domain {domain} ({shown}): {array[index]}"
      )
  invalid = np.abs(arrays["r"] - np.rint(arrays["r"])) > tolerance
  if invalid.any():
    index = tuple(np.argwhere(invalid)[0])
    shown = ",".join(str(i + 1) for i in index)
    raise ValueError(
      f"r domain NonNegativeIntegers ({shown}): {arrays['r'][index]}"
    )

  x, y, z, r = (
    arrays[name].astype(float, copy = False) for name in ("x", "y", "z", "r")
  )
  incoming_load = np.array([
    [values["incoming_load"][(n + 1, f + 1)] for f in range(Nf)]
    for n in range(Nn)
  ])
  neighborhood = np.zeros((Nn, Nn))
  for (n, m), is_neighbor in values["neighborhood"].items():
    neighborhood[n - 1, m - 1] = is_neighbor
  invalid = y - incoming_load[:, None, :] * neighborhood[:, :, None] > tolerance
  if invalid.any():
    n, m, f = np.argwhere(invalid)[0]
    raise ValueError(f"offload_only_to_neighbors ({n + 1},{m + 1},{f + 1})")

  invalid = (y.sum(axis = 1) > tolerance) & (y.sum(axis = 0) > tolerance)
  if invalid.any():
    n, f = np.argwhere(invalid)[0]
    raise ValueError(f"no_ping_pong ({n + 1},{f + 1})")

  invalid = np.abs(x + y.sum(axis = 1) + z - incoming_load) > tolerance
  if invalid.any():
    n, f = np.argwhere(invalid)[0]
    raise ValueError(f"no_traffic_loss ({n + 1},{f + 1})")

  demand = np.array([
    [values["demand"][(n + 1, f + 1)] for f in range(Nf)]
    for n in range(Nn)
  ])
  max_utilization = np.array([
    values["max_utilization"][f + 1] for f in range(Nf)
  ])
  utilization = demand * (x + y.sum(axis = 0))
  invalid = utilization - r * max_utilization > tolerance
  if invalid.any():
    n, f = np.argwhere(invalid)[0]
    raise ValueError(f"utilization_equilibrium ({n + 1},{f + 1})")
  invalid = (r - 1) * max_utilization - utilization > tolerance
  if invalid.any():
    n, f = np.argwhere(invalid)[0]
    raise ValueError(f"utilization_equilibrium2 ({n + 1},{f + 1})")

  memory_requirement = np.array([
    values["memory_requirement"][f + 1] for f in range(Nf)
  ])
  memory_capacity = np.array([
    values["memory_capacity"][n + 1] for n in range(Nn)
  ])
  invalid = r @ memory_requirement - memory_capacity > tolerance
  if invalid.any():
    n = np.argwhere(invalid)[0, 0]
    raise ValueError(
      f"residual_capacity ({n + 1}: "
      f"{r[n,:] @ memory_requirement} > {memory_capacity[n]})"
    )


def check_feasibility(
      x: np.array,
      omega: np.array,
      z: np.array,
      r: np.array,
      cpu_utilization: np.array,
      data: dict
    ) -> Tuple[bool, str]:
    Nn, Nf = x.shape
    for n in range(1, Nn+1):
      for f in range(1, Nf+1):
        # no traffic loss
        managed_load = x[n-1,f-1] + omega[n-1,f-1] + z[n-1,f-1]
        load = data[None]["incoming_load"][(n,f)]
        if abs(managed_load - load) > 1e-3:
          return False, f"no traffic loss ({n},{f}): {managed_load} != {load}"
        # max utilization
        utilization = cpu_utilization[n-1,f-1]
        max_utilization = data[None]["max_utilization"][f]
        if utilization - max_utilization > 1e-5:
          return False, f"max utilization ({n},{f}): {utilization}"
    # memory capacity
    for n, ram in data[None]["memory_capacity"].items():
      used_memory = 0
      for f, req_memory in data[None]["memory_requirement"].items():
        used_memory += r[n-1,f-1] * req_memory
        if used_memory - ram > 1e-5:
          return False, f"memory capacity ({n},{f}): {used_memory} > {ram}"
    return True, ""


def get_current_load(
    input_requests_traces: dict, agents: list, t: int
  ) -> dict:
  incoming_load = {
    (a+1, f+1): input_requests_traces[f][a][t] \
      for a in agents for f in input_requests_traces
  }
  return incoming_load


def update_1d_variables(
    var: np.array, res: pd.DataFrame
  ) -> pd.DataFrame:
  Nn = var.shape[0]
  df = {f"n{n}": [var[n]] for n in range(Nn)}
  res = pd.concat(
    [res, pd.DataFrame(df)], ignore_index = True
  )
  return res


def update_2d_variables(
    var: np.array, res: pd.DataFrame
  ) -> pd.DataFrame:
  Nn, Nf = var.shape
  df = var.reshape(1,-1).tolist()
  cols = [f"n{n}_f{f}" for n in range(Nn) for f in range(Nf)]
  res = pd.concat(
    [res, pd.DataFrame(df, columns = cols)], ignore_index = True
  )
  return res


def update_3d_variables(
    y: np.array, offloading: pd.DataFrame, detailed_offloading: pd.DataFrame
  ) -> pd.DataFrame:
  Nn, _, Nf = y.shape
  df = {f"n{n}_f{f}": [] for n in range(Nn) for f in range(Nf)}
  detailed_df = {
    f"n{n1}_f{f}_n{n2}": [] \
      for n1 in range(Nn) for f in range(Nf) for n2 in range(Nn) if n2 != n1
  }
  for f in range(Nf):
    for n1 in range(Nn):
      df[f"n{n1}_f{f}"].append(y[n1,:,f].sum())
      for n2 in range(Nn):
        if n1 != n2:
          detailed_df[f"n{n1}_f{f}_n{n2}"].append(y[n1,n2,f])
  offloading = pd.concat(
    [offloading, pd.DataFrame(df)], ignore_index = True
  )
  detailed_offloading = pd.concat(
    [detailed_offloading, pd.DataFrame(detailed_df)], ignore_index = True
  )
  return offloading, detailed_offloading
