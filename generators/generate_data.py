from utils.common import generate_random_float, generate_random_int
from utils.common import load_base_instance

from copy import deepcopy
from itertools import combinations
from typing import Tuple
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay


def add_network_latency(
    graph: nx.Graph, limits: dict, rng: np.random.Generator
  ) -> nx.Graph:
  weights = limits.get("weights", {})
  latency_limits = weights.get("edge_network_latency")
  edges = list(graph.edges())
  if latency_limits and latency_limits.get("mode") in {
      "euclidean", "euclidean_permuted"
    }:
    jitter_limits = latency_limits.get("jitter", {"min": 0.0, "max": 0.0})
    latencies = [
      latency_limits.get("base", 0.0)
      + latency_limits.get("distance_factor", 1.0)
      * graph.edges[edge]["edge_length"]
      + generate_random_float(rng, jitter_limits)
      for edge in edges
    ]
    if latency_limits["mode"] == "euclidean_permuted":
      latencies = rng.permutation(latencies)
    for edge, latency in zip(edges, latencies):
      graph.edges[edge]["network_latency"] = float(latency)
  elif latency_limits:
    for edge in edges:
      graph.edges[edge]["network_latency"] = generate_random_float(
        rng, latency_limits
      )
  else:
    nx.set_edge_attributes(graph, 1.0, "network_latency")
  if "edge_bandwidth" in weights:
    for edge in edges:
      graph.edges[edge]["edge_bandwidth"] = generate_random_int(
        rng, weights["edge_bandwidth"]
      )
  return graph


def from_existing_instance(limits: dict, rng: np.random.Generator) -> dict:
  # load data
  base_instance_data, load_limits = load_base_instance(limits["path"])
  # number of nodes and function classes (cannot be changed!)
  Nn = base_instance_data[None]["Nn"][None]
  Nf = base_instance_data[None]["Nf"][None]
  # neighborhood
  graph = None
  if "neighborhood" in limits:
    neighborhood, graph = generate_neighborhood(Nn, limits, rng)
    base_instance_data[None]["neighborhood"] = {
      (i+1, j+1): int(neighborhood[i,j]) for i in range(Nn) for j in range(Nn)
    }
  else:
    neighborhood = np.zeros((Nn,Nn))
    for (n1, n2), p in base_instance_data[None]["neighborhood"].items():
      neighborhood[n1-1,n2-1] = p
    graph = add_network_latency(nx.from_numpy_array(neighborhood), limits, rng)
  # weights
  if "weights" in limits:
    alpha, beta, gamma, delta = generate_weights(Nn, Nf, limits, rng, graph)
    base_instance_data[None]["alpha"] = {
      (n+1, f+1): float(alpha[f]) for n in range(Nn) for f in range(Nf)
      # (n+1, f+1): float(alpha[n][f]) for n in range(Nn) for f in range(Nf)
    }
    base_instance_data[None]["beta"] = {
      (n1+1, n2+1, f+1): float(beta[n1,n2,f]) \
        for n1 in range(Nn) \
          for n2 in range(Nn) \
            for f in range(Nf)
    }
    base_instance_data[None]["gamma"] = {
      (n+1, f+1): float(gamma[n,f]) for n in range(Nn) for f in range(Nf)
    }
    base_instance_data[None]["delta"] = {
      (n+1, f+1): float(delta[n,f]) for n in range(Nn) for f in range(Nf)
    }
  # demand
  if "demand" in limits:
    demand = generate_demand(Nn, Nf, limits, rng)
    demand_type = limits["demand"].get("type", "homogeneous")
    base_instance_data[None]["demand"] = {
      (n+1, f+1): float(
        demand[f]
      ) if demand_type == "homogeneous" else demand[
        n,f
      ] for n in range(Nn) for f in range(Nf)
    }
  # memory requirement
  if "memory_requirement" in limits:
    memory_requirement = generate_memory_requirement(Nf, limits, rng)
    base_instance_data[None]["memory_requirement"] = {
      f+1: memory_requirement[f] for f in range(Nf)
    }
  # memory capacity
  if "memory_capacity" in limits:
    memory_capacity, speedup_factors = generate_memory_capacity(
      Nn, limits, rng
    )
    base_instance_data[None]["memory_capacity"] = {}
    for n in range(Nn):
      base_instance_data[None]["memory_capacity"][n+1] = memory_capacity[n]
      # -- correct demand according to the speedup factor
      for f in range(Nf):
        base_instance_data[None]["demand"][(n+1,f+1)] /= speedup_factors[n]
  # load limits
  if "load" in limits and (
      limits["load"]["trace_type"] == "load_existing" or
      limits["load"]["trace_type"] == "load_filtered"
    ):
    load_limits[0] = {n: None for n in range(Nn)}
    load_limits["load_existing"] = limits["load"]["path"]
  return base_instance_data, load_limits, graph


def generate_data(
    scenario: str,
    rng: np.random.Generator = None,
    limits: dict = None
  ) -> Tuple[dict, dict]:
  data = {}
  if scenario == "random":
    data = random_instance_data(limits, rng)
  elif scenario == "load_existing":
    data = from_existing_instance(limits, rng)
  else:
    raise KeyError(f"Undefined scenario: {scenario}")
  return data


def generate_demand(
    Nn: int, Nf: int, limits: dict, rng: np.random.Generator
  ) -> np.array:
  demand = []
  if "values" in limits["demand"] and len(limits["demand"]["values"]) == Nf:
    demand = np.array(limits["demand"]["values"])
  elif limits["demand"].get("type", "homogeneous") == "homogeneous":
    demand = np.array([
      generate_random_float(rng, limits["demand"]) for _ in range(Nf)
    ])
  else:
    demand = np.array([
      generate_random_float(
        rng, limits["demand"]
      ) for _ in range(Nn) for _ in range(Nf)
    ]).reshape((Nn,Nf))
  return demand


def generate_memory_capacity(
    Nn: int, limits: dict, rng: np.random.Generator
  ) -> Tuple[list, list]:
  memory_capacity = []
  speedup_factors = []
  if "repeated_values" in limits["memory_capacity"]:
    idx = 0
    set_nodes = 0
    for (perc, memory) in limits["memory_capacity"]["repeated_values"]:
      nnodes = int(perc * Nn)
      if idx == len(limits["memory_capacity"]["repeated_values"]) - 1:
        nnodes = max(nnodes, Nn - set_nodes)
      memory_capacity += ([memory] * nnodes)
      # -- check whether speedup factors are provided
      if "speedup_factors" in limits["demand"]:
        speedup_factors += (
          [limits["demand"]["speedup_factors"][str(memory)]] * nnodes
        )
      else:
        speedup_factors += ([1.0] * nnodes)
      idx += 1
      set_nodes += nnodes
  else:
    memory_capacity = [
      generate_random_int(
        rng, limits["memory_capacity"]
      ) if "values" not in limits["memory_capacity"] else limits[
        "memory_capacity"
      ]["values"][n] for n in range(Nn)
    ]
    speedup_factors = [1.0] * Nn
  return memory_capacity, speedup_factors


def generate_memory_requirement(
    Nf: int, limits: dict, rng: np.random.Generator
  ) -> list:
  memory_requirement = [
      generate_random_int(
      rng, limits["memory_requirement"]
    ) if "values" not in limits["memory_requirement"] else limits[
      "memory_requirement"
    ]["values"][f] for f in range(Nf)
  ]
  return memory_requirement


def generate_neighborhood(
    Nn: int, limits: dict, rng: np.random.Generator
  ) -> Tuple[np.array, nx.Graph]:
  neighborhood = np.zeros((Nn, Nn))
  graph = None
  neighborhood_limits = limits["neighborhood"]
  shape = neighborhood_limits.get("type", neighborhood_limits.get("shape"))
  if shape == "euclidean_planar":
    mean_degree = neighborhood_limits.get(
      "mean_degree",
      neighborhood_limits.get("degree", neighborhood_limits.get("k")),
    )
    density = neighborhood_limits.get("density", 1.0)
    if Nn < 3 or density <= 0 or mean_degree is None:
      raise ValueError(
        "connected Euclidean planar neighborhood requires Nn >= 3, "
        "positive density, and mean_degree"
      )
    side = np.sqrt(Nn / density)
    points = rng.uniform(0, side, size=(Nn, 2))
    candidate = nx.Graph()
    candidate.add_nodes_from(
      (node, {"pos": tuple(point)}) for node, point in enumerate(points)
    )
    for simplex in Delaunay(points).simplices:
      candidate.add_edges_from(combinations(map(int, simplex), 2))
    for u, v in candidate.edges():
      candidate.edges[u, v]["edge_length"] = float(
        np.linalg.norm(points[u] - points[v])
      )
    target_edges = round(Nn * mean_degree / 2)
    if not Nn - 1 <= target_edges <= candidate.number_of_edges():
      raise ValueError(
        "connected Euclidean planar neighborhood has an infeasible edge budget"
      )
    graph = nx.minimum_spanning_tree(candidate, weight="edge_length")
    remaining = list(set(candidate.edges()) - set(graph.edges()))
    for index in rng.permutation(len(remaining))[:target_edges - (Nn - 1)]:
      u, v = remaining[index]
      graph.add_edge(u, v, **candidate.edges[u, v])
    neighborhood = nx.to_numpy_array(graph, dtype=int)
  elif shape == "planar":
    degree = neighborhood_limits.get("degree", neighborhood_limits.get("k", 3))
    if degree != 3 or Nn < 4 or Nn % 2:
      raise ValueError("planar regular neighborhoods require degree=3 and even Nn >= 4")
    # Replacing a cubic vertex with a triangle preserves planarity and degree.
    # ponytail: samples the vertex-expansion family, not uniformly all cubic
    # planar graphs; use a dedicated sampler if unbiased graph sampling is needed.
    graph = nx.complete_graph(4)
    while graph.number_of_nodes() < Nn:
      node = int(rng.choice(list(graph.nodes())))
      neighbors = list(graph.neighbors(node))
      new_nodes = [node, graph.number_of_nodes(), graph.number_of_nodes() + 1]
      graph.remove_node(node)
      graph.add_edges_from(combinations(new_nodes, 2))
      graph.add_edges_from(zip(neighbors, rng.permutation(new_nodes)))
    graph = nx.relabel_nodes(graph, dict(zip(range(Nn), rng.permutation(Nn))))
    ordered_graph = nx.Graph()
    ordered_graph.add_nodes_from(range(Nn))
    ordered_graph.add_edges_from(graph.edges())
    graph = ordered_graph
    neighborhood = nx.to_numpy_array(graph, nodelist=range(Nn), dtype=int)
  elif "p" in limits["neighborhood"]:
    for _ in range(1000):
      neighborhood = np.zeros((Nn, Nn))
      for n1 in range(Nn):
        for n2 in range(n1+1,Nn):
          neighborhood[n1,n2] = rng.binomial(1, limits["neighborhood"]["p"])
          neighborhood[n2,n1] = neighborhood[n1,n2]
      graph = nx.from_numpy_array(neighborhood)
      if nx.is_connected(graph):
        break
    else:
      raise ValueError(
        "could not generate a connected random neighborhood in 1000 attempts"
      )
  elif "m" in limits["neighborhood"]:
    for _ in range(1000):
      graph = nx.gnm_random_graph(
        Nn,
        limits["neighborhood"]["m"],
        seed=int(rng.integers(low=0, high=4850 * 4850 * 4850)),
      )
      if nx.is_connected(graph):
        break
    else:
      raise ValueError(
        "could not generate a connected fixed-edge neighborhood in 1000 attempts"
      )
    neighborhood = nx.adjacency_matrix(graph).toarray()
  elif "k" in limits["neighborhood"]:
    for _ in range(1000):
      graph = nx.random_regular_graph(
        d=limits["neighborhood"]["k"],
        n=Nn,
        seed=int(rng.integers(low=0, high=4850 * 4850 * 4850)),
      )
      if nx.is_connected(graph):
        break
    else:
      raise ValueError(
        "could not generate a connected regular neighborhood in 1000 attempts"
      )
    neighborhood = nx.adjacency_matrix(graph).toarray()
  if graph is None:
    raise ValueError(f"unsupported neighborhood configuration: {neighborhood_limits}")
  # -- add network latency (if available)
  graph = add_network_latency(graph, limits, rng)
  return neighborhood, graph


def offload_gain_mean(beta: np.ndarray) -> np.ndarray:
  """Zero self-forwarding rewards in-place, then average over all N nodes.

  Call again after any transformation of beta, before serializing an instance.
  This is the delta convention used by the paper's 25.11.22 generator.
  """
  nodes = np.arange(beta.shape[0])
  beta[nodes, nodes, :] = 0.
  return beta.mean(axis=1)


def generate_weights(
    Nn: int, Nf: int, limits: dict, rng: np.random.Generator, graph: nx.Graph
  ) -> Tuple[list, np.array, np.array, np.array]:
  # weights (different for each function, equal for all nodes)
  alpha, beta, gamma, delta = [None] * 4
  weights_generator = generate_random_float if limits["weights"].get(
    "dtype", "float"
  ) == "float" else generate_random_int
  if "initialization_time" not in limits["weights"]:
    alpha = [
      weights_generator(rng, limits["weights"]["alpha"]) for _ in range(Nf)
    ]
    beta = np.zeros((Nn,Nn,Nf))
    gamma = np.zeros((Nn,Nf))
    delta = np.zeros((Nn,Nf))
    if limits["weights"].get("type", "homogeneous") == "homogeneous":
      b = None
      if "beta_multiplier" in limits["weights"]:
        b = [
          alpha[f] * generate_random_float(
            rng, limits["weights"]["beta_multiplier"]
          ) for f in range(Nf)
        ]
      else:
        b = [
          weights_generator(
            rng, limits["weights"]["beta"]
          ) for _ in range(Nf)
        ]
      g = [
        weights_generator(
          rng, limits["weights"]["gamma"]
        ) for _ in range(Nf)
      ]
      d = [
        b[f] * generate_random_float(
          rng, limits["weights"]["delta_multiplier"]
        ) for f in range(Nf)
      ]
      for n1 in range(Nn - 1):
        gamma[n1,:] = g
        delta[n1,:] = d
        for n2 in range(n1, Nn):
          beta[n1,n2,:] = b
          beta[n2,n1,:] = b
      gamma[Nn-1,:] = g
      delta[Nn-1,:] = d
    else:
      for n1 in range(Nn - 1):
        g = [
          weights_generator(
            rng, limits["weights"]["gamma"]
          ) for _ in range(Nf)
        ]
        gamma[n1,:] = g
        for n2 in range(n1, Nn):
          b = [
            alpha[f] * generate_random_float(
              rng, limits["weights"]["beta_multiplier"]
            ) for f in range(Nf)
          ]
          beta[n1,n2,:] = b
          beta[n2,n1,:] = b
        d = [
          beta[n1,:,f].mean() * generate_random_float(
            rng, limits["weights"]["delta_multiplier"]
          ) for f in range(Nf)
        ]
        delta[n1,:] = d
      gamma[Nn-1,:] = [
        weights_generator(rng, limits["weights"]["gamma"]) for _ in range(Nf)
      ]
      delta[Nn-1,:] = [
        beta[Nn-1,:,f].mean() * generate_random_float(
          rng, limits["weights"]["delta_multiplier"]
        ) for f in range(Nf)
      ]
  else:
    # -- local execution
    alpha = [
      generate_random_float(
        rng, limits["weights"]["initialization_time"]
      ) for _ in range(Nf)
    ]
    min_price = min(alpha)
    max_price = min_price
    # -- network transfer time
    data_size = [
      generate_random_float(
        rng, limits["weights"]["input_data"]
      ) for _ in range(Nf)
    ]
    cloud_bandwidth = generate_random_int(
      rng, limits["weights"]["cloud_bandwidth"]
    )
    beta = np.zeros((Nn,Nn,Nf))
    gamma = np.zeros((Nn,Nf))
    for n1 in range(Nn - 1):
      for f in range(Nf):
        gamma[n1,f] = generate_random_float(
          rng, limits["weights"]["cloud_network_latency"]
        ) + (
          data_size[f] / cloud_bandwidth
        )
        for n2 in range(n1 + 1, Nn):
          # -- if the edge exists, compute the price based on network latency
          if graph.has_edge(n1,n2):
            beta[n1,n2,f] = alpha[f] + graph.edges[n1,n2]["network_latency"] + (
              data_size[f] / graph.edges[n1,n2]["edge_bandwidth"]
            )
          else:
            # -- otherwise, assign -1
            beta[n1,n2,f] = -1
          beta[n2,n1,f] = beta[n1,n2,f]
          max_price = max(max_price, beta[n2,n1,f])
    for f in range(Nf):
      gamma[Nn - 1,f] = generate_random_float(
        rng, limits["weights"]["cloud_network_latency"]
      ) + (
        data_size[f] / cloud_bandwidth
      )
    min_g = gamma.min()
    max_g = gamma.max()
    gamma_range = max_g - min_g
    price_range = max_price - min_price
    # -- normalize
    alpha = [
      0.0 if price_range == 0 else 1 - ((a - min_price) / price_range)
      for a in alpha
    ]
    for n1 in range(Nn - 1):
      for f in range(Nf):
        gamma[n1,f] = (
          0.0 if gamma_range == 0 else (gamma[n1,f] - min_g) / gamma_range
        )
        for n2 in range(n1 + 1, Nn):
          if beta[n1,n2,f] > 0:
            beta[n1,n2,f] = (
              0.0 if price_range == 0
              else 1 - ((beta[n1,n2,f] - min_price) / price_range)
            )
          else:
            beta[n1,n2,f] = 0
          beta[n2,n1,f] = beta[n1,n2,f]
    for f in range(Nf):
      gamma[Nn - 1,f] = (
        0.0 if gamma_range == 0 else (gamma[Nn - 1,f] - min_g) / gamma_range
      )
  # Keep the historical delta_multiplier draws above to preserve the RNG stream:
  # changing it would also change subsequent beta/gamma and resource samples.
  # The final delta always derives from the final beta, without that multiplier.
  delta = offload_gain_mean(beta)
  return alpha, beta, gamma, delta


def update_data(data: dict, fixed_values: dict) -> dict:
  updated_data = deepcopy(data)
  for k, v in fixed_values.items():
    updated_data[None][k] = v
  return updated_data


def random_instance_data(
    limits: dict, rng: np.random.Generator
  ) -> Tuple[dict, dict, nx.Graph]:
  # number of nodes and function classes
  Nn = rng.integers(limits["Nn"]["min"], limits["Nn"]["max"], endpoint = True)
  Nf = rng.integers(limits["Nf"]["min"], limits["Nf"]["max"], endpoint = True)
  # neighborhood
  neighborhood, graph = generate_neighborhood(Nn, limits, rng)
  # weights (different for each function, equal for all nodes)
  alpha, beta, gamma, delta = generate_weights(Nn, Nf, limits, rng, graph)
  # demand
  demand = generate_demand(Nn, Nf, limits, rng)
  # data
  demand_type = limits["demand"].get("type", "homogeneous")
  # memory requirement
  memory_requirement = generate_memory_requirement(Nf, limits, rng)
  # memory capacity
  memory_capacity, speedup_factors = generate_memory_capacity(Nn, limits, rng)
  # build dictionary
  data = {None: {
    "Nn": {None: int(Nn)},
    "Nf": {None: int(Nf)},
    "demand": {
      (n+1, f+1): float(
        demand[f] / speedup_factors[n]
      ) if demand_type == "homogeneous" else demand[
        n,f
      ] / speedup_factors[n] for n in range(Nn) for f in range(Nf)
    },
    "memory_requirement": {
      f+1: memory_requirement[f] for f in range(Nf)
    },
    "memory_capacity": {
      n+1: memory_capacity[n] for n in range(Nn)
    },
    "neighborhood": {
      (i+1, j+1): int(neighborhood[i,j]) for i in range(Nn) for j in range(Nn)
    },
    "max_utilization": {
      f+1: generate_random_float(
        rng, limits["max_utilization"]
      ) for f in range(Nf)
    },
    "alpha": {
      (n+1, f+1): float(alpha[f]) for n in range(Nn) for f in range(Nf)
      # (n+1, f+1): float(alpha[n][f]) for n in range(Nn) for f in range(Nf)
    },
    "beta": {
      (n1+1, n2+1, f+1): float(beta[n1,n2,f]) \
        for n1 in range(Nn) \
          for n2 in range(Nn) \
            for f in range(Nf)
    },
    "gamma": {
      (n+1, f+1): float(gamma[n,f]) for n in range(Nn) for f in range(Nf)
    },
    "delta": {
      (n+1, f+1): float(delta[n,f]) for n in range(Nn) for f in range(Nf)
    }
  }}
  # load limits
  load_limits = {}
  if (
      limits["load"]["trace_type"] == "load_existing" or
      limits["load"]["trace_type"] == "load_filtered"
    ):
    load_limits[0] = {n: None for n in range(Nn)}
    load_limits["load_existing"] = limits["load"]["path"]
  elif "values" in limits["load"]:
    if len(limits["load"]["values"]) == Nf and (
        limits["load"]["values"][0] != "auto"
      ):
      load_limits = {
        f: {
          n: limits["load"]["values"][f] for n in range(Nn)
        } for f in range(Nf)
      }
    elif limits["load"]["values"][0] == "auto":
      load_limits = {
        f: {
          n: round((
            (
              data[None]["memory_capacity"][n+1] *
                data[None]["max_utilization"][f+1]
            ) / (
              data[None]["memory_requirement"][f+1] *
                data[None]["demand"][(n+1,f+1)]
            ) * data[None]["Nn"][None] / data[None]["Nf"][None]
          ), 3) - 1 for n in range(Nn)
        } for f in range(Nf)
      }
      # lmax = []
      # for f in range(Nf):
      #   l = 0
      #   for n in range(Nn):
      #     l += (
      #       data[None]["memory_capacity"][n+1] *
      #         data[None]["max_utilization"][f+1] / (
      #           data[None]["memory_requirement"][f+1] *
      #             data[None]["demand"][(n+1,f+1)]
      #         )
      #     )
      #     lmax.append(l)
      # load_limits = {
      #   f: {
      #     n: round(lmax[f], 3) * Nn / Nf + 1 for n in range(Nn)
      #   } for f in range(Nf)
      # }
  else:
    load_limits = {
      f: {
        n: {
          "min": generate_random_float(rng, limits["load"]["min"]),
          "max": generate_random_float(rng, limits["load"]["max"])
        } for n in range(Nn)
      } for f in range(Nf)
    }
  return data, load_limits, graph
