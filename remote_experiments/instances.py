"""Materialize complete, checksummed inputs shared by remote experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx
import numpy as np

from generators.generate_data import generate_data
from generators.generate_load import generate_load_traces
from utils.common import (
  NpEncoder,
  delete_tuples,
  load_base_instance,
  load_requests_traces,
)

from .batch import Batch, Experiment

PAYLOAD_FILES = (
  "base_instance_data.json",
  "load_limits.json",
  "input_requests_traces.json",
  "graph.json",
)


def generation_spec(experiment: Experiment) -> dict:
  config = experiment.config
  spec = {
    "generation_seed": config.get("instance_seed", experiment.seed),
    "limits": config["limits"],
    "max_steps": config["max_steps"],
    "min_run_time": config.get("min_run_time", 0),
    "max_run_time": config.get("max_run_time", config["max_steps"]),
    "run_time_step": config.get("run_time_step", 1),
  }
  if (config["limits"].get("instance_type", "random") == "random"
      or "weights" in config["limits"]):
    spec["weight_generation"] = "paper-delta-mean-v1"
  return spec


def instance_id(experiment: Experiment) -> str:
  encoded = json.dumps(
    generation_spec(experiment), sort_keys=True, separators=(",", ":"),
  ).encode()
  return f"instance-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def validate_instance(path: str | Path) -> dict:
  instance_path = Path(path)
  metadata_path = instance_path / "metadata.json"
  if not metadata_path.is_file():
    raise ValueError(f"missing instance metadata: {metadata_path}")
  metadata = json.loads(metadata_path.read_text())
  for filename in PAYLOAD_FILES:
    payload_path = instance_path / filename
    if not payload_path.is_file():
      raise ValueError(f"missing instance file: {payload_path}")
    expected = metadata.get("files", {}).get(filename)
    actual = _sha256(payload_path)
    if expected != actual:
      raise ValueError(
        f"checksum mismatch for {filename}: expected {expected}, got {actual}"
      )
  return metadata


def materialize_instance(experiment: Experiment, path: str | Path) -> Path:
  instance_path = Path(path)
  if instance_path.exists():
    metadata = validate_instance(instance_path)
    if metadata.get("generation") != generation_spec(experiment):
      raise ValueError(f"generation specification mismatch for {instance_path}")
    return instance_path

  instance_path.parent.mkdir(parents=True, exist_ok=True)
  spec = generation_spec(experiment)
  limits = spec["limits"]
  rng = np.random.default_rng(seed=spec["generation_seed"])
  base_data, load_limits, graph = generate_data(
    limits.get("instance_type", "random"), rng=rng, limits=limits,
  )
  traces = generate_load_traces(
    load_limits,
    spec["max_steps"],
    spec["generation_seed"],
    limits["load"].get("trace_type", "fixed_sum"),
    enable_plotting=False,
  )

  # Publish only a complete directory; interruption leaves no final instance,
  # and an existing nonempty directory cannot be overwritten by rename.
  with TemporaryDirectory(prefix=f".{instance_path.name}-", dir=instance_path.parent) as temporary:
    draft = Path(temporary) / "instance"
    draft.mkdir()
    (draft / "base_instance_data.json").write_text(json.dumps(
      delete_tuples(base_data), indent=2, cls=NpEncoder,
    ))
    (draft / "load_limits.json").write_text(json.dumps(
      load_limits, indent=2, cls=NpEncoder,
    ))
    (draft / "input_requests_traces.json").write_text(json.dumps(
      traces, indent=2, cls=NpEncoder,
    ))
    (draft / "graph.json").write_text(json.dumps(
      nx.node_link_data(graph, edges="edges"), indent=2, cls=NpEncoder,
    ))
    metadata = {
      "schema_version": 1,
      "instance_id": instance_id(experiment),
      "suite": experiment.suite,
      "generation": spec,
      "files": {
        filename: _sha256(draft / filename) for filename in PAYLOAD_FILES
      },
    }
    (draft / "metadata.json").write_text(json.dumps(metadata, indent=2))
    draft.rename(instance_path)
  return instance_path


def load_materialized_instance(path: str | Path) -> tuple[dict, dict, object, nx.Graph]:
  instance_path = Path(path)
  validate_instance(instance_path)
  base_data, load_limits = load_base_instance(str(instance_path))
  traces = load_requests_traces(str(instance_path))[0]
  graph = nx.node_link_graph(
    json.loads((instance_path / "graph.json").read_text()), edges="edges",
  )
  return base_data, traces, load_limits[0].keys(), graph


def materialize_batch(batch: Batch, root: str | Path) -> Path:
  suite_path = Path(root) / batch.suite
  data_path = suite_path / "data"
  data_path.mkdir(parents=True, exist_ok=True)
  experiment_instances = {}
  for experiment in batch.experiments:
    identifier = instance_id(experiment)
    materialize_instance(experiment, data_path / identifier)
    experiment_instances[experiment.id] = identifier

  identifiers = sorted(set(experiment_instances.values()))
  manifest = {
    "schema_version": 1,
    "suite": batch.suite,
    "instances": identifiers,
    "experiments": experiment_instances,
  }
  (suite_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
  (suite_path / "README.md").write_text(
    f"# {batch.suite} materialized instances\n\n"
    f"This suite contains {len(identifiers)} immutable instances used by "
    f"{len(batch.experiments)} experiments. Each directory under `data/` contains "
    "the optimization data, load limits, complete temporal request traces, exact "
    "graph, and SHA-256 metadata. Algorithm seeds and solver options remain in "
    "the experiment batch.\n"
  )
  return suite_path
