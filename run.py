"""Run one paper method on a materialized instance; outputs never overwrite a run."""
import argparse
import json
from pathlib import Path
import time


def main(argv=None):
  # Keep help usable before scientific dependencies have been installed.
  parser = argparse.ArgumentParser(description=__doc__)
  names = ["FaaS-MASeA", "FaaS-MASeA(r)", "PG", "Greedy-RR", "centralized", "selfish-centralized"]
  parser.add_argument("--list", action="store_true")
  parser.add_argument("--algorithm", choices=names)
  parser.add_argument("--instance", type=Path)
  parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/paper.json")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--parallelism", type=int, default=0, help="0: serial local solves; positive: worker count")
  parser.add_argument("--time-limit", type=float, default=14400, help="centralized wall-clock limit in seconds")
  parser.add_argument("--gap", type=float, default=1e-6, help="centralized relative MIP gap")
  args = parser.parse_args(argv)
  if args.list:
    print(json.dumps(names))
    return 0
  if not args.algorithm or args.instance is None or args.output is None:
    parser.error("--algorithm, --instance and --output are required")
  if args.parallelism < 0 or not 0 < args.time_limit < float('inf') or not 0 <= args.gap < float('inf'):
    parser.error("parallelism and gap must be nonnegative; time-limit must be positive and finite")
  from remote_experiments.instances import validate_instance
  validate_instance(args.instance)
  args.output.mkdir(parents=True, exist_ok=False)
  import methods
  started = time.perf_counter()
  result = {"algorithm": args.algorithm, "instance": args.instance.name}
  try:
    if args.algorithm in methods.CENTRALIZED:
      from catalog_centralized import solve
      row = solve(args.instance, args.time_limit, threads=max(1, args.parallelism), gap=args.gap,
                  log_file=str(args.output / "solver.log"), **methods.CENTRALIZED[args.algorithm])
      result.update(status=row["status"], welfare=[row["welfare"]], details=row)
    else:
      import pandas as pd
      config = json.loads(args.config.read_text())
      config["limits"] = {"instance_type": "materialized", "path": str(args.instance.resolve()), "load": {}}
      config["base_solution_folder"] = str((args.output / "run").resolve())
      runner = {**methods.HARMONIZED, **methods.SEQUENTIAL_METHODS}[args.algorithm]
      folder = Path(runner(config, args.parallelism, log_on_file=True, disable_plotting=True))
      result.update(status="ok", welfare=pd.read_csv(folder / "obj.csv").iloc[:, -1].tolist(),
                    algorithm_s=pd.read_csv(folder / "runtime.csv")["algorithm_s"].tolist())
      if config.get("communication_metrics"):
        result["communication"] = json.loads((folder / "communication.json").read_text())
    result["wall_s"] = time.perf_counter() - started
  except Exception as exc:
    result.update(status="error", error=f"{type(exc).__name__}: {exc}", wall_s=time.perf_counter() - started)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    raise
  (args.output / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
  print(json.dumps(result, default=str))
  return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
  raise SystemExit(main())
