# FaaS-MASeA

Research artifact for **Decentralizing Edge-Cloud FaaS Orchestration: A Multi-Agent Auction Approach**.

This repository implements the six methods retained for the paper. They allocate
function replicas and route requests among edge nodes and the cloud. It is an
optimization/simulation artifact, not a deployed FaaS control plane.

| Method identifier | Experimental implementation | Role |
| --- | --- | --- |
| `FaaS-MASeA` | `MADEA-one-shot-detailed` | Multi-agent auction with detailed local initialization |
| `FaaS-MASeA(r)` | `MADEA-one-shot-selective` | Auction with selective memory requests and bounded retries |
| `PG` | `FaaS-MAPG-R-detailed-fixed-x` | Random-order potential-game search with initial local service fixed |
| `Greedy-RR` | `Greedy-RR` | Greedy neighbor selection and round-robin service |
| `centralized` | `centralized` | Centralized load-management optimization |
| `selfish-centralized` | `selfish-centralized` | Centralized optimization preserving each node's local gain floor |

The public names above are used in the CLI and result artifacts. PG is specifically
the fixed-x variant; ordinary PG and iterative MADEA are not selectable.

## Install

Python **3.10** preserves the experimental implementation's supported environment.
Install [uv](https://docs.astral.sh/uv/), then:

```sh
git clone https://github.com/Nanofaas/faas-masea.git
cd faas-masea
uv sync --locked
uv run --locked python run.py --list
```

CPU execution uses the existing dynamic-programming backend and open-source
HiGHS for centralized optimization and supported fallback solves. It does not
require a Gurobi license. Optional Gurobi tests require `uv sync --locked --extra gurobi`
and a usable license. The `cuda` extra supplies the retained optional CUDA backend;
a compatible NVIDIA host and driver are required. CPU defaults work on other hosts.

## Run an example

The bundled `examples/tiny` is a checksummed, generated ten-node input with two
load steps. The default configuration runs the first step, as in the paper's
single-step campaigns. Choose a new output directory for each run:

```sh
uv run --locked python run.py --algorithm FaaS-MASeA \
  --instance examples/tiny --output results/masea
uv run --locked python run.py --algorithm 'FaaS-MASeA(r)' \
  --instance examples/tiny --output results/masea-r
uv run --locked python run.py --algorithm PG \
  --instance examples/tiny --output results/pg
uv run --locked python run.py --algorithm Greedy-RR \
  --instance examples/tiny --output results/greedy
uv run --locked python run.py --algorithm centralized \
  --instance examples/tiny --time-limit 60 --output results/centralized
uv run --locked python run.py --algorithm selfish-centralized \
  --instance examples/tiny --time-limit 60 --output results/selfish
```

Use `--parallelism 4` for four local-solver workers (or four HiGHS threads for
centralized methods). `--config config/paper.json` selects decentralized settings:
seed 4850, at most 100 iterations, a 300 modelled-second algorithm budget and
logical communication accounting. These settings are not a new performance result.
For the DGX backend, install `--extra cuda` and set `dp_device` and `compute_device`
to `cuda` in `solver_options.general`; the default uses CPU.

Each invocation writes `result.json`, with method, objective values, status and
wall-clock duration. Decentralized runs also retain configuration, per-step
solutions, `obj.csv`, `runtime.csv`, logs and logical-message counts under `run/`.
Modelled algorithm time and elapsed wall time are distinct. Logical messages are
algorithmic accounting, not captured network packets. Centralized results include
the bound, termination and feasibility status; a time-limited incumbent is not
necessarily an optimum. Centralized reference solves use step 0 only.

Existing output directories are rejected. Input payloads are verified against
`metadata.json` before execution. To run other inputs, supply a materialized
instance directory containing `base_instance_data.json`, `load_limits.json`,
`input_requests_traces.json`, `graph.json`, and `metadata.json`.

## Validation

```sh
uv run --locked pytest -q
uv run --locked ruff check .
```

Tests check topology, centralized feasibility, communication accounting, output
names, fixed-x behavior, output preservation, and numerical equivalence with the
archived source on the bundled input, including two-worker execution. Optional
Gurobi tests skip explicitly if no usable installation/license is available.
See [validation and scope](docs/validation.md).

## Code and provenance

- `methods.py`: the paper's method registry.
- `run.py`: common CLI for the six methods.
- `decentralized_auction.py`: FaaS-MASeA and FaaS-MASeA(r).
- `decentralized_potentialgame.py`: PG with fixed local service.
- `decentralized_greedy_rr.py`: Greedy-RR.
- `catalog_centralized.py`, `models/model.py`, `models/selfish.py`: centralized references.
- `models/`, `utils/`, `generators/`: shared numerical models, runtime and input handling.

Some internal filenames retain upstream names to keep the extraction auditable.
`run_faasmacro.py` and `run_faasmadea.py` contain shared helpers, not the removed
MACrO/iterative-MADEA runners. `plasma/core` retains only shared replica-cost data
structures and DP helpers, not a PLASMA algorithm entry point.

Derived from [DFaaSOptimizer](https://github.com/unimib-datAI/DFaaSOptimizer),
commit `d41237229b5f7827eb3fff34e0a17e48df0755d6`, the archived experiment snapshot
containing all selected variants. This is not the current tip of the original
`feat/frank-wolfe-relaxed-replicas` worktree. See [provenance](docs/provenance.md)
and the upstream file hashes in `source-provenance.json`.

For citation, identify the paper title above and cite this repository with the
exact commit used. This extraction does not bundle the complete DGX campaign data
or regenerate the paper's tables.

## License and attribution

The original GNU Affero General Public License v3 is preserved in [LICENSE](LICENSE).
The implementation derives from the work of the DFaaSOptimizer contributors.
The upstream project also accompanies *Distributed replica allocation and load
balancing for Edge-Cloud FaaS: A cooperative multi-agent orchestration approach*
(Federica Filippini, Marin Lujak and Michele Ciavotta, JSA, 2026;
[DOI](https://doi.org/10.1016/j.sysarc.2026.103787)). This earlier paper is distinct
from the FaaS-MASeA article named above.
