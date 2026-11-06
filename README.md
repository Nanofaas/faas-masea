# FaaS-MASeA

Code for **Decentralizing Edge-Cloud FaaS Orchestration: A Multi-Agent Auction Approach**.

The six methods allocate function replicas and route requests across edge nodes
and the cloud:

- **FaaS-MASeA**: multi-agent auction.
- **FaaS-MASeA(r)**: auction with selective memory requests and bounded retries.
- **PG**: potential-game search with initial local service fixed.
- **Greedy-RR**: greedy neighbor selection and round-robin service.
- **centralized**: centralized optimization.
- **selfish-centralized**: centralized optimization preserving each node's local gain floor.

## Install and run

Requires Python 3.10 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run --locked python run.py --algorithm FaaS-MASeA \
  --instance examples/tiny --output results/masea
```

Replace `FaaS-MASeA` with any method above; quote `'FaaS-MASeA(r)'` in the shell.
Use `--parallelism 4` for four local workers or centralized solver threads.
`--help` lists the available options.

`examples/tiny` contains a checksummed ten-node example with two load steps.
The default `config/paper.json` runs the first step. Other instances use the same
five-file format: `base_instance_data.json`, `load_limits.json`,
`input_requests_traces.json`, `graph.json` and `metadata.json`.

Results include `result.json`, with objectives, status and elapsed time.
Decentralized runs also save decisions, configuration, modelled algorithm time
and logical-message counts. Centralized references solve step 0 and report the
bound and termination status. Modelled time differs from wall time; logical
messages are not captured network packets. Output directories must be new.

CPU execution uses dynamic programming and HiGHS, without a Gurobi license.
Optional CUDA support requires `uv sync --locked --extra cuda`, an NVIDIA host,
and `dp_device` and `compute_device` set to `cuda` in `solver_options.general`.

## Tests

```sh
uv run --locked pytest -q
uv run --locked ruff check .
```

Tests cover numerical results, feasibility, message counts and parallel execution.
Optional Gurobi cases skip when the solver/license is unavailable. CUDA and the
full DGX campaigns are not covered by the local CPU checks. The complete paper
dataset and results are not included in this repository.

## License

Derived from [DFaaSOptimizer](https://github.com/unimib-datAI/DFaaSOptimizer).
Distributed under the original [GNU AGPLv3 license](LICENSE).
