"""Select the opt-in local DP backend, with the configured Pyomo solver as fallback."""
from models.dp_solver import try_solve_local
from utils import step_time, run_log
from collections import Counter

log = run_log.logger(__name__)


def dp_enabled(solver_options):
  """Require a JSON boolean: strings such as "false" must not enable DP."""
  enabled = solver_options.get('use_dp', False)
  if not isinstance(enabled, bool):
    raise ValueError('use_dp must be a boolean')
  return enabled


def dp_device(solver_options):
  """Validate application device selection without importing CUDA."""
  enabled = dp_enabled(solver_options)
  device = solver_options.get('dp_device', 'cpu')
  if device not in ('cpu', 'cuda') or (device == 'cuda' and not enabled):
    raise ValueError('dp_device must be cpu or cuda; cuda requires use_dp=True')
  return device


def solve_agent_problem(model, data, solver_options, solver_name):
  """Use the same opt-in backend in sequential and multiprocessing SP calls."""
  dp_device(solver_options)
  if dp_enabled(solver_options):
    solution = try_solve_local(model, data)
    if solution is not None:
      return solution
  return model.solve(model.generate_instance(data), solver_options, solver_name)


def solve_agent_batch(data, agents, model, solver_options, solver_name, prices=None):
  """Solve an explicitly requested CUDA group in input order, in this process."""
  started = step_time.now()
  agents = list(agents)
  if compute_batch_enabled(model, solver_options) and len(agents)>=30:
    return _solve_compute_agent_batch(data, agents, model, solver_options, solver_name, prices)
  from utils.compute_backend import cuda_compute_enabled, record_operation
  if cuda_compute_enabled():
    record_operation('local_numeric','cpu',len(agents),
                     reason='direct phase failed performance gate or numeric batch below 30')
  if dp_device(solver_options) != 'cuda':
    raise ValueError('solve_agent_batch requires dp_device=cuda')
  from models.dp_solver import prepare_local_problem, finish_local_solution
  from models.dp_gpu import gpu_problem_fits, minimize_replica_costs_batch
  prepared, local_data, eligible, results = {}, {}, [], {}
  agents = list(agents)
  reasons = Counter()
  for agent in agents:
    local = data.copy()
    local[None] = data[None].copy()
    local[None]['whoami'] = {None: agent + 1}
    if prices is not None:
      local[None]['pi'] = {f + 1: price for f, price in enumerate(prices[agent])}
    local_data[agent] = local
    problem = prepare_local_problem(model, local)
    prepared[agent] = problem
    if (problem is not None and not problem.direct
        and gpu_problem_fits(problem.costs, problem.memory, problem.ram_cap)):
      eligible.append(agent)
    else:
      reasons['unsupported_or_direct_infeasible' if problem is None else
              'direct' if problem.direct else 'device_buffer_or_index_limit'] += 1
  gpu_count = len(eligible) if len(eligible) >= 2 else 0
  if not gpu_count:
    reasons['batch_cardinality'] += len(eligible)
  if gpu_count:
    problems = [prepared[a] for a in eligible]
    choices = minimize_replica_costs_batch([p.costs for p in problems],
                                         [p.memory for p in problems],
                                         [p.ram_cap for p in problems])
    for agent, choice in zip(eligible, choices):
      if choice is not None:
        results[agent] = finish_local_solution(prepared[agent], choice, runtime=0.)
      else:
        reasons['gpu_infeasible'] += 1
  for agent in agents:
    if agent not in results:
      results[agent] = solve_agent_problem(model, local_data[agent], solver_options, solver_name)
  latency = step_time.now() - started
  for result in results.values():
    result.update(runtime=latency, runtime_kind='batch_latency')
  with step_time.untimed():
    log.info('dp_device=cuda timing_basis=gpu_batch_wall gpu_problems=%d cpu_problems=%d cpu_reasons=%s',
             gpu_count, sum(reasons.values()), dict(reasons))
  return {a: results[a] for a in agents}


def compute_batch_enabled(model, options):
  from utils.compute_backend import cuda_compute_enabled
  return (cuda_compute_enabled() and dp_enabled(options) and dp_device(options)=='cuda'
          and model.name!='LSPr_x' and '_fixedr' not in model.name)


def _solve_compute_agent_batch(data, agents, model, options, solver_name, prices):
  started = step_time.now()
  step_time.mark_shared_device()
  from models.local_numeric_gpu import try_solve_local_batch_cuda
  agents = list(agents)
  batch = []
  for agent in agents:
    local = data.copy()
    local[None] = data[None].copy()
    local[None]['whoami'] = {None: agent+1}
    if prices is not None:
      local[None]['pi'] = {f+1:price for f,price in enumerate(prices[agent])}
    batch.append(local)
  solutions = try_solve_local_batch_cuda(model,batch)
  gpu_count = sum(solution is not None for solution in solutions)
  results = {}
  for agent,local,solution in zip(agents,batch,solutions):
    results[agent] = solution if solution is not None else solve_agent_problem(model,local,options,solver_name)
  latency = step_time.now()-started
  for solution in results.values():
    solution.update(runtime=latency,runtime_kind='batch_latency')
  with step_time.untimed():
    log.info('compute_device=cuda timing_basis=gpu_batch_wall gpu_problems=%d cpu_problems=%d cpu_reasons=unsupported_or_infeasible_or_limits_or_cardinality',gpu_count,len(agents)-gpu_count)
  return results
