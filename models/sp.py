from models.model import BaseLoadManagementModel, PYO_VAR_TYPE, PYO_PARAM_TYPE

import pyomo.environ as pyo


class SPAbstractModel(BaseLoadManagementModel):
  def __init__(self):
    super().__init__()
    self.name = "SPAbstractModel"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # index of the current node
    self.model.whoami = pyo.Param(
      within = pyo.NonNegativeIntegers, default = 1
    )
    # objective function weights
    self.model.alpha = pyo.Param(
      self.model.N, self.model.F, within = pyo.NonNegativeReals, default = 1
    )
    self.model.delta = pyo.Param(
      self.model.N, self.model.F, within = pyo.NonNegativeReals, default = 0.9
    )
    ###########################################################################
    # Problem variables
    ###########################################################################
    # number of enqueued requests
    self.model.x = pyo.Var(
      self.model.F,
      domain = PYO_VAR_TYPE
    )
    # number of reserved instances
    self.model.r = pyo.Var(
      self.model.F,
      domain = pyo.NonNegativeIntegers
    )


class LSP_v0(SPAbstractModel):
  def __init__(self):
    super().__init__()
    self.name = "LSP_v0"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # objective function weights
    self.model.pi = pyo.Param(
      self.model.F,
      within = pyo.NonNegativeReals, default = 0.0
    )
    ###########################################################################
    # Problem variables
    ###########################################################################
    # number of forwarded requests
    self.model.omega = pyo.Var(
      self.model.F,
      domain = PYO_VAR_TYPE
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.no_traffic_loss_v0 = pyo.Constraint(
      self.model.F, rule = self.no_traffic_loss_v0
    )
    self.model.utilization_equilibrium = pyo.Constraint(
      self.model.F, rule = self.utilization_equilibrium
    )
    self.model.utilization_equilibrium2 = pyo.Constraint(
      self.model.F, rule = self.utilization_equilibrium2
    )
    self.model.residual_capacity = pyo.Constraint(
      rule = self.residual_capacity
    )
    ###########################################################################
    # Objective function
    ###########################################################################
    self.set_objective(rule = self.minimize_processing_cost_v0)

  @staticmethod
  def no_traffic_loss_v0(model, f):
    return (
      model.x[f] + model.omega[f]
    ) == model.incoming_load[model.whoami,f]

  @staticmethod
  def utilization_equilibrium(model, f):
    return (
      model.demand[model.whoami,f] * (
        model.x[f]
      ) <= model.r[f] * model.max_utilization[f]
    )

  @staticmethod
  def utilization_equilibrium2(model, f):
    return (
      model.demand[model.whoami,f] * (
        model.x[f]
      ) >= (model.r[f] - 1) * model.max_utilization[f]
    )

  @staticmethod
  def residual_capacity(model):
    return sum(
      model.r[f] * model.memory_requirement[f] for f in model.F
    ) <= model.memory_capacity[model.whoami]

  @staticmethod
  def minimize_processing_cost_v0(model):
    return - (
      sum(
        (
          model.alpha[model.whoami,f] * model.x[f] +
          model.delta[model.whoami,f] * model.omega[f]
        ) / (model.incoming_load[model.whoami,f] or 1) for f in model.F
      )
    ) + sum(
      model.pi[f] * model.omega[f] / (
        model.incoming_load[model.whoami,f] or 1
      ) for f in model.F
    )

class LSP(LSP_v0):
  def __init__(self):
    super().__init__()
    self.name = "LSP"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # objective function weights
    self.model.gamma = pyo.Param(
      self.model.N, self.model.F, within = pyo.NonNegativeReals, default = 0.1
    )
    ###########################################################################
    # Problem variables
    ###########################################################################
    # number of rejected requests
    self.model.z = pyo.Var(
      self.model.F,
      domain = PYO_VAR_TYPE
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    # The inherited LSP_v0 balance (x + omega == load) is an equality with no
    # room for the cloud-rejection variable z: combined with the z-aware
    # balance below it would force z == 0 (silently disabling cloud rejection
    # and the gamma*z objective term, and making LSP_capped infeasible whenever
    # the offload cap is below the overflow). Drop it; no_traffic_loss is the
    # real flow-conservation constraint once z exists.
    self.model.del_component(self.model.no_traffic_loss_v0)
    self.model.no_traffic_loss = pyo.Constraint(
      self.model.F, rule = self.no_traffic_loss
    )
    ###########################################################################
    # Objective function
    ###########################################################################
    self.set_objective(rule = self.minimize_processing_cost)

  @staticmethod
  def no_traffic_loss(model, f):
    return (
      model.x[f] + model.omega[f] + model.z[f]
    ) == model.incoming_load[model.whoami,f]

  @staticmethod
  def minimize_processing_cost(model):
    return - (
      sum(
        (
          model.alpha[model.whoami,f] * model.x[f] +
          model.delta[model.whoami,f] * model.omega[f] -
          model.gamma[model.whoami,f] * model.z[f]
        ) / (model.incoming_load[model.whoami,f] or 1) for f in model.F
      )
    ) + sum(
      model.pi[f] * model.omega[f] / (
        model.incoming_load[model.whoami,f] or 1
      ) for f in model.F
    )

class LSP_detailed(SPAbstractModel):
  def __init__(self):
    super().__init__()
    self.name = "LSP_detailed"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # number and set of nodes
    # neighborhood (n_{ij}=1 if neighbors)
    self.model.neighborhood = pyo.Param(
      self.model.N, self.model.N, within = pyo.Binary,
      default = 0
    )
    # objective function weights
    self.model.pi = pyo.Param(
      self.model.F,
      within = pyo.NonNegativeReals, default = 0.0
    )
    self.model.beta = pyo.Param(
      self.model.N, self.model.N, self.model.F,
      within = pyo.NonNegativeReals, default = 0.9
    )
    self.model.gamma = pyo.Param(
      self.model.N, self.model.F, within = pyo.NonNegativeReals, default = 0.1
    )
    ###########################################################################
    # Problem variables
    ###########################################################################
    # number of rejected requests
    self.model.y = pyo.Var(
      self.model.N, self.model.F,
      domain = PYO_VAR_TYPE
    )
    self.model.z = pyo.Var(
      self.model.F,
      domain = PYO_VAR_TYPE
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.no_traffic_loss = pyo.Constraint(
      self.model.F, rule = self.no_traffic_loss
    )
    self.model.utilization_equilibrium = pyo.Constraint(
      self.model.F, rule = self.utilization_equilibrium
    )
    self.model.utilization_equilibrium2 = pyo.Constraint(
      self.model.F, rule = self.utilization_equilibrium2
    )
    self.model.residual_capacity = pyo.Constraint(
      rule = self.residual_capacity
    )
    self.model.offload_only_to_neighbors = pyo.Constraint(
      self.model.N, self.model.F, rule = self.offload_only_to_neighbors
    )
    ###########################################################################
    # Objective function
    ###########################################################################
    self.set_objective(rule = self.maximize_processing, sense = pyo.maximize)

  @staticmethod
  def offload_only_to_neighbors(model, n, f):
    return model.y[n,f] <= (
      model.incoming_load[model.whoami,f] * model.neighborhood[model.whoami,n]
    )

  @staticmethod
  def maximize_processing(model):
    """
    Equivalent to minimizing the original cost, with a positive gain sign
    """
    return -LSP_detailed.minimize_processing_cost(model)

  @staticmethod
  def no_traffic_loss(model, f):
    return (
      model.x[f] + sum(model.y[n,f] for n in model.N) + model.z[f]
    ) == model.incoming_load[model.whoami,f]

  @staticmethod
  def utilization_equilibrium(model, f):
    return (
      model.demand[model.whoami,f] * (
        model.x[f]
      ) <= model.r[f] * model.max_utilization[f]
    )

  @staticmethod
  def utilization_equilibrium2(model, f):
    return (
      model.demand[model.whoami,f] * (
        model.x[f]
      ) >= (model.r[f] - 1) * model.max_utilization[f]
    )

  @staticmethod
  def residual_capacity(model):
    return sum(
      model.r[f] * model.memory_requirement[f] for f in model.F
    ) <= model.memory_capacity[model.whoami]

  @staticmethod
  def minimize_processing_cost(model):
    return - (
      sum(
        (
          model.alpha[model.whoami,f] * model.x[f] +
          sum(
            model.beta[model.whoami,n,f] * model.y[n,f] for n in model.N
          ) -
          model.gamma[model.whoami,f] * model.z[f]
        ) / (model.incoming_load[model.whoami,f] or 1) for f in model.F
      )
    ) + sum(
      model.pi[f] * sum(
        model.y[n,f] for n in model.N
      ) / (model.incoming_load[model.whoami,f] or 1) for f in model.F
    )


class LSPr_v0(SPAbstractModel):
  def __init__(self):
    super().__init__()
    self.name = "LSPr_v0"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # assigned offloading
    self.model.omega_bar = pyo.Param(
      self.model.N, self.model.F,
      within = PYO_PARAM_TYPE
    )
    self.model.y_bar = pyo.Param(
      self.model.N, self.model.N, self.model.F,
      within = PYO_PARAM_TYPE
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.no_traffic_loss_v0 = pyo.Constraint(
      self.model.F, rule = self.no_traffic_loss_v0
    )
    self.model.utilization_equilibrium = pyo.Constraint(
      self.model.F, rule = self.utilization_equilibrium
    )
    self.model.utilization_equilibrium2 = pyo.Constraint(
      self.model.F, rule = self.utilization_equilibrium2
    )
    self.model.residual_capacity = pyo.Constraint(
      rule = self.residual_capacity
    )
    ###########################################################################
    # Objective function
    ###########################################################################
    self.set_objective(rule = self.minimize_processing_cost_v0)

  @staticmethod
  def no_traffic_loss_v0(model, f):
    return (
      model.x[f] + model.omega_bar[model.whoami,f]
    ) <= model.incoming_load[model.whoami,f]

  @staticmethod
  def utilization_equilibrium(model, f):
    return (
      model.demand[model.whoami,f] * (
        model.x[f] + sum(model.y_bar[m,model.whoami,f] for m in model.N)
      ) <= model.r[f] * model.max_utilization[f]
    )

  @staticmethod
  def utilization_equilibrium2(model, f):
    return (
      model.demand[model.whoami,f] * (
        model.x[f] + sum(model.y_bar[m,model.whoami,f] for m in model.N)
      ) >= (model.r[f] - 1) * model.max_utilization[f]
    )

  @staticmethod
  def residual_capacity(model):
    return sum(
      model.r[f] * model.memory_requirement[f] for f in model.F
    ) <= model.memory_capacity[model.whoami]

  @staticmethod
  def minimize_processing_cost_v0(model):
    return - (
      sum(
        (
          model.alpha[model.whoami,f] * model.x[f] +
          model.delta[model.whoami,f] * model.omega_bar[model.whoami,f]
        ) / (model.incoming_load[model.whoami,f] or 1) for f in model.F
      )
    )

class LSPr(LSPr_v0):
  def __init__(self):
    super().__init__()
    self.name = "LSPr"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # objective function weights
    self.model.gamma = pyo.Param(
      self.model.N, self.model.F, within = pyo.NonNegativeReals, default = 0.1
    )
    ###########################################################################
    # Problem variables
    ###########################################################################
    # number of rejected requests
    self.model.z = pyo.Var(
      self.model.F,
      domain = PYO_PARAM_TYPE
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.del_component(self.model.no_traffic_loss_v0)
    self.model.no_traffic_loss = pyo.Constraint(
      self.model.F, rule = self.no_traffic_loss
    )
    ###########################################################################
    # Objective function
    ###########################################################################
    self.set_objective(rule = self.minimize_processing_cost)

  @staticmethod
  def no_traffic_loss(model, f):
    return (
      model.x[f] + model.omega_bar[model.whoami,f] + model.z[f]
    ) == model.incoming_load[model.whoami,f]

  @staticmethod
  def minimize_processing_cost(model):
    return - (
      sum(
        (
          model.alpha[model.whoami,f] * model.x[f] +
          model.delta[model.whoami,f] * model.omega_bar[model.whoami,f] -
          model.gamma[model.whoami,f] * model.z[f]
        ) / (model.incoming_load[model.whoami,f] or 1) for f in model.F
      )
    )


class LSPr_x(LSPr_v0):
  def __init__(self):
    super().__init__()
    self.name = "LSPr_x"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    self.model.x_bar = pyo.Param(
      self.model.N, self.model.F,
      within = PYO_PARAM_TYPE
    )
    # objective function weights
    self.model.gamma = pyo.Param(
      self.model.N, self.model.F, within = pyo.NonNegativeReals, default = 0.1
    )
    ###########################################################################
    # Problem variables
    ###########################################################################
    # number of rejected requests
    self.model.z = pyo.Var(
      self.model.F,
      domain = PYO_PARAM_TYPE
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.del_component(self.model.no_traffic_loss_v0)
    self.model.no_traffic_loss = pyo.Constraint(
      self.model.F, rule = self.no_traffic_loss
    )
    self.model.temp_fix_x = pyo.Constraint(
      self.model.F, rule = self.temp_fix_x
    )
    ###########################################################################
    # Objective function
    ###########################################################################
    self.set_objective(rule = self.minimize_processing_cost)

  @staticmethod
  def no_traffic_loss(model, f):
    return (
      model.x[f] + model.omega_bar[model.whoami,f] + model.z[f]
    ) == model.incoming_load[model.whoami,f]

  @staticmethod
  def temp_fix_x(model, f):
    return (
      model.x[f] == model.x_bar[model.whoami,f]
    )

  @staticmethod
  def minimize_processing_cost(model):
    return - (
      sum(
        (
          model.alpha[model.whoami,f] * model.x[f] +
          model.delta[model.whoami,f] * model.omega_bar[model.whoami,f] -
          model.gamma[model.whoami,f] * model.z[f]
        ) / (model.incoming_load[model.whoami,f] or 1) for f in model.F
      )
    )


##############################################################################
# TEMPORARILY FIX r
##############################################################################

class LSP_fixedr_v0(LSP_v0):
  def __init__(self):
    super().__init__()
    self.name = "LSP_fixedr_v0"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # number of assigned replicas
    self.model.r_bar = pyo.Param(
      self.model.N, self.model.F,
      within = pyo.NonNegativeIntegers, default = 0
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.fix_r = pyo.Constraint(
      self.model.F, rule = self.fix_r
    )

  @staticmethod
  def fix_r(model, f):
    return model.r[f] == model.r_bar[model.whoami,f]

class LSP_fixedr(LSP):
  def __init__(self):
    super().__init__()
    self.name = "LSP_fixedr"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # number of assigned replicas
    self.model.r_bar = pyo.Param(
      self.model.N, self.model.F,
      within = pyo.NonNegativeIntegers, default = 0
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.fix_r = pyo.Constraint(
      self.model.F, rule = self.fix_r
    )
    self.model.del_component(self.model.utilization_equilibrium2)

  @staticmethod
  def fix_r(model, f):
    return model.r[f] == model.r_bar[model.whoami,f]


class LSPr_fixedr(LSPr):
  def __init__(self):
    super().__init__()
    self.name = "LSPr_fixedr"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # assigned offloading
    self.model.r_bar = pyo.Param(
      self.model.N, self.model.F,
      within = pyo.NonNegativeIntegers
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.fix_r = pyo.Constraint(
      self.model.F, rule = self.fix_r
    )

  @staticmethod
  def fix_r(model, f):
    return model.r[f] == model.r_bar[model.whoami,f]


##############################################################################
# OFFLOADING CAP (for FaaS-MABR-O capped local best response)
##############################################################################

class LSP_capped(LSP):
  def __init__(self):
    super().__init__()
    self.name = "LSP_capped"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # per-function upper bound on horizontal offloading
    self.model.omega_ub = pyo.Param(
      self.model.F, within = pyo.NonNegativeReals, default = 1e9
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.cap_offloading = pyo.Constraint(
      self.model.F, rule = self.cap_offloading
    )

  @staticmethod
  def cap_offloading(model, f):
    return model.omega[f] <= model.omega_ub[f]


class LSP_capped_fixedr(LSP_fixedr):
  def __init__(self):
    super().__init__()
    self.name = "LSP_capped_fixedr"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # per-function upper bound on horizontal offloading
    self.model.omega_ub = pyo.Param(
      self.model.F, within = pyo.NonNegativeReals, default = 1e9
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.cap_offloading = pyo.Constraint(
      self.model.F, rule = self.cap_offloading
    )

  @staticmethod
  def cap_offloading(model, f):
    return model.omega[f] <= model.omega_ub[f]


##############################################################################
# POTENTIAL GAME (FaaS-MAPG node proposal model)
##############################################################################

class LSP_pg(LSP):
  def __init__(self):
    super().__init__()
    self.name = "LSP_pg"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # committed inbound offloading (other players' strategies, fixed)
    self.model.y_bar = pyo.Param(
      self.model.N, self.model.N, self.model.F,
      within = PYO_PARAM_TYPE, default = 0.0
    )
    # per-function upper bound on horizontal offloading
    self.model.omega_ub = pyo.Param(
      self.model.F, within = pyo.NonNegativeReals, default = 1e9
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    # replicas must also cover committed inbound flows (LSPr-style)
    self.model.del_component(self.model.utilization_equilibrium)
    self.model.del_component(self.model.utilization_equilibrium2)
    self.model.utilization_equilibrium = pyo.Constraint(
      self.model.F, rule = self.utilization_equilibrium_pg
    )
    self.model.utilization_equilibrium2 = pyo.Constraint(
      self.model.F, rule = self.utilization_equilibrium2_pg
    )
    self.model.cap_offloading = pyo.Constraint(
      self.model.F, rule = self.cap_offloading
    )

  @staticmethod
  def utilization_equilibrium_pg(model, f):
    return (
      model.demand[model.whoami,f] * (
        model.x[f] + sum(model.y_bar[m,model.whoami,f] for m in model.N)
      ) <= model.r[f] * model.max_utilization[f]
    )

  @staticmethod
  def utilization_equilibrium2_pg(model, f):
    return (
      model.demand[model.whoami,f] * (
        model.x[f] + sum(model.y_bar[m,model.whoami,f] for m in model.N)
      ) >= (model.r[f] - 1) * model.max_utilization[f]
    )

  @staticmethod
  def cap_offloading(model, f):
    return model.omega[f] <= model.omega_ub[f]


class LSP_pg_fixedx(LSP_pg):
  """PG proposal with local service pinned to its detailed initialization."""
  def __init__(self):
    super().__init__()
    self.name = "LSP_pg_fixedx"
    self.model.x_bar = pyo.Param(self.model.N, self.model.F, within=PYO_PARAM_TYPE)
    self.model.fix_x = pyo.Constraint(
      self.model.F, rule=lambda model, f: model.x[f] == model.x_bar[model.whoami, f])


class LSP_pg_fixedr(LSP_pg):
  def __init__(self):
    super().__init__()
    self.name = "LSP_pg_fixedr"
    ###########################################################################
    # Problem parameters
    ###########################################################################
    # number of assigned replicas
    self.model.r_bar = pyo.Param(
      self.model.N, self.model.F,
      within = pyo.NonNegativeIntegers, default = 0
    )
    ###########################################################################
    # Constraints
    ###########################################################################
    self.model.fix_r = pyo.Constraint(
      self.model.F, rule = self.fix_r
    )
    # with r pinned the replica lower bound may conflict with commitments
    # (same reasoning as LSP_fixedr)
    self.model.del_component(self.model.utilization_equilibrium2)

  @staticmethod
  def fix_r(model, f):
    return model.r[f] == model.r_bar[model.whoami,f]
