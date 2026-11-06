from utils.common import load_requests_traces, run_steps

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from typing import Tuple
from parse import parse
import pandas as pd
import numpy as np
import argparse
import json
import os


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description = "Compare results",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
    "-i", "--postprocessing_folder",
    help = "Results folder",
    type = str,
    required = True
  )
  parser.add_argument(
    "--models",
    help = "List of model names",
    nargs = "*",
    default = ["LoadManagementModel", "FaaS-MACrO"]
  )
  parser.add_argument(
    "--plot_all_history",
    help = "True to plot each experiment history (default: false)",
    default = False,
    action = "store_true"
  )
  # Parse the arguments
  args: argparse.Namespace = parser.parse_known_args()[0]
  return args


def add_node_function_info(df: pd.DataFrame, cols: np.array) -> pd.DataFrame:
  updated_df = df.loc[:, cols].transpose()
  updated_df["node"] = [i[0] for i in updated_df.index.str.split("_")]
  updated_df["function"] = [i[1] for i in updated_df.index.str.split("_")]
  return updated_df

def count_subcols(
    df: pd.DataFrame, cols: np.array, all_models_count: dict, model_name: str
  ) -> pd.DataFrame:
  subcols = add_node_function_info(df, cols)
  tot_by_node = group_count(subcols, "node")
  tot_by_function = group_count(subcols, "function")
  for col in tot_by_node:
    if col not in all_models_count["by_node"]:
      all_models_count["by_node"][col] = pd.DataFrame()
    all_models_count["by_node"][col][model_name] = tot_by_node[col]
  for col in tot_by_function:
    if col not in all_models_count["by_function"]:
      all_models_count["by_function"][col] = pd.DataFrame()
    all_models_count["by_function"][col][model_name] = tot_by_function[col]
  return all_models_count

def group_count(
    df: pd.DataFrame, groupby_key: str
  ) -> pd.DataFrame:
  tot = df.groupby(groupby_key).sum(numeric_only = True)
  tot.columns = [f"t = {t}" for t in tot.columns]
  tot["tot"] = tot.sum(axis = "columns")
  return tot

def invert_count(original_dict: dict) -> pd.DataFrame:
  flattened = [
    ((outer_key, inner_key), value)
    for outer_key, inner_dict in original_dict.items()
    for inner_key, value in inner_dict.items()
  ]
  df = pd.DataFrame.from_dict(dict(flattened), orient='index')
  df.index = pd.MultiIndex.from_tuples(df.index, names=["time", "model"])
  return df

def load_models_results(
    solution_folder: str,
    models_keys: list,
    models_names: list,
    plot_all_history: bool = False
  ) -> Tuple[dict, dict, dict, dict]:
  all_models_local_count = {"by_node": {}, "by_function": {}}
  all_models_fwd_count = {"by_node": {}, "by_function": {}}
  all_models_rej_count = {"by_node": {}, "by_function": {}}
  all_models_replicas = {"by_node": {}, "by_function": {}}
  all_models_ping_pong = {}
  for model_key, model_name in zip(models_keys, models_names):
    if os.path.exists(
        os.path.join(solution_folder, f"{model_key}_solution.csv")
      ):
      # load solution
      solution, replicas, detailed_fwd_sol, utilization, obj = load_solution(
        solution_folder, model_key
      )
      if plot_all_history:
        # load input requests
        requests, mt, Mt, ts = load_requests_traces(solution_folder)
        # plot
        plot_history(
          requests,
          mt,
          Mt,
          ts,
          solution,
          utilization,
          replicas,
          pd.read_csv(
            os.path.join(solution_folder, f"{model_key}_offloaded.csv")
          ),
          obj,
          os.path.join(solution_folder, f"{model_key}.png")
        )
      # count locally-processed requests per node/class
      local_cols = solution.columns.str.endswith("_loc")
      all_models_local_count = count_subcols(
        solution, local_cols, all_models_local_count, model_name
      )
      # count forwarded requests per node/class
      fwd_cols = solution.columns.str.endswith("_fwd")
      all_models_fwd_count = count_subcols(
        solution, fwd_cols, all_models_fwd_count, model_name
      )
      # count rejected requests per node/class
      rej_cols = ~(local_cols) & ~(fwd_cols)
      all_models_rej_count = count_subcols(
        solution, rej_cols, all_models_rej_count, model_name
      )
      # count replicas per node/class
      all_models_replicas = count_subcols(
        replicas, replicas.columns, all_models_replicas, model_name
      )
      # check ping-pong
      nodes, functions = map(
        set, zip(*(s.split('_') for s in solution.columns[rej_cols]))
      )
      all_models_ping_pong[model_name] = []
      for n1 in nodes:
        for f in functions:
          for n2 in nodes:
            if n1 != n2:
              times_fwd = np.where(
                detailed_fwd_sol.loc[:,f"{n1}_{f}_{n2}_tot"] > 0
              )[0]
              if len(times_fwd) > 0:
                times_bwd = np.where(
                  detailed_fwd_sol.loc[:,f"{n2}_{f}_{n1}_tot"] > 0
                )[0]
                for t in times_fwd:
                  if t in times_bwd:
                    all_models_ping_pong[model_name].append((n1,f,n2,detailed_fwd_sol.index[t]))
  return (
    all_models_local_count,
    all_models_fwd_count,
    all_models_rej_count,
    all_models_replicas,
    all_models_ping_pong
  )


def result_steps(solution_folder: str, count: int):
  """Recover physical steps, including legacy CSVs whose rows omit the step.

  Modern runtime.csv is authoritative. Historical folders use config.json;
  folders without either retain the original positional indexing.
  """
  runtime_path = os.path.join(solution_folder, "runtime.csv")
  if os.path.exists(runtime_path):
    runtime = pd.read_csv(runtime_path)
    if "step" in runtime:
      if len(runtime) < count:
        raise ValueError("runtime.csv has fewer steps than the solution")
      return runtime["step"].iloc[:count].to_numpy()
  config_path = os.path.join(solution_folder, "config.json")
  if os.path.exists(config_path):
    with open(config_path) as stream:
      config = json.load(stream)
    if "max_steps" in config:
      steps = run_steps(config)
      if len(steps) < count:
        raise ValueError("config.json has fewer steps than the solution")
      return steps[:count]
  return range(count)


def load_solution(
    solution_folder: str, model_name: str
  ) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
  ]:
  solution = pd.read_csv(
    os.path.join(solution_folder, f"{model_name}_solution.csv")
  )
  replicas = pd.read_csv(
    os.path.join(solution_folder, f"{model_name}_replicas.csv")
  )
  detailed_fwd_solution = pd.read_csv(
    os.path.join(solution_folder, f"{model_name}_detailed_fwd_solution.csv")
  )
  utilization = pd.read_csv(
    os.path.join(solution_folder, f"{model_name}_utilization.csv")
  )
  obj = pd.DataFrame()
  if os.path.exists(os.path.join(solution_folder, "obj.csv")):
    obj = pd.read_csv(os.path.join(solution_folder, "obj.csv"))
    for key in ["LSP", "SP/coord"]:
      if key in obj:
        obj.rename(columns = {key: "FaaS-MACrO"}, inplace = True)
  steps = result_steps(solution_folder, len(solution))
  for frame in (solution, replicas, detailed_fwd_solution, utilization, obj):
    if len(frame):
      frame.index = steps
  return solution, replicas, detailed_fwd_solution, utilization, obj


def plot_count(
    df: pd.DataFrame, groupby_key: str, plt_folder: str, plot_all: bool = False
  ) -> pd.DataFrame:
  tot = group_count(df, groupby_key)
  if plot_all:
    tot.plot.bar(
      rot = 0,
      grid = True,
      fontsize = 14
    )
    plt.savefig(
      os.path.join(plt_folder, f"{groupby_key}.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  else:
    tot.plot.bar(
      y = "tot",
      rot = 0,
      fontsize = 14
    )
    if len(tot) <= 10:
      plt.grid(True, axis = "both")
    else:
      plt.grid(True, axis = "y")
    plt.savefig(
      os.path.join(plt_folder, f"{groupby_key}_tot.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  return tot

def plot_global_count(
    all_models_count: dict,
    title: str,
    plot_all: bool = False,
    plot_folder: str = None,
    logy: bool = False
  ):
  for key, all_models_count_by_key in all_models_count.items():
    if plot_all:
      for col, df in all_models_count_by_key.items():
        df.plot.bar(rot = 0, logy = logy, grid = True)
        plt.grid(True, which = "both")
        plt.title(f"{title} {key} --> {col}")
        if plot_folder is not None:
          plt.savefig(
            os.path.join(
              plot_folder, f"{title}-{key}-{col.replace(' = ', '')}.png"
            ),
            dpi = 300,
            format = "png",
            bbox_inches = "tight"
          )
          plt.close()
        else:
          plt.show()
    else:
      if "tot" in all_models_count_by_key:
        all_models_count_by_key["tot"].plot.bar(
          rot = 0, logy = logy
        )
        if len(all_models_count_by_key["tot"]) <= 10:
          plt.grid(True, which = "both", axis = "both")
        else:
          plt.grid(True, which = "both", axis = "y")
        plt.title(f"{title} {key}")
        if plot_folder is not None:
          plt.savefig(
            os.path.join(
              plot_folder, f"{title}-{key}-tot.png"
            ),
            dpi = 300,
            format = "png",
            bbox_inches = "tight"
          )
          plt.close()
        else:
          plt.show()

def plot_history(
    input_requests_traces: dict,
    min_run_time: int,
    max_run_time: int,
    run_time_step: int,
    solution: pd.DataFrame,
    utilization: pd.DataFrame,
    replicas: pd.DataFrame,
    offloaded: pd.DataFrame,
    obj_values: list,
    plot_filename: str = None
  ):
  # Bars use row positions; align line plots to those positions as well.
  solution, utilization, replicas, offloaded = (
    frame.reset_index(drop=True) for frame in (solution, utilization, replicas, offloaded)
  )
  Nf = len(input_requests_traces)
  Nn = len(input_requests_traces[0])
  max_workload = {
    f: max(
      [max(ll) for ll in input_requests_traces[f].values()]
    ) for f in range(Nf)
  }
  min_workload = {
    f: 0 for f in range(Nf)
    # min(
    #   [min(ll) for ll in input_requests_traces[f].values()]
    # ) for f in range(Nf)
  }
  time = range(min_run_time, max_run_time, run_time_step)
  if max_run_time == min_run_time:
    time = range(min_run_time, max_run_time + run_time_step, run_time_step)
  _, axs = plt.subplots(
    nrows = Nn,
    ncols = 2 * Nf + 3,
    sharex = True,
    figsize = ((2 * Nf + 3) * 8, 6 * Nn),
    squeeze = False
  )
  for function, traces in input_requests_traces.items():
    for agent, incoming_load in traces.items():
      # incoming load
      axs[agent,function].plot(
        range(len(time)),
        incoming_load[list(time)],
        ".-",
        color = "k"
      )
      # requests management
      solution.loc[
        :,solution.columns.str.startswith(f"n{agent}_f{function}")
      ].plot.bar(
        stacked = True,
        ax = axs[agent,function],
        rot = 0
      )
      # received offloads
      if len(offloaded) > 0:
        offloaded.loc[
          :,offloaded.columns.str.startswith(f"n{agent}_f{function}")
        ].plot.bar(
          stacked = True,
          ax = axs[agent,Nf+function],
          rot = 0
        )
      # utilization
      utilization.loc[
        :,utilization.columns.str.startswith(f"n{agent}_f{function}")
      ].plot(
        marker = ".",
        # color = "r",
        ax = axs[agent,-3]
      )
      # replicas
      replicas.loc[
        :,replicas.columns.str.startswith(f"n{agent}_f{function}")
      ].plot(
        marker = ".",
        # color = "b",
        ax = axs[agent,-2]
      )
      # axis properties
      axs[agent,function].grid(axis = "y")
      axs[agent,Nf+function].grid(axis = "y")
      axs[agent,-3].grid(axis = "y")
      axs[agent,-2].grid(axis = "y")
      axs[agent,function].set_ylim(
        min_workload[function], max_workload[function]
      )
      axs[agent,Nf+function].set_ylim(
        min_workload[function], max_workload[function]
      )
      axs[agent,function].set_xticks(range(len(time)), time)
      axs[agent,Nf+function].set_xticks(range(len(time)), time)
      axs[agent,-3].set_xticks(range(len(time)), time)
      axs[agent,-2].set_xticks(range(len(time)), time)
  # objective function value
  axs[0,-1].plot(
    range(len(obj_values)), obj_values, ".-", linewidth = 2, color = "r"
  )
  axs[0,-1].grid(axis = "y")
  if plot_filename is not None:
    plt.savefig(
      plot_filename, dpi = 300, format = "png", bbox_inches = "tight"
    )
    plt.close()
  else:
    plt.show()

def process_results(
    solution_folder: str, models: list, plot_all_history: bool = False
  ) -> str:
  (
    all_models_local_count,
    all_models_fwd_count,
    all_models_rej_count,
    all_models_replicas,
    all_models_ping_pong
  ) = load_models_results(solution_folder, models, models, plot_all_history)
  # create folder to store plots
  plot_folder = os.path.join(solution_folder, "postprocessing")
  os.makedirs(plot_folder, exist_ok = True)
  #
  for key in all_models_local_count:
    if len(all_models_local_count[key]) > 0:
      model = all_models_local_count[key]["tot"].columns[0]
      tot_load = (
        all_models_local_count[key]["tot"] +
        all_models_fwd_count[key]["tot"] +
        all_models_rej_count[key]["tot"]
      )
      amlc = (
        all_models_local_count[key]["tot"] / tot_load
      ).rename(columns={model: "ratio"})
      amfc = (
        all_models_fwd_count[key]["tot"] / tot_load
      ).rename(columns={model: "ratio"})
      amrc = (
        all_models_rej_count[key]["tot"] / tot_load
      ).rename(columns={model: "ratio"})
      all_tot = amlc.join(amfc, lsuffix = "_loc", rsuffix = "_fwd").join(amrc)
      ax = all_tot.plot.bar(
        rot = 0,
        stacked = True
      )
      (amlc + amfc).rename(columns={"ratio": "PROCESSED"}).plot(
        ax = ax,
        color = mcolors.TABLEAU_COLORS["tab:red"],
        linewidth = 2
      )
      if len(all_tot) <= 10:
        plt.grid(True, axis = "both")
      else:
        plt.grid(True, axis = "y")
      plt.ylim((0,1.05))
      # plt.title(key)
      plt.savefig(
        os.path.join(
          plot_folder, f"{key}-ALL.png"
        ),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
      )
      plt.close()
  # requests
  plot_global_count(
    all_models_local_count,
    "local",
    plot_all = False,
    plot_folder = plot_folder
  )
  plot_global_count(
    all_models_fwd_count,
    "fwd",
    plot_all = False,
    plot_folder = plot_folder
  )
  plot_global_count(
    all_models_rej_count,
    "rej",
    plot_all = False,
    plot_folder = plot_folder
  )
  # replicas
  plot_global_count(
    all_models_replicas,
    "replicas",
    plot_all = False,
    plot_folder = plot_folder,
    logy = True
  )
  # plot total processing per function
  inv_local_count = invert_count(all_models_local_count["by_function"])
  inv_fwd_count = invert_count(all_models_fwd_count["by_function"])
  inv_rej_count = invert_count(all_models_rej_count["by_function"])
  inv_count = inv_local_count.join(
    inv_fwd_count, lsuffix = "_local", rsuffix = "_fwd"
  ).join(inv_rej_count).reset_index()
  functions = [
    int(
      parse("f{}_local", c)[0]
    ) for c in inv_count.columns if c.endswith("_local")
  ]
  for m in models:
    if len(functions) <= 10:
      for f in functions:
        curr_df = inv_count.loc[
          (inv_count["time"] != "tot") & (inv_count["model"] == m),
          inv_count.columns.str.contains(f"f{f}")
        ]
        if len(curr_df) > 0:
          curr_df.plot.bar(
            stacked = True,
            rot = 0
          )
          plt.grid(axis = "y")
          plt.savefig(
            os.path.join(
                plot_folder, f"{m}_f{f}.png"
              ),
              dpi = 300,
              format = "png",
              bbox_inches = "tight"
          )
          plt.close()
          # plot ratio
          ratio_df = curr_df.copy(deep = True)
          tot = ratio_df.sum(axis = "columns")
          ratio_df[f"f{f}_local"] = ratio_df[f"f{f}_local"] / tot * 100
          ratio_df[f"f{f}_fwd"] = ratio_df[f"f{f}_fwd"] / tot * 100
          ratio_df[f"f{f}"] = ratio_df[f"f{f}"] / tot * 100
          _, ax = plt.subplots()#figsize = (15,6))
          ratio_df.plot.bar(
            stacked = True,
            rot = 0,
            ax = ax
          )
          ratio_df[f"f{f}_PROCESSED"] = ratio_df[f"f{f}_local"] + ratio_df[f"f{f}_fwd"]
          ratio_df[f"f{f}_PROCESSED"].plot(
            ax = ax,
            color = mcolors.TABLEAU_COLORS["tab:red"]
          )
          if len(ratio_df) > 50:
            tt = len(ratio_df)
            plt.xticks(range(0,tt+1,10), list(range(0,tt+1,10)))
          plt.xlabel("Control time period $t$")
          plt.ylabel("Percentage of local/forwarded/rejected requests")
          plt.grid(axis = "y")
          plt.savefig(
            os.path.join(
                plot_folder, f"{m}_f{f}_ratio.png"
              ),
              dpi = 300,
              format = "png",
              bbox_inches = "tight"
          )
          plt.close()
  # save
  inv_count.to_csv(
    os.path.join(plot_folder, "by_node_count.csv"), index = False
  )
  # ping-pong
  pd.DataFrame(all_models_ping_pong).to_csv(
    os.path.join(plot_folder, "ping_pong.csv"), index = False
  )
  return plot_folder


def load_runtime(solution_folder: str) -> pd.DataFrame:
  """Per-step runtime of one run: column "runtime" (algorithm_s of the five-column
  files, else the historical single column), "method" (the obj.csv label) and,
  when the termination file names it, "termination_condition" and "color"."""
  runtime = pd.read_csv(os.path.join(solution_folder, "runtime.csv"))
  colname = runtime.columns[0]
  obj_path = os.path.join(solution_folder, "obj.csv")
  label = pd.read_csv(obj_path, nrows = 0).columns[0] if os.path.exists(obj_path) else colname
  values = runtime["algorithm_s"] if "algorithm_s" in runtime else runtime[colname]
  runtime = pd.DataFrame({"runtime": values, "method": label})
  runtime.index = result_steps(solution_folder, len(runtime))
  # -- termination condition
  tc_path = os.path.join(solution_folder, "termination_condition.csv")
  if os.path.exists(tc_path):
    tc = pd.read_csv(tc_path)
    if label in tc:
      runtime["color"] = [
        mcolors.TABLEAU_COLORS["tab:green"] if c == "optimal"
          else mcolors.TABLEAU_COLORS["tab:red"]
            for c in tc[label]
      ]
      runtime["termination_condition"] = tc[label].to_numpy()
  return runtime


def runtime_obj_boxplot(
    df: pd.DataFrame, colname: str, plot_folder: str, title: str
  ):
  nmethods = len(df["method"].unique())
  _, axs = plt.subplots(
    nrows = nmethods,
    ncols = 1,
    figsize = (21,6)
  )
  idx = 0
  for method, mdata in df.groupby("method"):
    bplot = mdata.plot.box(
      column = colname,
      by = "Nn",
      logy = True if colname == "runtime" else False,
      grid = True,
      showmeans = True,
      return_type = "dict",
      patch_artist = True,
      ax = axs if nmethods == 1 else axs[idx]
    )
    # colors
    for patch in bplot[colname]["boxes"]:
      patch.set_facecolor(mcolors.CSS4_COLORS["lightskyblue"])
    for median in bplot[colname]["medians"]:
      median.set_color(mcolors.TABLEAU_COLORS["tab:orange"])
    for mean in bplot[colname]["means"]:
      mean.set_markerfacecolor(mcolors.TABLEAU_COLORS["tab:red"])
      mean.set_markeredgecolor(mcolors.TABLEAU_COLORS["tab:red"])
    if nmethods == 1:
      axs.set_ylabel(method)
    else:
      axs[idx].set_ylabel(method)
    idx += 1
  plt.grid(True, which = "both")
  plt.title(None)
  plt.savefig(
    os.path.join(plot_folder, f"{title}.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()


if __name__ == "__main__":
  args = parse_arguments()
  base_solution_folder = args.postprocessing_folder
  models = args.models
  plot_all_history = args.plot_all_history
  # solution_folder = base_solution_folder
  # process_results(solution_folder, models)
  # load results
  all_runtimes = pd.DataFrame()
  all_obj = pd.DataFrame()
  for foldername in os.listdir(base_solution_folder):
    solution_folder = os.path.join(base_solution_folder, foldername)
    if (
        os.path.isdir(solution_folder) and
          not foldername.startswith(".DS_") and
            foldername != "postprocessing"
      ):
      print(foldername)
      plot_folder = process_results(solution_folder, models, plot_all_history)
      # models runtime and termination condition
      if os.path.exists(os.path.join(solution_folder, "runtime.csv")):
        runtime = load_runtime(solution_folder)
        # -- plot runtime
        runtime["time"] = runtime.index
        runtime.plot.scatter(
          x = "time",
          y = "runtime",
          marker = ".",
          grid = True,
          c = "color",
          s = 100,
          label = None
        )
        rt_avg = runtime["runtime"].mean()
        plt.axhline(
          y = rt_avg,
          color = mcolors.TABLEAU_COLORS["tab:orange"],
          linewidth = 2,
          label = f"Average: {rt_avg:.2f}s"
        )
        plt.xlabel("Control time period $t$", fontsize = 14)
        plt.ylabel("Time to solution [s]", fontsize = 14)
        plt.legend(fontsize = 14)
        plt.savefig(
          os.path.join(plot_folder, "runtime.png"),
          dpi = 300,
          format = "png",
          bbox_inches = "tight"
        )
        plt.close()
        runtime["exp"] = foldername
        all_runtimes = pd.concat([all_runtimes, runtime], ignore_index = True)
      # models objective function
      if os.path.exists(os.path.join(solution_folder, "obj.csv")):
        obj = pd.read_csv(
          os.path.join(solution_folder, "obj.csv")
        )
        obj.index = result_steps(solution_folder, len(obj))
        for key in ["LSP", "SP/coord"]:
          if key in obj:
            obj.rename(columns = {key: "FaaS-MACrO"}, inplace = True)
        colname = obj.loc[:,~obj.columns.str.startswith("Unnamed")].columns[0]
        obj.loc[:,~obj.columns.str.startswith("Unnamed")].plot(
          marker = ".",
          grid = True,
          color = mcolors.TABLEAU_COLORS["tab:green"]
        )
        plt.xlabel("Control time period $t$")
        plt.ylabel("Objective function value")
        plt.savefig(
          os.path.join(plot_folder, "obj.png"),
          dpi = 300,
          format = "png",
          bbox_inches = "tight"
        )
        plt.close()
        obj.rename(columns = {colname: "obj"}, inplace = True)
        obj["method"] = colname
        obj["time"] = obj.index
        obj["exp"] = foldername
        all_obj = pd.concat([all_obj, obj], ignore_index = True)
  # load experiments list (if available)
  base_postprocessing_folder = os.path.join(
    base_solution_folder, "postprocessing"
  )
  os.makedirs(base_postprocessing_folder, exist_ok = True)
  if os.path.exists(os.path.join(base_solution_folder, "experiments.json")):
    experiments = {}
    with open(
        os.path.join(base_solution_folder, "experiments.json"), "r"
      ) as ist:
      experiments = json.load(ist)
    # match experiment name with description
    exp_description_match = {}
    for exp, exp_description_tuple in zip(
        experiments["centralized"], experiments["experiments_list"]
      ):
      exp_description_match[os.path.basename(exp)] = {
        "Nn": int(exp_description_tuple[0]),
        "seed": int(exp_description_tuple[-1])
      }
    for exp, exp_description_tuple in zip(
        experiments["faas-macro"], experiments["experiments_list"]
      ):
      exp_description_match[os.path.basename(exp)] = {
        "Nn": int(exp_description_tuple[0]),
        "seed": int(exp_description_tuple[-1])
      }
    # add information to runtime/obj dictionaries
    all_obj["Nn"] = [
      exp_description_match[exp]["Nn"] for exp in all_obj["exp"]
    ]
    all_obj["seed"] = [
      exp_description_match[exp]["seed"] for exp in all_obj["exp"]
    ]
    all_runtimes["Nn"] = [
      exp_description_match[exp]["Nn"] for exp in all_runtimes["exp"]
    ]
    all_runtimes["seed"] = [
      exp_description_match[exp]["seed"] for exp in all_runtimes["exp"]
    ]
    # plot runtime
    if len(all_runtimes) > 0:
      runtime_obj_boxplot(
        all_runtimes, "runtime", base_postprocessing_folder, "runtime_box"
      )
      all_runtimes.to_csv(
        os.path.join(base_postprocessing_folder, "runtime.csv"), index = False
      )
    if len(all_obj) > 0:
      runtime_obj_boxplot(
        all_obj, "obj", base_postprocessing_folder, "obj_box"
      )
      all_obj.to_csv(
        os.path.join(base_postprocessing_folder, "obj.csv"), index = False
      )
