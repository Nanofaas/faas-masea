from utils.common import load_requests_traces, NpEncoder
from generators.load_generator import LoadGenerator
from utils.centralized import encode_solution
from postprocessing import load_solution

import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
import numpy as np
import json
import os


def count_requests(
    solution_folder: str, model_name: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  # count nodes and functions
  Nn, Nf = None, None
  with open(
      os.path.join(solution_folder, "base_instance_data.json"), "r"
    ) as istream:
    data = json.load(istream)
    Nn = int(data["None"]["Nn"]["None"])
    Nf = int(data["None"]["Nf"]["None"])
  # load solution
  solution, replicas, detailed_fwd_solution = [pd.DataFrame()] * 3
  if os.path.exists(
      os.path.join(solution_folder, f"{model_name}_solution.csv")
    ):
    solution, replicas, detailed_fwd_solution, _, _ = load_solution(
      solution_folder, model_name
    )
  # count local/fwd/rejected requests
  all_local = pd.DataFrame()
  all_sentrecv = pd.DataFrame()
  all_rej = pd.DataFrame()
  for t in detailed_fwd_solution.index:
    x, y, z, _, _ = encode_solution(
      Nn, Nf, solution, detailed_fwd_solution, replicas, t
    )
    local = {
      "all": []
    }
    sentrecv = {
      "sent": [],
      "recv": []
    }
    rej = {
      "all": []
    }
    for n in range(Nn):
      # -- total
      local["all"].append(x[n,:].sum())
      sentrecv["sent"].append(y[n,:,:].sum())
      sentrecv["recv"].append(y[:,n,:].sum())
      rej["all"].append(z[n,:].sum())
      # -- by function
      for f in range(Nf):
        if f"f{f}" not in local:
          local[f"f{f}"] = []
        local[f"f{f}"].append(x[n,f])
        if f"f{f}_sent" not in sentrecv:
          sentrecv[f"f{f}_sent"] = []
          sentrecv[f"f{f}_recv"] = []
        sentrecv[f"f{f}_sent"].append(y[n,:,f].sum())
        sentrecv[f"f{f}_recv"].append(y[:,n,f].sum())
        if f"f{f}" not in rej:
          rej[f"f{f}"] = []
        rej[f"f{f}"].append(z[n,f])
    # add info
    local = pd.DataFrame(local)
    local["node"] = range(Nn)
    local["t"] = t
    sentrecv = pd.DataFrame(sentrecv)
    sentrecv["node"] = range(Nn)
    sentrecv["t"] = t
    rej = pd.DataFrame(rej)
    rej["node"] = range(Nn)
    rej["t"] = t
    # merge
    all_local = pd.concat([all_local, local], ignore_index = True)
    all_sentrecv = pd.concat([all_sentrecv, sentrecv], ignore_index = True)
    all_rej = pd.concat([all_rej, rej], ignore_index = True)
  return all_local, all_sentrecv, all_rej


def filter_traces(
    all_sentrecv: pd.DataFrame, all_rej: pd.DataFrame
  ) -> Tuple[list, list]:
  only_local = []
  with_offloading = []
  with_reject = []
  sentrecv_sum = all_sentrecv.groupby("t")["sent"].sum()
  reject_sum = all_rej.groupby("t")["all"].sum()
  for t, s, r in zip(sentrecv_sum.index, sentrecv_sum, reject_sum):
    if s <= 0.0 and r <= 0.0:
      only_local.append(t)
    elif r > 0.0:
      with_reject.append(t)
    else:
      with_offloading.append(t)
  return only_local, with_offloading, with_reject


def plot_filtered_traces(
    solution_folder, only_local, with_offloading, with_reject
  ):
  traces, mt, Mt, ts  = load_requests_traces(solution_folder)
  offloading_times = np.asarray(with_offloading)
  for f, f_traces in traces.items():
    # f_traces_array = {
    #   a: np.array(tr)[with_offloading] for a, tr in f_traces.items()
    # }
    f_traces_array = {
      a: np.empty_like(np.array(tr)) for a, tr in f_traces.items()
    }
    for a in f_traces_array:
      v = np.array(f_traces[a])
      last_valid = np.maximum.accumulate(
        np.where(
          np.isin(np.arange(len(v)), offloading_times),
          np.arange(len(v)),
          -1
        )
      )
      f_traces_array[a] = v[last_valid]
    LoadGenerator.plot_input_load(
      f_traces_array,
      plot_filename = os.path.join(solution_folder, "load", f"f{f}_fwd.png")
    )


def main(solution_folder: str, model_name: str):
  all_local, all_sentrecv, all_rej = count_requests(
    solution_folder, model_name
  )
  input_load, mt, Mt, ts = load_requests_traces(solution_folder)
  if len(all_sentrecv) > 0:
    only_local, with_offloading, with_reject = filter_traces(
      all_sentrecv, all_rej
    )
    os.makedirs(os.path.join(solution_folder, "load"), exist_ok = True)
    with open(
        os.path.join(solution_folder, "load", "trace_filtered.json"), "w"
      ) as ost:
      ost.write(
        json.dumps(
          {
            "only_local": only_local,
            "with_offloading": with_offloading,
            "with_reject": with_reject
          },
          indent = 2
        )
      )
    # plot_filtered_traces(
    #   solution_folder, only_local, with_offloading, with_reject
    # )
    # plot all
    t = all_sentrecv["t"].iloc[0]
    all_sentrecv[all_sentrecv["t"] == t][["sent","recv"]].plot.bar(
      logy = True
    )
    plt.grid(which = "both", axis = "y")
    plt.savefig(
      os.path.join(solution_folder, f"sentrecv_t{t}.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
  # modify traces subtracting rejected requests
  norej_traces = {}
  for f, f_traces in input_load.items():
    norej_traces[f] = {}
    for n, n_trace in f_traces.items():
      norej_traces[f][n] = np.array(n_trace, dtype=float, copy=True)
      for t in all_rej["t"].unique():
        norej_traces[f][n][t] -= float(
          all_rej[
            (all_rej["t"] == t) & (all_rej["node"] == n)
          ][f"f{f}"].iloc[0]
        )
    if len(f_traces) <= 10:
      LoadGenerator.plot_input_load(
        norej_traces[f],
        0,
        os.path.join(solution_folder, "load", f"f{f}_norej.png")
      )
  with open(
      os.path.join(
        solution_folder, "load", f"norej_trace-{mt}_{Mt}_{ts}.json"
      ),
      "w"
    ) as ost:
    ost.write(
      json.dumps(norej_traces, indent = 2, cls = NpEncoder)
    )


if __name__ == "__main__":
  base_solution_folder = "solutions/Nf_3-model_variant-selfish_centralized"
  model_name = "SelfishLoadManagementModel"
  for dname in os.listdir(base_solution_folder):
    if os.path.isdir(os.path.join(base_solution_folder, dname)) and not (
        dname.startswith(".") or dname.startswith("postprocessing")
      ):
      print(dname)
      solution_folder = os.path.join(base_solution_folder, dname)
      main(solution_folder, model_name)
