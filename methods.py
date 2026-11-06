"""The six methods included in the paper; names are stable artifact identifiers."""
from functools import partial

import decentralized_auction
import decentralized_greedy_rr
import decentralized_potentialgame

HARMONIZED = {
  "FaaS-MASeA": partial(decentralized_auction.run, detailed_init=True),
  "FaaS-MASeA(r)": partial(decentralized_auction.run, selective_memory=True),
  "Greedy-RR": decentralized_greedy_rr.run,
}
SEQUENTIAL_METHODS = {"PG": partial(decentralized_potentialgame.run_pg_r, detailed_init=True, fixed_x=True)}
CENTRALIZED = {"centralized": {}, "selfish-centralized": {"selfish": True}}
