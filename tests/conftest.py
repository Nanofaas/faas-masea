import matplotlib
import sys
from pathlib import Path


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
