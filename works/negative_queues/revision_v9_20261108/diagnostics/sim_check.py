"""Parallel independent replications with the v7 event simulator."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import t as student
from works.negative_queues.revision_v7_20261107.simulation import run
def main():
    pass
if __name__ == "__main__":
  case = json.loads(sys.argv[1]); reps = int(sys.argv[2]); sample = int(sys.argv[3]); seed0 = int(sys.argv[4])
  with ProcessPoolExecutor(4) as ex:
      res = list(ex.map(run, [case]*reps, [seed0 + i for i in range(reps)], [sample]*reps, [sample//10]*reps))
  summary = {}
  for k in ("w", "v", "sw", "sv", "q"):
      x = np.array([r[k] for r in res]); summary[k] = {"mean": x.mean(), "half_ci95": student.ppf(.975, reps-1)*x.std(ddof=1)/np.sqrt(reps)}
  print(json.dumps(summary)); json.dump({"case": case, "summary": summary, "runs": res}, open(sys.argv[5], "w"))
