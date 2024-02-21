import numpy as np
from pathlib import Path

from ..utils import read_pickle, write_pickle
from ..sensitivity_analysis import run_mc_simulations_wo_lowinf

GSA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "sensitivity-analysis"
DATA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "datapackages"
MC_BATCH_SIZE = 100


def run_mc_simulations_screening(project, fp_ecoinvent, factor, cutoff, max_calc, iterations, seed=42):
    """Run Monte Carlo simulations for high-dimensional screening."""
    fp = GSA_DIR / f"scores.wo_lowinf.{seed}.{iterations}.pickle"

    if fp.exists():
        scores = read_pickle(fp)

    else:
        scores = []

        starts = np.arange(0, iterations, MC_BATCH_SIZE)
        n_batches = len(starts)
        np.random.seed(seed)
        seeds = np.random.random_integers(100_000, 999_999, n_batches)

        for i in range(n_batches):
            print(f"MC simulations for screening -- random seed {i+1:2d} / {n_batches:2d} -- {seeds[i]}")
            scores_current = run_mc_simulations_wo_lowinf(
                project, fp_ecoinvent, factor, cutoff, max_calc, MC_BATCH_SIZE, seeds[i]
            )
            scores += scores_current

        write_pickle(scores, fp)

    return scores
