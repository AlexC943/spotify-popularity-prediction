"""End-to-end orchestrator.

Stages:
  1. EDA — figures + summary tables, including split balance.
  2. Classical models — mean baseline, Ridge, Lasso, RF (tuned), LightGBM.
  3. PyTorch MLP — validation sweep, single test eval on the winner.
  4. Ablation — refit LightGBM without `track_genre` to quantify the
     strict audio-only ceiling.
  5. Multi-seed — refit RF, LightGBM, MLP under three seeds and report
     the mean and standard deviation of the test metrics.
  6. Comparison + report — figures, importance plots, final report.
"""
from __future__ import annotations

from . import eda, train_classical, train_nn, ablation, multi_seed, compare


def main() -> None:
    print("=" * 60); print("STEP 1/6: EDA"); print("=" * 60)
    eda.run()

    print("=" * 60); print("STEP 2/6: classical models"); print("=" * 60)
    train_classical.main()

    print("=" * 60); print("STEP 3/6: PyTorch MLP"); print("=" * 60)
    train_nn.main()

    print("=" * 60); print("STEP 4/6: genre ablation"); print("=" * 60)
    ablation.main()

    print("=" * 60); print("STEP 5/6: multi-seed stability"); print("=" * 60)
    multi_seed.main()

    print("=" * 60); print("STEP 6/6: comparison + report"); print("=" * 60)
    compare.main()

    print("done.")


if __name__ == "__main__":
    main()
