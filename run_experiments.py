"""
Script maestro de experimentos comparables con Dynamask
(Crabbé & van der Schaar, 2021).

Ejecuta:
  Experimento 1 — White-box sintético con importancia conocida
                  (análogo a Tablas 1 y 2 de Dynamask)
  Experimento 3 — Faithfulness en datos reales S&P 500
                  (análogo a Figuras 4 y 5 de Dynamask)

Uso
---
    python run_experiments.py           # ambos experimentos
    python run_experiments.py --exp1    # solo Experimento 1
    python run_experiments.py --exp3    # solo Experimento 3
    python run_experiments.py --fast    # versión rápida (pocas repeticiones)
"""
import sys
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import RESULTS_DIR
from experiments.exp1_whitebox import run_whitebox_experiments, print_results_table
from experiments.exp3_faithfulness import run_faithfulness_experiment
from visualization.plots import plot_exp1_barplot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1", action="store_true",
                        help="Ejecutar solo el Experimento 1")
    parser.add_argument("--exp3", action="store_true",
                        help="Ejecutar solo el Experimento 3")
    parser.add_argument("--fast", action="store_true",
                        help="Modo rápido: menos repeticiones y muestras")
    args = parser.parse_args()

    run_both = not args.exp1 and not args.exp3

    print("=" * 70)
    print("  EXPERIMENTOS COMPARABLES CON DYNAMASK (Crabbé et al., 2021)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Experimento 1
    # -----------------------------------------------------------------------
    if args.exp1 or run_both:
        print("\n[EXPERIMENTO 1] White-box sintético con importancia conocida")
        print("  (análogo a Tablas 1 y 2 de Dynamask)")

        t0 = time.time()
        if args.fast:
            exp1_results = run_whitebox_experiments(
                T=20, V=15, N=10, n_rep=3,
                n_salient_features=3, n_salient_times=4,
                n_svs_samples=10, verbose=True,
            )
        else:
            exp1_results = run_whitebox_experiments(
                T=20, V=15, N=20, n_rep=10,
                n_salient_features=3, n_salient_times=4,
                n_svs_samples=20, verbose=True,
            )

        print_results_table(exp1_results)
        elapsed = time.time() - t0
        print(f"\nExp. 1 completado en {elapsed:.1f}s")

        # Guardar JSON
        json_path = RESULTS_DIR / "exp1_whitebox_results.json"
        json_path.write_text(json.dumps(exp1_results, indent=2), encoding="utf-8")
        print(f"JSON: {json_path}")

        # Plots
        for scenario in ["rare_feature", "rare_time"]:
            plot_exp1_barplot(
                exp1_results,
                scenario=scenario,
                save_path=RESULTS_DIR / f"exp1_{scenario}_barplot.png"
            )
        print(f"Plots guardados en {RESULTS_DIR}/")

    # -----------------------------------------------------------------------
    # Experimento 3
    # -----------------------------------------------------------------------
    if args.exp3 or run_both:
        print("\n[EXPERIMENTO 3] Faithfulness en datos reales S&P 500")
        print("  (análogo a Figuras 4 y 5 de Dynamask)\n")

        t0 = time.time()
        exp3_curves = run_faithfulness_experiment(verbose=True)
        elapsed = time.time() - t0
        print(f"\nExp. 3 completado en {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("  Todos los resultados guardados en results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
