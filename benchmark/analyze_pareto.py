import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GATE = {"auc_include": 0.89, "disliked_false_accept": 0.15, "auc_exclude": 0.78}


def load_trials(report_paths, variant):
    trials = []
    for report_path in report_paths:
        if not Path(report_path).exists():
            continue
        with open(report_path) as f:
            report = json.load(f)
        weights = report.get("objective_weights", {})
        for result in report.get("results", []):
            if result.get("failed") or result.get("name") != variant:
                continue
            optimization = result.get("optimization")
            if not optimization:
                continue
            for trial in optimization.get("trial_history", []):
                if trial.get("value") is None:
                    continue
                trials.append(
                    {
                        "source": Path(report_path).stem,
                        "weights": weights,
                        **trial,
                    }
                )
    return trials


def gate_margin(trial):
    return (
        trial["auc_include"]
        - GATE["auc_include"]
        + (GATE["disliked_false_accept"] - trial["disliked_false_accept"])
        + trial["auc_exclude"]
        - GATE["auc_exclude"]
    )


def passes_gate(trial):
    return (
        trial["auc_include"] >= GATE["auc_include"]
        and trial["disliked_false_accept"] <= GATE["disliked_false_accept"]
        and trial["auc_exclude"] >= GATE["auc_exclude"]
    )


def pareto_front(trials):
    front = []
    for t in trials:
        dominated = any(
            (
                o["auc_include"] >= t["auc_include"]
                and o["auc_exclude"] >= t["auc_exclude"]
            )
            and (
                o["auc_include"] > t["auc_include"]
                or o["auc_exclude"] > t["auc_exclude"]
            )
            for o in trials
        )
        if not dominated:
            front.append(t)
    return sorted(front, key=lambda t: -t["auc_include"])


def per_source_summary(trials):
    summary = {}
    for t in trials:
        summary.setdefault(t["source"], []).append(t)
    result = {}
    for source, source_trials in summary.items():
        top = sorted(source_trials, key=lambda t: -(t["value"] or 0.0))[:5]
        result[source] = {
            "n_trials": len(source_trials),
            "top5_median": {
                key: sorted(t[key] for t in top)[len(top) // 2]
                for key in (
                    "auc_include",
                    "auc_exclude",
                    "disliked_false_accept",
                    "liked_false_reject",
                )
            },
            "best_by_value": max(source_trials, key=lambda t: t["value"] or 0.0),
        }
    return result


def trial_row(t, prefix=""):
    return (
        f"{prefix}{t['source']:24s} inc={t['auc_include']:.4f} "
        f"exc={t['auc_exclude']:.4f} dFA={t['disliked_false_accept']:.4f} "
        f"fr={t['liked_false_reject']:.4f} "
        f"knn={t['params']['knn_k_min']}-{t['params']['knn_k_max']}"
        f"/{t['params']['knn_k_scale']:.3f} "
        f"gmm={t['params']['gmm_components_max']}"
        f"/{t['params']['gmm_min_points_per_component']} "
        f"out={t['params']['outlier_threshold']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pool benchmark trials, apply ship gate, report Pareto front"
    )
    parser.add_argument("--reports", type=str, nargs="+", required=True)
    parser.add_argument("--variant", type=str, default="full")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()

    trials = load_trials(args.reports, args.variant)
    if not trials:
        raise SystemExit(f"No trials found for variant {args.variant!r}")

    gate_passers = sorted(
        [t for t in trials if passes_gate(t)], key=gate_margin, reverse=True
    )
    front = pareto_front(trials)
    summary = per_source_summary(trials)

    print(
        f"Pooled {len(trials)} trials for variant {args.variant!r} "
        f"from {len(summary)} runs"
    )
    print(
        f"Gate: auc_include>={GATE['auc_include']}, "
        f"dFA<={GATE['disliked_false_accept']}, "
        f"auc_exclude>={GATE['auc_exclude']}"
    )
    print(f"Gate passers: {len(gate_passers)}/{len(trials)}")
    for t in gate_passers[:10]:
        print(trial_row(t, "  PASS "))
    if not gate_passers:
        near = sorted(trials, key=gate_margin, reverse=True)[:10]
        print("No passers; nearest trials by gate margin:")
        for t in near:
            print(trial_row(t, "  NEAR "))

    print("Pareto front (auc_include vs auc_exclude):")
    for t in front[:15]:
        print(trial_row(t, "  FRONT"))

    print("Per-run top5-median:")
    for source, stats in sorted(summary.items()):
        med = stats["top5_median"]
        print(
            f"  {source:24s} n={stats['n_trials']:3d} "
            f"inc={med['auc_include']:.4f} exc={med['auc_exclude']:.4f} "
            f"dFA={med['disliked_false_accept']:.4f}"
        )

    if args.output:
        payload = {
            "gate": GATE,
            "variant": args.variant,
            "n_trials": len(trials),
            "n_gate_passers": len(gate_passers),
            "gate_passers": gate_passers,
            "pareto_front": front,
            "per_source": {
                source: {
                    "n_trials": stats["n_trials"],
                    "top5_median": stats["top5_median"],
                }
                for source, stats in summary.items()
            },
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved analysis to {args.output}")

    if args.plot:
        fig, ax = plt.subplots(figsize=(10, 7))
        sources = sorted({t["source"] for t in trials})
        for source in sources:
            subset = [t for t in trials if t["source"] == source]
            ax.scatter(
                [t["auc_include"] for t in subset],
                [t["auc_exclude"] for t in subset],
                s=[t["disliked_false_accept"] * 600 for t in subset],
                alpha=0.6,
                label=source,
            )
        if gate_passers:
            ax.scatter(
                [t["auc_include"] for t in gate_passers],
                [t["auc_exclude"] for t in gate_passers],
                marker="*",
                s=200,
                color="red",
                label="gate passers",
                zorder=5,
            )
        ax.axvline(GATE["auc_include"], color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(GATE["auc_exclude"], color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("auc_include")
        ax.set_ylabel("auc_exclude")
        ax.set_title(f"{args.variant}: pooled trials (size = disliked_false_accept)")
        ax.legend(fontsize=7)
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
