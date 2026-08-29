"""Fused decision-rule study: contrastive gate decisions from both one-class scores.

Motivation: the shipped exclude gate uses only the dislike-model score
(`DualOneClassModel.decide`: dislike.calibrated < threshold). The liked
model's score is computed at inference time but plays no role in exclusion.
The curation reference (lfr@0.9 ~= 0.209) shows the single-score percentile
rule cannot reach lfr@0.9 <= 0.20 even with perfectly clean markup — the
decision rule itself is a bottleneck, independent of the feature space.

This study evaluates score-level fusion on out-of-fold scores from the exact
gates_study CV protocol (5-fold x seeds, ship-candidate config):

  diff fusion:  exclude if dislike_cal - w * like_cal >= t(w)
                (threshold = percentile of the fused score on OOF disliked
                scores, mirroring the prod threshold calibration)
  and-rule:     exclude if dislike_cal >= t1 AND like_cal <= t2
                (2D percentile sweep, best lfr at disliked-recall >= target;
                grid-selected per seed — optimistic, reported alongside the
                single-parameter diff rule for scale)

Hard gate (owner): lfr@0.85 <= 0.20 AND dfa@0.775 <= 0.20. Stretch: lfr@0.9
<= 0.20. Soft advisory: dfa@0.775 <= 0.12 (fusion must not buy lfr by
tanking dfa back toward the cap).

Also reports a supervised logistic probe (offline diagnostic only, never
shipped) at the same operating points as an information-ceiling reference,
plus a shuffle placebo (like-scores permuted within each set) to separate
real pairing information from rank-calibration scale effects.
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from benchmark.gates_study import (
    ESSENTIA_DIMS,
    SHIPPED_OUTLIER_BUDGET,
    load_features,
    run_cv,
)

logger = logging.getLogger(__name__)

CORPUS_DIR_DEFAULT = (
    "benchmark/essentia-2.1b6.dev1438+profile-9d999a32d40d3078"
    "+panns-0dc499e40e9761ef+schema-0280c8457fb073d5_arm4380"
)

# Parity anchors: single-rule run_cv metrics from the P6-M1b arm4380 run
# (data/benchmark/fused_rule_study_arm4380.json; same corpus, same CV
# protocol, schema identical to the post-P6-B production schema). The
# single-score rule (w=0) must reproduce these before any fused row is
# trusted. Cells without an anchor entry get an annotate-only parity block.
PARITY_ANCHORS = {
    "prod_baseline": {
        "lfr_at_0.7": 0.24927953890489915,
        "lfr_at_0.8": 0.44716618635926997,
        "lfr_at_0.9": 0.6769932756964457,
        "dfa_at_0.775": 0.15665125834617358,
        "auc_exclude": 0.794519278162369,
    },
    "ship_candidate": {
        "lfr_at_0.7": 0.16042267050912584,
        "lfr_at_0.8": 0.2581652257444765,
        "lfr_at_0.9": 0.4721421709894333,
        "dfa_at_0.775": 0.08115048793014895,
        "auc_exclude": 0.839719978567485,
    },
}

# Outlier budget pinned at the shipped 0.07 (prod_fused) for every cell so
# the selection x w comparison is apples-to-apples; prod_baseline keeps the
# historic shipped budget for the parity anchor.
CELLS = {
    "prod_baseline": {
        "outlier_method": "prod_fused",
        "outlier_budget": SHIPPED_OUTLIER_BUDGET,
        "selection": "welch64",
    },
    "ship_candidate": {
        "outlier_method": "prod_fused",
        "outlier_budget": 0.07,
        "selection": "per:quota64/ridge_select64",
    },
    "per_welch64_ridge64": {
        "outlier_method": "prod_fused",
        "outlier_budget": 0.07,
        "selection": "per:welch64/ridge_select64",
    },
    "per_quota32_ridge64": {
        "outlier_method": "prod_fused",
        "outlier_budget": 0.07,
        "selection": "per:quota32/ridge_select64",
    },
    "quota64_shared": {
        "outlier_method": "prod_fused",
        "outlier_budget": 0.07,
        "selection": "quota64",
    },
}

LFR_TARGETS = (0.7, 0.8, 0.85, 0.9)
DFA_TARGETS = (0.775, 0.8)
W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
HARD_LFR_CAP = 0.20
HARD_DFA_CAP = 0.20
SOFT_DFA_CAP = 0.12
PROBE_C = 0.1

# AND-rule calibration grid. t1 = percentile of dislike_cal within the OOF
# disliked scores (s_dd); t2 = percentile of like_cal within the OOF disliked
# scores (s_ld). Exclude iff dislike_cal >= t1 AND like_cal <= t2.
AND_T1_PERCENTILES = (2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30)
AND_T2_PERCENTILES = (10, 20, 30, 40, 50, 60, 70, 80, 90)
PLACEBO_W_GRID = (0.5, 1.0)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    try:
        return float(roc_auc_score(y, np.concatenate([pos, neg])))
    except ValueError:
        return 0.5


def fused_metrics(scores: dict, w: float) -> dict:
    """Gate metrics for the diff-fusion decision rule with weight w.

    Exclude score of a point x: dislike_cal(x) - w * like_cal(x); threshold =
    percentile(100 * (1 - rt)) over the OOF fused scores of DISLIKED points
    (the fused analog of the prod threshold calibration). w=0 reduces exactly
    to the shipped single-score rule (same >= / > conventions as
    gates_study.compute_metrics).
    """
    s_ll = scores["like_on_liked"]
    s_ld = scores["like_on_disliked"]
    s_dd = scores["dislike_on_disliked"]
    s_dl = scores["dislike_on_liked"]

    f_dd = s_dd - w * s_ld
    f_dl = s_dl - w * s_ll
    f_ll = s_ll - w * s_dl
    f_ld = s_ld - w * s_dd

    metrics = {
        "auc_exclude": _auc(f_dd, f_dl),
        "auc_include": _auc(f_ll, f_ld),
    }
    for rt in LFR_TARGETS:
        thr = float(np.percentile(f_dd, 100 * (1 - rt)))
        metrics[f"lfr_at_{rt}"] = float(np.mean(f_dl >= thr))
    for rt in DFA_TARGETS:
        thr = float(np.percentile(f_ll, 100 * (1 - rt)))
        metrics[f"dfa_at_{rt}"] = float(np.mean(f_ld > thr))
    return metrics


def and_rule_metrics(
    scores: dict,
    recall_targets=LFR_TARGETS,
    t1_percentiles: tuple[float, ...] = AND_T1_PERCENTILES,
    t2_percentiles: tuple[float, ...] = AND_T2_PERCENTILES,
) -> dict:
    """Best AND-rule (t1 on dislike_cal, t2 on like_cal) per recall target.

    For each target the grid pair with disliked-recall >= target and minimal
    lfr is selected (per seed, on the same OOF arrays — optimistic; the diff
    rule carries the honest headline). Falls back to the loosest grid corner
    with its actual recall when no pair reaches the target.
    """
    s_ll = scores["like_on_liked"]
    s_ld = scores["like_on_disliked"]
    s_dd = scores["dislike_on_disliked"]
    s_dl = scores["dislike_on_liked"]

    results: dict = {}
    for rt in recall_targets:
        best = None
        for p1 in t1_percentiles:
            t1 = float(np.percentile(s_dd, p1))
            for p2 in t2_percentiles:
                t2 = float(np.percentile(s_ld, p2))
                recall = float(np.mean((s_dd >= t1) & (s_ld <= t2)))
                if recall >= rt:
                    lfr = float(np.mean((s_dl >= t1) & (s_ll <= t2)))
                    if best is None or lfr < best["lfr_at_target"]:
                        best = {
                            "lfr_at_target": lfr,
                            "t1_pct": p1,
                            "t2_pct": p2,
                            "recall": recall,
                        }
        if best is None:
            p1 = min(t1_percentiles)
            p2 = max(t2_percentiles)
            t1 = float(np.percentile(s_dd, p1))
            t2 = float(np.percentile(s_ld, p2))
            best = {
                "lfr_at_target": float(np.mean((s_dl >= t1) & (s_ll <= t2))),
                "t1_pct": p1,
                "t2_pct": p2,
                "recall": float(np.mean((s_dd >= t1) & (s_ld <= t2))),
            }
        results[f"lfr_at_{rt}"] = best["lfr_at_target"]
        results[f"t1_pct_at_{rt}"] = best["t1_pct"]
        results[f"t2_pct_at_{rt}"] = best["t2_pct"]
        results[f"recall_at_{rt}"] = best["recall"]
    return results


def probe_scores(
    X_liked: np.ndarray, X_disliked: np.ndarray, seeds: tuple[int, ...]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """OOF supervised logistic probe (offline diagnostic only, never shipped).

    Same fold structure as gates_study.run_cv. Returns per-seed tuples of
    (P(dislike) on disliked points, P(dislike) on liked points).
    """
    per_seed: list[tuple[np.ndarray, np.ndarray]] = []
    for seed in seeds:
        n_splits = min(5, len(X_liked), len(X_disliked))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        p_liked: list[float] = []
        p_disliked: list[float] = []
        for (l_tr, l_te), (d_tr, d_te) in zip(kf.split(X_liked), kf.split(X_disliked)):
            X_tr = np.concatenate([X_liked[l_tr], X_disliked[d_tr]])
            y_tr = np.concatenate([np.zeros(len(l_tr)), np.ones(len(d_tr))])
            scaler = StandardScaler().fit(X_tr)
            clf = LogisticRegression(C=PROBE_C, class_weight="balanced", max_iter=2000)
            clf.fit(scaler.transform(X_tr), y_tr)
            p_liked.extend(clf.predict_proba(scaler.transform(X_liked[l_te]))[:, 1])
            p_disliked.extend(
                clf.predict_proba(scaler.transform(X_disliked[d_te]))[:, 1]
            )
        per_seed.append((np.asarray(p_disliked), np.asarray(p_liked)))
    return per_seed


def probe_metrics(p_disliked: np.ndarray, p_liked: np.ndarray) -> dict:
    """Exclude-arm ceiling metrics for the supervised probe"""
    metrics = {"auc_exclude": _auc(p_disliked, p_liked)}
    for rt in LFR_TARGETS:
        thr = float(np.percentile(p_disliked, 100 * (1 - rt)))
        metrics[f"lfr_at_{rt}"] = float(np.mean(p_liked >= thr))
    return metrics


def placebo_metrics(scores: dict, w: float, rng: np.random.Generator) -> dict:
    """Shuffle placebo: permute like-scores within each set before fusing.

    Destroys the per-point pairing (victim liked-rank vs its dislike score)
    while preserving the marginals (liked like-ranks ~ U(0,1), disliked
    like-scores mostly clamped low). If the fused lfr stays at placebo-low
    levels, the fusion gain is a rank-calibration scale artifact rather than
    information carried by the like model.
    """
    shuffled = {
        "like_on_liked": rng.permutation(scores["like_on_liked"]),
        "like_on_disliked": rng.permutation(scores["like_on_disliked"]),
        "dislike_on_disliked": scores["dislike_on_disliked"],
        "dislike_on_liked": scores["dislike_on_liked"],
    }
    return fused_metrics(shuffled, w)


def score_diagnostics(scores: dict) -> dict:
    """Clamp asymmetry and pairing stats behind the fusion effect"""
    s_ll = scores["like_on_liked"]
    s_ld = scores["like_on_disliked"]
    s_dl = scores["dislike_on_liked"]
    return {
        "disliked_like_score_clamped_frac": float(np.mean(s_ld <= 1e-6)),
        "liked_dislike_score_clamped_frac": float(np.mean(s_dl <= 1e-6)),
        "liked_pairing_corr": float(np.corrcoef(s_dl, s_ll)[0, 1]),
        "liked_like_score_mean": float(np.mean(s_ll)),
        "disliked_like_score_mean": float(np.mean(s_ld)),
    }


def aggregate(per_seed: list[dict]) -> dict:
    """Mean and std across seeds for every key"""
    out: dict = {}
    for key in per_seed[0]:
        vals = np.array([m[key] for m in per_seed], dtype=float)
        out[key] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std())
    return out


def rule_verdict(m: dict) -> str:
    """hard gate: lfr@0.85 <= 0.20 AND dfa@0.775 <= 0.20; stretch adds 0.9"""
    if not (m["lfr_at_0.85"] <= HARD_LFR_CAP and m["dfa_at_0.775"] <= HARD_DFA_CAP):
        return "fail"
    return "stretch" if m["lfr_at_0.9"] <= HARD_LFR_CAP else "pass"


def check_parity(name: str, metrics: dict) -> dict:
    """Compare single-rule run_cv metrics against the anchor arm values.

    Cells without a PARITY_ANCHORS entry get an annotate-only block
    (ok=None) — they cannot fail parity, they simply have no reference.
    """
    if name not in PARITY_ANCHORS:
        return {"ok": None, "keys": {}, "note": "no parity anchor for this cell"}
    anchor = PARITY_ANCHORS[name]
    keys: dict = {}
    ok = True
    for key, ref in anchor.items():
        diff = abs(metrics[key] - ref)
        tol = max(0.01, 2 * metrics.get(f"{key}_std", 0.01))
        key_ok = diff <= tol
        ok = ok and key_ok
        keys[key] = {
            "study": metrics[key],
            "anchor": ref,
            "abs_diff": diff,
            "tolerance": tol,
            "ok": key_ok,
        }
    return {"ok": ok, "keys": keys}


def _fmt(m: dict) -> str:
    return (
        f"lfr@0.7/0.8/0.85/0.9 = {m['lfr_at_0.7']:.3f}/{m['lfr_at_0.8']:.3f}/"
        f"{m['lfr_at_0.85']:.3f}/{m['lfr_at_0.9']:.3f}  "
        f"dfa@0.775 = {m['dfa_at_0.775']:.3f}  auc_exc = {m['auc_exclude']:.3f}"
    )


def evaluate_cell(
    name: str,
    X_liked: np.ndarray,
    X_disliked: np.ndarray,
    seeds: tuple[int, ...],
    essentia_dims: int,
    skip_probe: bool,
) -> dict:
    cell = CELLS[name]
    start = time.time()
    res = run_cv(
        X_liked,
        X_disliked,
        outlier_method=cell["outlier_method"],
        outlier_budget=cell["outlier_budget"],
        selection=cell["selection"],
        seeds=seeds,
        essentia_dims=essentia_dims,
        return_scores=True,
    )
    if res is None:
        raise RuntimeError(f"cell {name}: degenerate CV (None from run_cv)")
    metrics, scores_per_seed = res["metrics"], res["scores"]
    logger.info(f"[{name}] run_cv done in {time.time() - start:.0f}s")

    parity = check_parity(name, metrics)
    logger.info(f"[{name}] parity vs arm4380 anchors: ok={parity['ok']}")
    if parity["ok"] is False:
        for key, row in parity["keys"].items():
            if not row["ok"]:
                logger.warning(
                    f"[{name}] parity MISS {key}: study={row['study']:.4f} "
                    f"anchor={row['anchor']:.4f} diff={row['abs_diff']:.4f} "
                    f"tol={row['tolerance']:.4f}"
                )

    diff_rows: dict = {}
    for w in W_GRID:
        row = aggregate([fused_metrics(s, w) for s in scores_per_seed])
        row["verdict"] = rule_verdict(row)
        row["dfa_soft_ok"] = row["dfa_at_0.775"] <= SOFT_DFA_CAP
        diff_rows[str(w)] = row
        logger.info(f"[{name}] diff fusion w={w}: {_fmt(row)} -> {row['verdict']}")

    and_row = aggregate([and_rule_metrics(s) for s in scores_per_seed])
    # AND rule leaves the include gate untouched: reuse the w=0 dfa values
    and_row["dfa_at_0.775"] = diff_rows["0.0"]["dfa_at_0.775"]
    and_row["dfa_at_0.775_std"] = diff_rows["0.0"]["dfa_at_0.775_std"]
    and_row["dfa_at_0.8"] = diff_rows["0.0"]["dfa_at_0.8"]
    and_row["auc_exclude"] = diff_rows["0.0"]["auc_exclude"]
    and_row["verdict"] = rule_verdict(and_row)
    and_row["dfa_soft_ok"] = and_row["dfa_at_0.775"] <= SOFT_DFA_CAP
    logger.info(
        f"[{name}] AND rule: lfr@0.85 = {and_row['lfr_at_0.85']:.3f} "
        f"(t1={and_row['t1_pct_at_0.85']:.1f}%, t2={and_row['t2_pct_at_0.85']:.1f}%) "
        f"lfr@0.9 = {and_row['lfr_at_0.9']:.3f} -> {and_row['verdict']}"
    )

    probe_row = None
    if not skip_probe:
        p_start = time.time()
        probe_row = aggregate(
            [
                probe_metrics(p_d, p_l)
                for p_d, p_l in probe_scores(X_liked, X_disliked, seeds)
            ]
        )
        logger.info(
            f"[{name}] probe ceiling ({time.time() - p_start:.0f}s): "
            f"lfr@0.85/0.9 = {probe_row['lfr_at_0.85']:.3f}/"
            f"{probe_row['lfr_at_0.9']:.3f}  auc = {probe_row['auc_exclude']:.3f}"
        )

    placebo_rng = np.random.default_rng(7)
    placebo_rows = {
        str(w): aggregate([placebo_metrics(s, w, placebo_rng) for s in scores_per_seed])
        for w in PLACEBO_W_GRID
    }
    for w, row in placebo_rows.items():
        logger.info(
            f"[{name}] PLACEBO w={w}: lfr@0.85/0.9 = "
            f"{row['lfr_at_0.85']:.3f}/{row['lfr_at_0.9']:.3f}  "
            f"auc = {row['auc_exclude']:.3f}"
        )

    diagnostics = aggregate([score_diagnostics(s) for s in scores_per_seed])
    logger.info(f"[{name}] score diagnostics: {diagnostics}")

    return {
        "config": cell,
        "parity": parity,
        "single_rule_metrics": metrics,
        "diff_fusion": diff_rows,
        "and_rule": and_row,
        "probe_ceiling": probe_row,
        "placebo": placebo_rows,
        "score_diagnostics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default=CORPUS_DIR_DEFAULT)
    parser.add_argument("--essentia-dims", type=int, default=4380)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["ship_candidate"],
        choices=sorted(CELLS),
    )
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--output", default="data/benchmark/fused_rule_study.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    seeds = tuple(args.seeds)

    X_liked, X_disliked = load_features(args.features_dir)
    logger.info(
        f"Loaded liked={len(X_liked)} disliked={len(X_disliked)} "
        f"dim={X_liked.shape[1]}"
    )

    cells = {
        name: evaluate_cell(
            name, X_liked, X_disliked, seeds, args.essentia_dims, args.skip_probe
        )
        for name in args.cells
    }

    # Parity anchors come from the anchor-arm corpus (essentia width ==
    # current schema): enforce them there — a failure means broken plumbing.
    # On narrower/wider arms single-rule drift is expected — annotate, don't
    # skip. Cells without an anchor entry cannot fail.
    anchor_run = args.essentia_dims == ESSENTIA_DIMS
    best = None
    for name, cell in cells.items():
        if anchor_run and cell["parity"]["ok"] is False:
            continue
        candidates = [
            (row["lfr_at_0.85"], row["lfr_at_0.9"], f"{name}/diff_w{w}", row)
            for w, row in cell["diff_fusion"].items()
        ]
        candidates.append(
            (
                cell["and_rule"]["lfr_at_0.85"],
                cell["and_rule"]["lfr_at_0.9"],
                f"{name}/and_rule",
                cell["and_rule"],
            )
        )
        for lfr85, lfr90, label, row in candidates:
            if row["verdict"] == "fail":
                continue
            if best is None or (lfr85, lfr90) < (best[0], best[1]):
                best = (
                    lfr85,
                    lfr90,
                    label,
                    row["verdict"],
                    anchor_run or cell["parity"]["ok"],
                )

    if best is None:
        summary = "no rule passes the hard gate (lfr@0.85 <= 0.20, dfa <= 0.20)"
    else:
        note = "" if best[4] else " [parity drift vs anchor-arm anchors]"
        summary = (
            f"best passing rule: {best[2]} lfr@0.85 = {best[0]:.3f}, "
            f"lfr@0.9 = {best[1]:.3f} ({best[3]}){note}"
        )
    logger.info(summary)

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features_dir": args.features_dir,
        "essentia_dims": args.essentia_dims,
        "seeds": list(seeds),
        "hard_gate": {
            "lfr_at_0.85_cap": HARD_LFR_CAP,
            "dfa_at_0.775_cap": HARD_DFA_CAP,
            "soft_dfa_cap": SOFT_DFA_CAP,
        },
        "w_grid": list(W_GRID),
        "and_grid": {
            "t1_percentiles": list(AND_T1_PERCENTILES),
            "t2_percentiles": list(AND_T2_PERCENTILES),
        },
        "cells": cells,
        "summary": summary,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
