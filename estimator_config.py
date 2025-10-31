import pathlib
from collections import defaultdict
from itertools import groupby

import numpy as np

if __name__ == "__main__":

    PROPS = (
        "suod_combination",
        "feature_bagging_type",
        "feature_bagging_n_estimators",
        "lscp_n_bins",
        "lscp_local_region_size",
        "loda_n_random_cuts",
    )
    ROWS_TO_CHECK = (
        "('raw', 'aggregates', 'standard_scaling', 'no_pca', 'contamination_fraction=0.33')",
        "('raw', 'aggregates', 'standard_scaling', 'pca', 'contamination_fraction=0.33')",
        "('raw', 'standard_scaling', 'pca', 'contamination_fraction=0.33')",
    )
    raw_report_file = pathlib.Path("data_report.csv")

    def param_value_has_influence(param_variant_accuracies: dict):
        averages = [float(e[0]) for e in param_variant_accuracies.values()]
        stds = [float(e[1]) for e in param_variant_accuracies.values()]
        return (
            np.std(averages) / np.average(averages) >= 0.33
            or np.std(stds) / np.average(stds) >= 0.33
        )

    with raw_report_file.open("rt") as raw_report:
        checked_target_rows = set()
        for line_no, line in enumerate((l.strip() for l in raw_report), 1):
            tree = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            data_config, estimators = tuple(line.split(":", 2))
            if data_config.strip() not in ROWS_TO_CHECK:
                continue
            checked_target_rows.add(data_config.strip())
            for estimator in estimators.strip().split(";"):
                if not estimator:
                    continue
                estimator_config, estimator_result = tuple(estimator.split("->", 2))
                estimator_average, estimator_deviation = tuple(
                    estimator_result.split("+-", 2)
                )
                estimator_type, estimator_param_line = tuple(
                    estimator_config.split("[", 2)
                )
                estimator_params = dict(
                    [
                        tuple(param.split("=", 2))
                        for param in estimator_param_line.split("]", 2)[0].split(",")
                    ]
                )
                for target in PROPS:
                    if target not in estimator_params:
                        continue
                    target_reduced_params = estimator_params.copy()
                    target_param_value = target_reduced_params.pop(target)
                    tree[estimator_type][tuple(target_reduced_params.items())][target][
                        target_param_value
                    ] = (estimator_average, estimator_deviation)

            meaningful_variations = {
                ((estimator_type,), *reduced_params): param_target
                for estimator_type, estimator_type_variants in tree.items()
                for reduced_params, param_target in estimator_type_variants.items()
                for target, param_variations in param_target.items()
                if len(param_variations) > 1
                and param_value_has_influence(param_variations)
            }

            # print(f"{data_config.strip()=}:")
            # for k, v in meaningful_variations.items():
            #     print(f"{",".join("=".join(param) for param in k)}: {v}")
            # print()
            print(f"{data_config.strip()=}:")
            for prop, all_variations_for_prop in groupby(
                sorted(meaningful_variations.values(), key=lambda d: [*d.keys()][0]),
                lambda d: [*d.keys()][0],
            ):
                all_variations_for_prop = list(all_variations_for_prop)
                best_prop_by_average_counter = defaultdict(int)
                best_prop_by_max_counter = defaultdict(int)
                for variation in all_variations_for_prop:
                    best_by_average = list(
                        sorted(
                            variation[prop].items(),
                            key=lambda i: float(i[1][0]),
                            reverse=True,
                        )
                    )[0][0]
                    best_prop_by_average_counter[best_by_average] += 1
                    best_by_max = list(
                        sorted(
                            variation[prop].items(),
                            key=lambda i: float(i[1][0]) + float(i[1][1]),
                            reverse=True,
                        )
                    )[0][0]
                    best_prop_by_max_counter[best_by_max] += 1
                by_average_stat = ",".join(
                    [
                        f"{value}->{best_count}/{len(all_variations_for_prop)}({best_count / len(all_variations_for_prop):.3f})"
                        for value, best_count in sorted(
                            best_prop_by_average_counter.items(),
                            key=lambda i: i[1],
                            reverse=True,
                        )
                    ]
                )
                by_max_stat = ",".join(
                    [
                        f"{value}->{best_count}/{len(all_variations_for_prop)}({best_count / len(all_variations_for_prop):.3f})"
                        for value, best_count in sorted(
                            best_prop_by_max_counter.items(),
                            key=lambda i: i[1],
                            reverse=True,
                        )
                    ]
                )
                print(f"{prop}: {by_average_stat=};{by_max_stat=}")
            if checked_target_rows != set(ROWS_TO_CHECK):
                print()
