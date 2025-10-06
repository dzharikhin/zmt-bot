import pathlib
from collections import defaultdict

if __name__ == "__main__":

    raw_report_file = pathlib.Path("data_report.csv")
    filtered_estimators = []
    with raw_report_file.open("rt") as raw_report:
        for line in (l.strip() for l in raw_report):
            data_config, estimators = tuple(line.split(":", 2))
            for estimator in estimators.strip().split(";"):
                if not estimator:
                    continue
                estimator_config, estimator_result = tuple(estimator.split("->", 2))
                estimator_average, estimator_deviation = tuple(
                    estimator_result.split("+-", 2)
                )
                if (
                    max_score := float(estimator_average) + float(estimator_deviation)
                ) > 0.6 and float(estimator_average) >= 0.55:
                    filtered_estimators.append(
                        (
                            max_score,
                            estimator_average,
                            estimator_deviation,
                            data_config.strip(),
                            estimator_config.strip(),
                        )
                    )

    count_by_data_config = defaultdict(int)
    count_by_estimator_config = defaultdict(int)
    for (
        max_score,
        estimator_average,
        estimator_deviation,
        data_config,
        estimator_config,
    ) in sorted(filtered_estimators, key=lambda r: r[0], reverse=True):
        count_by_data_config[data_config] += 1
        count_by_estimator_config[
            estimator_config.split("[", 2)[1].split("]", 2)[0]
        ] += 1
        print(
            f"{data_config};{estimator_config};{estimator_average};{estimator_deviation};{max_score:.3f}"
        )
    print()
    for data_cfg, count in sorted(
        count_by_data_config.items(), key=lambda t: t[1], reverse=True
    ):
        print(f"{data_cfg}:{count}")
    print()
    for estimator_cfg, count in sorted(
        count_by_estimator_config.items(), key=lambda t: (t[1], t[0]), reverse=True
    ):
        print(f"{estimator_cfg}:{count}")
