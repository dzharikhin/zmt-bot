import numpy as np


def aggregate(per_segment_vectors: list[np.ndarray], strategy: str) -> np.ndarray:
    """Aggregate per-segment vectors into a single track vector.

    - "mean":    np.mean(stack, axis=0)       # dim = D
    - "meanstd": np.concatenate([mean, std])   # dim = 2D
    - "max":     np.max(stack, axis=0)         # dim = D

    When there is only one segment (e.g., "full" policy), "meanstd" returns
    [mean, zeros] since std is identically zero for a single sample.
    """
    if not per_segment_vectors:
        raise ValueError("per_segment_vectors must not be empty")

    stack = np.stack(per_segment_vectors)

    if strategy == "mean":
        return np.mean(stack, axis=0)
    elif strategy == "meanstd":
        mean = np.mean(stack, axis=0)
        std = (
            np.zeros_like(mean)
            if len(per_segment_vectors) == 1
            else np.std(stack, axis=0)
        )
        return np.concatenate([mean, std])
    elif strategy == "max":
        return np.max(stack, axis=0)
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy!r}")
