"""
Test helpers for worker tests. These functions must be at module level
to be picklable for spawn context multiprocessing.
"""

import multiprocessing as mp


def _worker_times_two(q, value):
    result = value * 2
    q.put(result)


def _worker_error(q, task_id):
    try:
        raise ValueError("test error")
    except Exception as e:
        q.put((task_id, False, str(e)))


def _worker_partitioned(q, tasks):
    for task in tasks:
        try:
            result = task * 2
            q.put((task, True, None))
        except Exception as e:
            q.put((task, False, str(e)))


def run_test_executor():
    executor = __import__("config").get_training_executor()

    futures = [executor.submit(_worker_times_two_helper, i) for i in range(4)]
    results = sorted([f.result() for f in futures])

    assert results == [0, 2, 4, 6], f"Expected [0, 2, 4, 6], got {results}"
    return True


def _worker_times_two_helper(x):
    return x * 2


def run_test_spawn_single():
    q = mp.Queue()
    spawn = mp.get_context("spawn")
    p = spawn.Process(target=_worker_times_two, args=(q, 5))
    p.start()
    result = q.get()
    p.join(timeout=5)

    assert result == 10, f"Expected 10, got {result}"
    return True


def run_test_spawn_multiple():
    q = mp.Queue()
    spawn = mp.get_context("spawn")

    processes = []
    for i in range(3):
        p = spawn.Process(target=_worker_times_two, args=(q, i))
        p.start()
        processes.append(p)

    results = sorted([q.get() for _ in range(3)])

    for p in processes:
        p.join(timeout=5)

    assert results == [0, 2, 4], f"Expected [0, 2, 4], got {results}"
    return True


def run_test_spawn_error():
    q = mp.Queue()
    spawn = mp.get_context("spawn")
    p = spawn.Process(target=_worker_error, args=(q, "task1"))
    p.start()
    task_id, success, error_msg = q.get()
    p.join(timeout=5)

    assert task_id == "task1"
    assert success is False
    assert "test error" in error_msg
    return True


def run_test_worker_pattern():
    q = mp.Queue()
    tasks = list(range(10))
    n_workers = 2
    partitions = [tasks[i::n_workers] for i in range(n_workers)]

    spawn = mp.get_context("spawn")
    processes = []

    for i in range(n_workers):
        p = spawn.Process(target=_worker_partitioned, args=(q, partitions[i]))
        p.start()
        processes.append(p)

    results = sorted([q.get() for _ in range(len(tasks))])

    for p in processes:
        p.join(timeout=5)

    expected = [(i, True, None) for i in range(10)]
    assert results == expected, f"Expected {expected}, got {results}"
    return True


def run_test_worker_pattern_failed():
    q = mp.Queue()
    tasks = [("ok1", True), ("bad", False), ("ok2", True)]

    spawn = mp.get_context("spawn")
    p = spawn.Process(target=_worker_partitioned_error, args=(q, tasks))
    p.start()

    results = sorted([q.get() for _ in range(len(tasks))])
    p.join(timeout=5)

    results_map = {r[0]: (r[1], r[2]) for r in results}
    assert results_map["ok1"] == (True, None)
    assert results_map["bad"] == (False, "bad task")
    assert results_map["ok2"] == (True, None)
    return True


def _worker_partitioned_error(q, tasks):
    for task_id, should_succeed in tasks:
        try:
            if should_succeed:
                q.put((task_id, True, None))
            else:
                raise ValueError("bad task")
        except Exception as e:
            q.put((task_id, False, str(e)))


def run_test_empty_tasks():
    q = mp.Queue()
    spawn = mp.get_context("spawn")

    p = spawn.Process(target=_worker_empty_helper, args=(q,))
    p.start()

    result = q.get()
    p.join(timeout=5)

    assert result == ("empty", True, None)
    return True


def _worker_empty_helper(q):
    q.put(("empty", True, None))


def run_test_path_resolution():
    import config

    q = mp.Queue()
    spawn = mp.get_context("spawn")

    p = spawn.Process(
        target=_worker_path_check,
        args=(
            q,
            str(config.data_path),
            str(config.local_data_path),
        ),
    )
    p.start()

    result = q.get()
    p.join(timeout=5)

    assert result["data_path_is_absolute"] is True
    assert result["local_data_path_is_absolute"] is True
    assert result["cwd_matches_parent"] is True
    assert result["arg_path_survives"] is True

    return True


def _worker_path_check(q, parent_data_path_str, parent_local_data_path_str):
    import pathlib

    import config

    result = {
        "data_path_is_absolute": config.data_path.is_absolute(),
        "local_data_path_is_absolute": config.local_data_path.is_absolute(),
        "cwd_matches_parent": str(pathlib.Path("data").resolve())
        == parent_data_path_str,
        "arg_path_survives": parent_data_path_str == str(config.data_path),
    }
    q.put(result)
