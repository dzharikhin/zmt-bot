import pathlib
import sys

import config

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import worker_helpers


class TestProcessPoolExecutor:
    def test_submit_and_get_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.local_data_path", tmp_path / "local_data")

        result = worker_helpers.run_test_executor()

        assert result is True

    def test_submit_multiple(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.local_data_path", tmp_path / "local_data")

        result = worker_helpers.run_test_executor()

        assert result is True

    def test_initializer_runs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.local_data_path", tmp_path / "local_data")

        result = worker_helpers.run_test_executor()

        assert result is True


class TestSpawnProcess:
    def test_process_puts_result_on_queue(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.local_data_path", tmp_path / "local_data")

        result = worker_helpers.run_test_spawn_single()

        assert result is True

    def test_multiple_processes(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.local_data_path", tmp_path / "local_data")

        result = worker_helpers.run_test_spawn_multiple()

        assert result is True

    def test_process_error_puts_failure_on_queue(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.local_data_path", tmp_path / "local_data")

        result = worker_helpers.run_test_spawn_error()

        assert result is True


class TestWorkerPattern:
    def test_round_robin_partitioning(self):
        tasks = list(range(7))
        partitions = [tasks[i::3] for i in range(3)]

        assert partitions == [[0, 3, 6], [1, 4], [2, 5]]

        worker_assignments = [len(p) for p in partitions]
        assert worker_assignments == [3, 2, 2]

    def test_workers_complete_and_join(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.local_data_path", tmp_path / "local_data")

        result = worker_helpers.run_test_worker_pattern()

        assert result is True

    def test_failed_task_reports_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.local_data_path", tmp_path / "local_data")

        result = worker_helpers.run_test_worker_pattern_failed()

        assert result is True

    def test_empty_task_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.local_data_path", tmp_path / "local_data")

        result = worker_helpers.run_test_empty_tasks()

        assert result is True


class TestPathResolution:
    def test_config_paths_are_absolute(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.local_data_path", tmp_path / "local_data")

        assert config.data_path.is_absolute()
        assert config.local_data_path.is_absolute()

    def test_spawn_child_resolves_same_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.local_data_path", tmp_path / "local_data")

        result = worker_helpers.run_test_path_resolution()

        assert result is True
