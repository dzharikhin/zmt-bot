import numpy as np
import pytest

from audio.aggregation import aggregate


class TestAggregateMean:
    def test_single_vector(self):
        v = np.array([1.0, 2.0, 3.0])
        result = aggregate([v], "mean")
        np.testing.assert_array_equal(result, v)

    def test_multiple_vectors(self):
        v1 = np.array([1.0, 2.0])
        v2 = np.array([3.0, 4.0])
        result = aggregate([v1, v2], "mean")
        np.testing.assert_array_almost_equal(result, [2.0, 3.0])

    def test_output_dim_equals_input_dim(self):
        vectors = [np.random.randn(50) for _ in range(5)]
        result = aggregate(vectors, "mean")
        assert result.shape == (50,)


class TestAggregateMeanStd:
    def test_single_vector_zeros_std(self):
        v = np.array([1.0, 2.0, 3.0])
        result = aggregate([v], "meanstd")
        assert result.shape == (6,)
        np.testing.assert_array_almost_equal(result[:3], v)
        np.testing.assert_array_almost_equal(result[3:], [0.0, 0.0, 0.0])

    def test_multiple_vectors_dim_2x(self):
        v1 = np.array([1.0, 2.0])
        v2 = np.array([3.0, 4.0])
        result = aggregate([v1, v2], "meanstd")
        assert result.shape == (4,)
        np.testing.assert_array_almost_equal(result[:2], [2.0, 3.0])

    def test_output_dim_is_twice_input_dim(self):
        vectors = [np.random.randn(50) for _ in range(5)]
        result = aggregate(vectors, "meanstd")
        assert result.shape == (100,)


class TestAggregateMax:
    def test_single_vector(self):
        v = np.array([1.0, 2.0, 3.0])
        result = aggregate([v], "max")
        np.testing.assert_array_equal(result, v)

    def test_multiple_vectors(self):
        v1 = np.array([1.0, 5.0])
        v2 = np.array([3.0, 2.0])
        result = aggregate([v1, v2], "max")
        np.testing.assert_array_almost_equal(result, [3.0, 5.0])

    def test_output_dim_equals_input_dim(self):
        vectors = [np.random.randn(50) for _ in range(5)]
        result = aggregate(vectors, "max")
        assert result.shape == (50,)


class TestAggregateEdgeCases:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            aggregate([], "mean")

    def test_unknown_strategy_raises(self):
        v = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            aggregate([v], "median")

    def test_full_policy_deduplication(self):
        vectors = [np.array([5.0, 10.0])]
        mean_result = aggregate(vectors, "mean")
        max_result = aggregate(vectors, "max")
        np.testing.assert_array_equal(mean_result, max_result)
