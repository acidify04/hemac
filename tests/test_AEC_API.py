"""Run_test test module."""
import pettingzoo.test as tests
from hemac import HeMAC_v0

my_env = HeMAC_v0.env()
env_func = HeMAC_v0
parallel_env = HeMAC_v0.parallel_env()


class TestHemac:
    """TestCoordinates."""

    def test_api(self):
        """Test api."""
        tests.api_test(my_env)

    def test_parallel_api(self):
        """Test parallel api."""
        tests.parallel_api_test(parallel_env, num_cycles=1000)

    def test_max_cycles(self):
        """Test max cycles."""
        tests.max_cycles_test(env_func)  # NOTE: this test will not pass if the episode is shorter than 4 steps, which sometimes happen by badluck.

    def test_render(self):
        """Test render."""
        tests.render_test(env_func.env)

    def test_save_obs(self):
        """Test save obs."""
        tests.test_save_obs(my_env)

    def test_measure_performance(self):
        """Test measure performance."""
        tests.performance_benchmark(my_env)

# tests.seed_test(env_func.env, num_cycles=10)  # TODO: use generator in reset() for deterministic env
