"""Tests for relative agent positions in observations."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemac import HeMAC_v0


def test_observations_include_other_drone_and_observer_positions():
    """Observer and drone observations should include normalized peer positions."""
    env = HeMAC_v0.env(
        n_observers=1,
        n_drones=2,
        n_provisioners=0,
        min_obstacles=0,
        max_obstacles=0,
        render_mode=None,
    )

    try:
        env.reset(seed=7)

        observer_obs = env.observe("observer_0")["vector"]
        drone_obs = env.observe("drone_0")["vector"]

        norm = np.hypot(800.0, 800.0)

        observer_peer_slice = observer_obs[5:-2]
        expected_observer_peers = np.array(
            [
                0.0,
                50.0 / norm,
                -50.0 / norm,
                -50.0 / norm,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(observer_peer_slice, expected_observer_peers, atol=1e-6)

        drone_peer_slice = drone_obs[4:-2]
        expected_drone_peers = np.array(
            [
                0.0,
                -50.0 / norm,
                -50.0 / norm,
                -100.0 / norm,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(drone_peer_slice, expected_drone_peers, atol=1e-6)
    finally:
        env.close()
