import numpy as np
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.objects import Obstacle

class HighwayObstaclesEnv(HighwayEnv):
    """Highway env with extra static obstacles on chosen lanes."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({
            "obstacle_positions": [60.0, 120.0, 180.0],  # longitudinal s (m) along the lane center
            "obstacle_lanes": [1, 2],                    # which lane indices to place obstacles on
            "obstacle_size": (5.0, 2.0),                 # (length, width) in meters
            # rewards/penalties
            "collision_reward": -25.0,
            "high_speed_reward": 5.0,
            "lane_change_reward": 2.0,
            "offroad_terminal": True,
        })
        return cfg

    def _reset(self):
        # Let HighwayEnv build its normal road & traffic
        super()._reset()

        length, width = self.config["obstacle_size"]

        # Place obstacles at given longitudinal positions and lane indices
        for lane_index in self.config["obstacle_lanes"]:
            lane = self.road.network.get_lane(("a", "b", lane_index))  # default HighwayEnv lanes
            for s in self.config["obstacle_positions"]:
                # Convert curvilinear (s, lateral=0) to world (x, y) on that lane
                x, y = lane.position(s, 0.0)
                # Add static obstacle to the road
                self.road.objects.append(
                    Obstacle(self.road, position=np.array([x, y]), heading=lane.heading_at(s),
                             length=length, width=width)
                )
