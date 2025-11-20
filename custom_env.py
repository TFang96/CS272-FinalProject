import numpy as np
from highway_env.envs.roundabout_env import RoundaboutEnv
from highway_env.vehicle.behavior import IDMVehicle

class RoundaboutYieldExitEnv(RoundaboutEnv):

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({
            # Episode/geometry
            "duration": 60, #episode is 60 seconds
            "action": {"type": "DiscreteMetaAction"},   # actions are discrete
            "lanes_count": 1,                           # 1 lane roundabout
            "vehicles_count": 15,                       # ~10–15 cars; choose max
            "aggressive_fraction": 0.3,                 # share of assertive drivers
            "target_exit": 2,                           # 0..3 for 4-way roundabout

            # Rewards from spec
            "collision_reward": -5.0, #collision
            "step_penalty": -0.01, #each step
            "approach_fast_penalty": -0.5, #approaching roundabout too fast
            "unsafe_dist_penalty": -0.3, #too close to other vehicles
            "successful_exit": +3.0, #reaching our target

            # Safety thresholds
            "approach_speed_limit": 10.0,               # m/s when near entry
            "approach_radius": 45.0,                    # meters from center counts as approach
            "min_time_headway": 1.2,                    # seconds
        })
        return cfg

    def _reset(self):
        super()._reset()
        self._succeeded = False

        # Make some traffic more aggressive (shorter headway / higher speed)
        for i, v in enumerate(self.road.vehicles):
            if i == 0:  # our own vehicle
                continue
            if np.random.rand() < self.config["aggressive_fraction"]:
                if isinstance(v, IDMVehicle):
                    v.target_speed *= 2.0
                    v.POLITENESS *= 0.2

    # ----- Reward shaping -----
    def _reward(self, action):
        r = self.config["step_penalty"]

        # Collision penalty (in addition to termination)
        if self.vehicle.crashed:
            r += self.config["collision_reward"]

        # Approaching roundabout too fast
        if self._on_approach_lane() and self.vehicle.speed > self.config["approach_speed_limit"]:
            r += self.config["approach_fast_penalty"]

        # Unsafe following: small time headway to the front vehicle in same lane
        lead = self.get_front_vehicle()
        if lead is not None:
            gap = np.linalg.norm(lead.position - self.vehicle.position) - self.vehicle.LENGTH
            v = max(self.vehicle.speed, 1e-3)
            if (gap / v) < self.config["min_time_headway"]:
                r += self.config["unsafe_dist_penalty"]

        # Success bonus (credited once when we exit at the target arm)
        if self._succeeded:
            r += self.config["successful_exit"]

        return r

    # ----- Termination/success -----
    def _is_terminal(self):
        done = super()._is_terminated()
        if not self.vehicle.crashed and self._exited_via_target():
            self._succeeded = True
            done = True
        return done

    # ----- Helpers -----
    def _on_approach_lane(self):
        """Heuristic: lanes that lead toward the circle are 'approach' lanes."""
        try:
            idx = self.vehicle.lane_index  # ('r', 'r_in', 0) style
            return isinstance(idx, tuple) and str(idx[1]).endswith("_in")
        except Exception:
            return True

    def _exited_via_target(self):
        """True if ego has exited the circle via the requested arm."""
        arm_map = {"r_out": 0, "s_out": 1, "t_out": 2, "u_out": 3}
        try:
            idx = self.vehicle.lane_index
            if isinstance(idx, tuple) and str(idx[1]).endswith("_out"):
                return arm_map.get(idx[1], -1) == int(self.config["target_exit"])
        except Exception:
            pass
        return False

    def get_front_vehicle(self):
        lane = self.vehicle.road.network.get_lane(self.vehicle.lane_index)
        s_ego, _ = lane.local_coordinates(self.vehicle.position)
        front, min_ds = None, float("inf")
        for v in self.vehicle.road.vehicles:
            if v is self.vehicle or getattr(v, "lane_index", None) != self.vehicle.lane_index:
                continue
            s_v, _ = lane.local_coordinates(v.position)
            ds = s_v - s_ego
            if ds > 0 and ds < min_ds:
                front, min_ds = v, ds
        return front
