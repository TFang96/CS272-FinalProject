from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle

class AggressiveCar(IDMVehicle):
    LENGTH = 4
    WIDTH = 2
    MAX_SPEED = 30.0
    MAX_ACCELERATION = 4.0
    MIN_ACCELERATION = -8.0

class Truck(IDMVehicle):
    LENGTH = 7.0
    WIDTH = 3
    MAX_SPEED = 20.0
    MAX_ACCELERATION = 2.0
    MIN_ACCELERATION = -6.0

class Motorcycle(IDMVehicle):
    LENGTH = 2.2
    WIDTH = 1.5
    MAX_SPEED = 35.0
    MAX_ACCELERATION = 6.0
    MIN_ACCELERATION = -10.0



class CustomRoundaboutEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "absolute": True,
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-15, 15],
                        "vy": [-15, 15],
                    },
                },
                "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 8, 16]},
                "incoming_vehicle_destination": None,
                "collision_reward": -1,
                "high_speed_reward": 0.2,
                "right_lane_reward": 0,
                "lane_change_reward": -0.05,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "duration": 11,
                "normalize_reward": True,
            }
        )
        return config

    def _reward(self, action: int) -> float:
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * r for name, r in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["high_speed_reward"]],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: int) -> dict[str, float]:
        return {
            "collision_reward": self.vehicle.crashed,
            "high_speed_reward": MDPVehicle.get_speed_index(self.vehicle)
            / (MDPVehicle.DEFAULT_TARGET_SPEEDS.size - 1),
            "lane_change_reward": action in [0, 2],
            "on_road_reward": self.vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        center = [0, 0]
        radius = 20
        alpha = 24

        net = RoadNetwork()
        radii = [radius, radius + 4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]

        for lane in [0, 1]:
            net.add_lane("se", "ex", CircularLane(center, radii[lane], np.deg2rad(90-alpha), np.deg2rad(alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("ex", "ee", CircularLane(center, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("ee", "nx", CircularLane(center, radii[lane], np.deg2rad(-alpha), np.deg2rad(-90+alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("nx", "ne", CircularLane(center, radii[lane], np.deg2rad(-90+alpha), np.deg2rad(-90-alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("ne", "wx", CircularLane(center, radii[lane], np.deg2rad(-90-alpha), np.deg2rad(-180+alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("wx", "we", CircularLane(center, radii[lane], np.deg2rad(-180+alpha), np.deg2rad(-180-alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("we", "sx", CircularLane(center, radii[lane], np.deg2rad(180-alpha), np.deg2rad(90+alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("sx", "se", CircularLane(center, radii[lane], np.deg2rad(90+alpha), np.deg2rad(90-alpha), clockwise=False, line_types=line[lane]))

        # Access lanes: (r)oad/(s)ine
        access = 170
        dev = 85
        a = 5
        delta_st = 0.2*dev
        delta_en = dev - delta_st
        w = 2*np.pi/dev

        net.add_lane("ser", "ses", StraightLane([2,access],[2,dev/2],line_types=(s,c)))
        net.add_lane("ses","se", SineLane([2+a,dev/2],[2+a,dev/2-delta_st],a,w,-np.pi/2,line_types=(c,c)))
        net.add_lane("sx","sxs",SineLane([-2 - a, -dev / 2 + delta_en],[-2 - a, dev / 2],a,w,-np.pi / 2 + w * delta_en,line_types=(c, c)))
        net.add_lane("sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c)))
        net.add_lane("eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c)))
        net.add_lane("ees","ee",SineLane([dev / 2, -2 - a],[dev / 2 - delta_st, -2 - a],a,w,-np.pi / 2,line_types=(c, c)))
        net.add_lane("ex","exs",SineLane([-dev / 2 + delta_en, 2 + a],[dev / 2, 2 + a],a,w,-np.pi / 2 + w * delta_en,line_types=(c, c)))
        net.add_lane("exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c)))
        net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c)))
        net.add_lane("nes","ne",SineLane([-2 - a, -dev / 2],[-2 - a, -dev / 2 + delta_st],a,w,-np.pi / 2,line_types=(c, c)))
        net.add_lane("nx","nxs",SineLane([2 + a, dev / 2 - delta_en],[2 + a, -dev / 2],a,w,-np.pi / 2 + w * delta_en,line_types=(c, c)))
        net.add_lane("nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c)))
        net.add_lane("wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c)))
        net.add_lane("wes","we",SineLane([-dev / 2, 2 + a],[-dev / 2 + delta_st, 2 + a],a,w,-np.pi / 2,line_types=(c, c)))
        net.add_lane("wx","wxs",SineLane([dev / 2 - delta_en, -2 - a],[-dev / 2, -2 - a],a,w,-np.pi / 2 + w * delta_en,line_types=(c, c)))
        net.add_lane("wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c)))

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        position_deviation = 200.0
        speed_deviation = 200.0

        # Ego-vehicle
        ego_lane = self.road.network.get_lane(("ser","ses",0))
        ego_vehicle = self.action_type.vehicle_class(self.road, ego_lane.position(125.0,0.0), speed=8.0, heading=ego_lane.heading_at(140.0))
        try:
            ego_vehicle.plan_route_to("nxs")
        except AttributeError:
            pass
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # Incoming vehicles
        destinations = ["exr", "sxr", "nxr"]
        vehicle_types = [AggressiveCar, Truck, Motorcycle]

        # spacing positions for 3 cars
        indices = [-1, 0, 1]    

        for i, VehicleClass in zip(indices, vehicle_types):
            longitudinal = 20.0 * float(i) + self.np_random.normal() * position_deviation

            vehicle = VehicleClass.make_on_lane(
                self.road,
                ("we", "sx", 1),
                longitudinal=longitudinal,
                speed=16.0 + self.np_random.normal() * speed_deviation,
            )

            # assign route
            if self.config["incoming_vehicle_destination"] is not None:
                destination = destinations[self.config["incoming_vehicle_destination"]]
            else:
                destination = self.np_random.choice(destinations)

            vehicle.plan_route_to(destination)
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # Other vehicles
        destinations = ["exr", "sxr", "nxr"]
        vehicle_types = [AggressiveCar, Truck, Motorcycle]

        # spacing positions for 3 cars
        indices = [-1, 0, 1]    

        for i, VehicleClass in zip(indices, vehicle_types):
            longitudinal = 20.0 * float(i) + self.np_random.normal() * position_deviation

            vehicle = VehicleClass.make_on_lane(
                self.road,
                ("we", "sx", 0),
                longitudinal=longitudinal,
                speed=16.0 + self.np_random.normal() * speed_deviation,
            )

            # assign route
            if self.config["incoming_vehicle_destination"] is not None:
                destination = destinations[self.config["incoming_vehicle_destination"]]
            else:
                destination = self.np_random.choice(destinations)

            vehicle.plan_route_to(destination)
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)