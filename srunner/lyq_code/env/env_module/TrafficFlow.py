"""
Modularization for RL training env.

Parameters are designed for Junction scenario in Town03

"""

from __future__ import print_function

import glob
import os
import sys

sys.path.append("/home/lyq/CARLA_simulator/CARLA_098/PythonAPI/carla")
sys.path.append("/home/lyq/CARLA_simulator/CARLA_098/PythonAPI/carla/agents")
carla_path = '/home/lyq/CARLA_simulator/CARLA_098/PythonAPI'

try:
    sys.path.append(glob.glob(carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import time
import numpy as np
import math
import random
import traceback
import datetime

# carla module
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import vector

# scenario_runner module
from srunner.tools.scenario_helper import generate_target_waypoint
from srunner.challenge.utils.route_manipulation import interpolate_trajectory

# self defined module

from srunner.lyq_code.env.BasicEnv import BasicEnv

# from srunner.util_development.util import generate_route
from srunner.util_development.carla_rgb_color import *
from srunner.util_development.util_visualization import *
from srunner.util_development.scenario_helper_modified import *
from srunner.util_development.util_junction import (TransMatrix_yaw,
                                                    plot_local_coordinate_frame)

# common carla color
red = carla.Color(r=255, g=0, b=0)
green = carla.Color(r=0, g=255, b=0)
blue = carla.Color(r=0, g=0, b=255)
yellow = carla.Color(r=255, g=255, b=0)
magenta = carla.Color(r=255, g=0, b=255)
yan = carla.Color(r=0, g=255, b=255)
orange = carla.Color(r=255, g=162, b=0)
white = carla.Color(r=255, g=255, b=255)

start_location = carla.Location(x=53.0, y=128.0, z=3.0)
start_rotation = carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)  # standard transform of this junction
start_transform = carla.Transform(start_location, start_rotation)
# junction center
junction_center = carla.Location(x=-1.32, y=132.69, z=0.00)


class TrafficFlow:
    """
    Generate and manage a multi-direction continuous traffic flow in RL training environment.

    todo: get_junction, get_npc_spawn_point is in developing

    """

    # ==================================================
    # dict to record npc vehicle info
    npc_info = {
        'left': {
            'transform': carla.Transform(carla.Location(x=5.767616, y=175.509048, z=0.000000),
                                         carla.Rotation(pitch=0.0, yaw=269.637451, roll=0.0)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'count': 0,
            'actor_list': [],
            'nearby_npc': [],  # use queue to manage
        },
        'right': {
            'transform': carla.Transform(carla.Location(x=-6.268355, y=90.840492, z=0.000000),
                                         carla.Rotation(pitch=0.0, yaw=89.637459, roll=0.0)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'count': 0,
            'actor_list': [],
            'nearby_npc': [],
        },
        'straight': {
            'transform': carla.Transform(carla.Location(x=-46.925552, y=135.031494, z=0.000000),
                                         carla.Rotation(pitch=0.0, yaw=-1.296802, roll=0.0)),
            'last_spawn_time': 0,
            'target_spawn_time': None,
            'count': 0,
            'actor_list': [],
            'nearby_npc': [],
        }
    }
    # ==================================================

    def __init__(self, client, world):
        """
        Use necessary carla API from env to initialize.

        """
        # basic API
        self.client = client
        self.world = world
        # common API
        self.map = self.world.get_map()
        self.debug = self.world.debug  # world debug for plot
        self.blueprint_library = self.world.get_blueprint_library()  # blueprint
        self.traffic_manager = self.client.get_trafficmanager()
        # all npc vehicles spawned by traffic flow
        self.npc_vehicles = []  # using list to manage npc vehicle
        self.count = 0  # total amount of spawned npc vehicles
        self.ego_vehicle = None  # ego vehicle in env

    def get_npc_list(self):
        """
        Get npc list spawned by traffic flow module
        """
        return self.npc_vehicles

    def set_ego_vehicle(self, ego_vehicle):
        """
        Get ego vehicle from env for traffic flow module
        """
        self.ego_vehicle = ego_vehicle

    def draw_waypoint(self, transform, color=[red, green]):
        """
        Draw a point determined by transform(or waypoint).
        A spot to mark location
        An arrow to x-axis direction vector

        :param transform: carla.Transform or carla.Waypoint
        :param color: color of arrow and spot
        """
        if isinstance(transform, carla.Waypoint):
            transform = transform.transform
        scalar = 1.5
        yaw = np.deg2rad(transform.rotation.yaw)
        vector = scalar * np.array([np.cos(yaw), np.sin(yaw)])
        start = transform.location
        end = start + carla.Location(x=vector[0], y=vector[1], z=0)
        # plot the waypoint
        self.debug.draw_point(start, size=0.15, color=color[0], life_time=99999)
        self.debug.draw_arrow(start, end, thickness=0.15, arrow_size=0.15, color=color[1], life_time=99999)
        self.world.tick()

    def get_junction(self, waypoint):
        """
        todo:

        Get the junction in forward direction.
        :param start_point: The start waypoint of a route. carla.Location
        :return: junction to approach, carla.Junction
        """
        sampling_radius = 1.0
        while True:
            wp_choice = waypoint.next(sampling_radius)
            #   Choose path at intersection
            if len(wp_choice) > 1:
                reached_junction = wp_choice[0].is_junction
                if reached_junction:
                    junction = wp_choice[0].get_junction()
                    break
            else:
                waypoint = wp_choice[0]

        # set spectator on the junction
        location = junction.bounding_box.location
        rotation = start_rotation
        transform = carla.Transform(location, rotation)
        # self.set_spectator(transform, 30)

        # ==================================================
        """
        # test carla.Junction get_waypoints method
        lane_type = wp_choice[0].lane_type
        wp_pair_list = junction.get_waypoints(lane_type)  # not quite understand how this list work
        # visualize wp_pair_list
        for tp in wp_pair_list:
            self.draw_waypoint(tp[0])
            self.draw_waypoint(tp[1])
        """
        # ==================================================

        # print('testing get_junction method')
        return junction

    def get_npc_spawn_point(self):
        """
        Get npc vehicle spawn point in a junction scenario.

        Current method is designed based on Town03 junction.

        todo: suit for random junction
        """

        # plot junction coord frame
        location = junction_center
        rotation = carla.Rotation()  # default rotation is world coordinate frame
        transform = carla.Transform(location, rotation)
        plot_local_coordinate_frame(self.world, transform)  # plot coord frame
        # self.set_spectator(carla.Transform(location, start_rotation))

        # using target waypoint of ego vehicle to get spawn location
        start_waypoint = self.map.get_waypoint(start_location)  # start waypoint of this junction
        left_target_waypoint = generate_target_waypoint(start_waypoint, -1)
        right_target_waypoint = generate_target_waypoint(start_waypoint, 1)
        straight_target_waypoint = generate_target_waypoint(start_waypoint, 0)

        self.draw_waypoint(left_target_waypoint)
        self.draw_waypoint(right_target_waypoint)
        self.draw_waypoint(straight_target_waypoint)

        lane_width = left_target_waypoint.lane_width  # consider each lane width is same, usually 3.5

        # ==================================================
        # get npc vehicle spawn location

        lon_dist = 30  # distance to the junction

        # left side
        location_left = left_target_waypoint.transform.location
        d_x = +1 * left_target_waypoint.lane_width * 3
        d_y = +1 * lon_dist
        location_left_2 = location_left + carla.Location(x=d_x, y=d_y)

        left_start_waypoint = self.map.get_waypoint(location_left_2)
        self.draw_waypoint(left_start_waypoint, color=[red, green])

        # right side
        location_right = right_target_waypoint.transform.location
        d_x = -1 * right_target_waypoint.lane_width * 3
        d_y = -1 * lon_dist
        location_right_2 = location_right + carla.Location(x=d_x, y=d_y)

        right_start_waypoint = self.map.get_waypoint(location_right_2)
        self.draw_waypoint(right_start_waypoint, color=[red, green])

        # straight side
        location_straight = straight_target_waypoint.transform.location
        d_y = +1 * straight_target_waypoint.lane_width
        d_x = -1 * lon_dist
        location_straight_2 = location_straight + carla.Location(x=d_x, y=d_y)

        straight_start_waypoint = self.map.get_waypoint(location_straight_2)
        self.draw_waypoint(straight_start_waypoint, color=[red, green])

        # todo: pack result
        # npc_spawn_transform_dict

        print("npc spawn location is generated.")

    def set_autopilot(self, vehicle, p_collision):
        """
        Set autopilot for specified vehicle

        :param vehicle: target vehicle(npc)
        :param p_collision: probability to avoid collision with ego vehicle
        """
        # set traffic manager for each vehicle
        vehicle.set_autopilot(True)
        # ignore traffic lights
        self.traffic_manager.ignore_lights_percentage(vehicle, 100.0)
        # test to show speed limit
        speed_limit = vehicle.get_speed_limit()
        # set speed (limit)
        percentage = -1.0
        self.traffic_manager.vehicle_percentage_speed_difference(vehicle, percentage)
        # collision detection with ego vehicle
        if self.ego_vehicle:
            # ignore conflict with ego vehicle with a probability
            if np.random.random() <= p_collision:
                collision_detection_flag = False
                print("This vehicle will NOT avoid collision with ego vehicle.")
            else:
                collision_detection_flag = True
                print("This vehicle will avoid collision with ego vehicle.")
            self.traffic_manager.collision_detection(vehicle, self.ego_vehicle, collision_detection_flag)

    def spawn_npc(self, transform, name=None, p=None):
        """
        Spawn a npc vehicle in a certain transform
        :return:
        """
        bp = self.blueprint_library.find('vehicle.tesla.model3')
        if bp.has_attribute('color'):
            color = '255, 0, 0'  # use string to identify a RGB color
            bp.set_attribute('color', color)

        if name:
            bp.set_attribute('role_name', name)  # set actor name
        # spawn npc vehicle
        vehicle = self.world.spawn_actor(bp, transform)  # use spawn method
        self.world.tick()
        print("Number", vehicle.id, "npc vehicle is spawned.")  # actor id number of this vehicle
        # set autopilot for ego vehicle
        if not p:
            p_collision = 0.9
        else:
            p_collision = p
        self.set_autopilot(vehicle, p_collision)

        return vehicle

    def get_time_interval(self):
        """
        Time interval till next vehicle spawning in seconds.
        :return: time_interval, float
        """
        # parameters
        lower_limit = 1.0
        upper_limit = 8.0
        time_interval = random.uniform(lower_limit, upper_limit)

        return time_interval

    def get_time(self):
        """
        Get current time using timestamp(carla.Timestamp)
        :return: current timestamp in seconds.
        """
        worldsnapshot = self.world.get_snapshot()
        timestamp = worldsnapshot.timestamp
        now_time = timestamp.elapsed_seconds

        return now_time

    def distance_exceed(self, vehicle):
        """
        Check if distance to junction origin exceed limits.

        todo: junction center as class attribute
        """
        limit = 65.0
        dist = junction_center.distance(vehicle.get_transform().location)
        if dist >= limit:
            return True
        else:
            return False

    def delete_npc(self):
        """
        Check and delete unused npc vehicle.
        """
        delete_list = []
        for vehicle in self.npc_vehicles:
            if self.distance_exceed(vehicle):
                delete_list.append(vehicle)
                self.npc_vehicles.remove(vehicle)

                for key in self.npc_info:
                    for actor in self.npc_info[key]['actor_list']:
                        if vehicle.id == actor.id:
                            self.npc_info[key]['actor_list'].remove(actor)

        if delete_list:
            # self.client.apply_batch([carla.command.DestroyActor(x) for x in delete_list])
            # print(self.client.apply_batch([carla.command.DestroyActor(x) for x in delete_list]))

            # carla.Client.apply_batch_sync method will not crash
            response_list = self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in delete_list], True)

            # attribute_error = response_list[0].error  # attribute error is not working
            method_error = response_list[0].has_error()
            if not method_error:
                print('npc vehicle', delete_list[0].id, "is destroyed")

            # for x in delete_list:
            #     carla.command.DestroyActor(x)

            # check if actor is destroyed successfully
            # actor_list = self.world.get_actors()
            # vehicle_list = self.world.get_actors().filter('vehicle.*')

        # print("vehicle", vehicle.id, "is destroyed")
        # delete_list.clear()

        # print('d')

        # ==================================================
        # original method
        # todo: memory leak for unknown reason
        """
        delete_list = []
        for vehicle in self.npc_vehicles:
            if self.distance_exceed(vehicle):
                delete_list.append(vehicle.id)
                self.npc_vehicles.remove(vehicle)

        if delete_list:
            # delete vehicle actor
            self.client.apply_batch([carla.command.DestroyActor(x) for x in delete_list])
            for x in delete_list:
                print("vehicle", x, "is deleted")

            # remove vehicle from all npc list
            for actor_id in delete_list:
                # npc_vehicles
                for actor in self.npc_vehicles:
                    if actor_id == actor.id:
                        self.npc_vehicles.remove(actor)

                for key in self.npc_info:
                    for actor in self.npc_info[key]['actor_list']:
                        if actor_id == actor.id:
                            self.npc_info[key]['actor_list'].remove(actor)
                print("vehicle", actor_id, "is deleted")
        
        delete_list.clear()        
        """
        # ==================================================

    def spawn_all_traffic_flow(self):
        """
        Spawn all traffic flow in this junction.
        3 tf if crossroad, 2 tf if T-road

        """
        for key in self.npc_info:

            transform = self.npc_info[key]['transform']
            transform.location.z = 1.5

            # set initial spawn time
            if not self.npc_info[key]['target_spawn_time']:
                self.npc_info[key]['target_spawn_time'] = 0

            # todo: add API to get a specified time interval for different tf
            # lower_limit = 1.0
            # upper_limit = 5.0
            # time_interval = random.uniform(lower_limit, upper_limit)
            # self.target_spawn_time = self.last_spawn_time + time_interval

            # check spawn condition of spawning vehicle
            # distance rule
            if self.npc_info[key]['actor_list']:
                # last_spawned_vehicle =self.straight_npc_list[-1]
                distance = transform.location.distance(self.npc_info[key]['actor_list'][-1].get_transform().location)
                distance_threshold = 5.0
                distance_rule = distance >= distance_threshold
                if distance_rule:
                    pass
            else:
                distance_rule = True

            # check if time rule satisfied
            now_time = self.get_time()
            if now_time >= self.npc_info[key]['target_spawn_time']:
                time_rule = True
            else:
                time_rule = False

            if distance_rule and time_rule:
                name = key + str(self.npc_info[key]['count'] + 1)  # name of the actor to be spawned
                try:
                    vehicle = self.spawn_npc(transform, name)
                    self.npc_vehicles.append(vehicle)  # add to npc vehicles list
                    self.npc_info[key]['actor_list'].append(vehicle)  # only store id
                    self.npc_info[key]['count'] += 1
                    self.count += 1
                    # update last spawn time when new vehicle spawned
                    self.npc_info[key]['last_spawn_time'] = self.get_time()
                    # time interval till spawn next vehicle
                    self.npc_info[key]['target_spawn_time'] = self.npc_info[key]['last_spawn_time'] \
                                                              + self.get_time_interval()
                    # return vehicle
                except:
                    print("fail to spawn a npc vehicle, please check.")
                    # return None

            # print('testing spawn traffic flow')

    def __call__(self):
        """"""
        self.run_step()

    def run_step(self):
        """"""
        # check and spawn new npc vehicles
        self.spawn_all_traffic_flow()
        # check and delete npc vehicles
        self.delete_npc()

        # print("")



