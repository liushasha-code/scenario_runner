"""
This module is to generate, manage and visualize route of a specified vehicle.
Current version is for ego vehicle.

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
from srunner.lyq_code.env.env_module.TrafficFlow import TrafficFlow  # class name same as py script

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


class RouteManager:
    """

    """
    def __init__(self, world):
        """"""
        self.world = world
        self.map = self.world.get_map()


    def generate_route(self, location, turn_flag=0, hop_resolution=1.0):
        """
        Generate a local route from current location to next intersection
        :return:
        """
        location
        end_waypoint = generate_target_waypoint(start_waypoint, turn_flag)
        waypoints = [start_waypoint.transform.location, end_waypoint.transform.location]
        gps_route, trajectory = interpolate_trajectory(world, waypoints, hop_resolution)


    def get_junction(self):
        """"""

        pass

    def plot_route(self):
        """
        Plot current route
        :return:
        """
        for item in trajectory:
            transform = item[0]
            scalar = 0.5
            yaw = np.deg2rad(transform.rotation.yaw)
            vector = scalar * np.array([np.cos(yaw), np.sin(yaw)])
            start = transform.location
            end = start + carla.Location(x=vector[0], y=vector[1], z=start.z)
            # plot the waypoint
            debug = world.debug
            debug.draw_arrow(start, end, thickness=0.25, arrow_size=0.25, color=red, life_time=9999)
            debug.draw_point(start, size=0.05, color=green, life_time=9999)
            world.tick()















