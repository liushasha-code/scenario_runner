"""
Create a test env to test TrafficFlow module.
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

start_location = carla.Location(x=53.0, y=128.0, z=3.0)
start_rotation = carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)  # standard transform of this junction
start_transform = carla.Transform(start_location, start_rotation)
# junction center
junction_center = carla.Location(x=-1.32, y=132.69, z=0.00)


class TestTrafficFlow(BasicEnv):

    def __init__(self, town='Town03', host='localhost', port=2000, client_timeout=2.0):
        super(TestTrafficFlow, self).__init__(town=town, host=host, port=port, client_timeout=client_timeout)
        self.set_world(sync_mode=True, frame_rate=25.0, no_render_mode=False)
        self.world.tick()
        print('sync mode: ', self.world.get_settings().synchronous_mode)

        # instantiate TrafficFlow module
        self.trafficflow = TrafficFlow(self.client, self.world)

        # actor in the test
        self.ego_vehicle = None  # consider only 1 ego vehicle
        self.npc_vehicles = []  # using list to manage npc vehicle

        # route of ego vehicle
        # self.route = None

        # parameters for traffic flow
        self.count = 0  # count spawned vehicle amount
        self.straight_npc_list = []
        self.last_spawn_time = 0
        self.target_spawn_time = None

    def run(self):

        # set spectator on the junction
        self.set_spectator(carla.Transform(junction_center, start_rotation), view=1, h=100)

        junction = self.trafficflow.get_junction_easy()

        count = 0
        while True:

            # call to run step
            self.trafficflow()

            # check status
            npc_list = self.trafficflow.get_npc_list()

            # print npc amount when new npc spawned
            if not count == self.trafficflow.count:
                print('current npc number is: ', self.trafficflow.count)
                count = self.trafficflow.count

            # get near npc
            near_npc_dict = self.trafficflow.get_near_npc()

            # check if getting near npc correctly
            for key in near_npc_dict:
                # if near_npc_dict[key]:

                if len(near_npc_dict[key]) >= 1:
                    print(key, 'near npc', len(near_npc_dict[key]))

            # keep env running
            self.world.tick()


def main():
    try:
        # create a env by instantiation
        test = TestTrafficFlow()
        test.run()

    except:
        traceback.print_exc()
    finally:
        del test


if __name__ == '__main__':
    main()
