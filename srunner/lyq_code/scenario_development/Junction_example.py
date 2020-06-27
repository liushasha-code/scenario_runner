"""
This is an example of junction scenario.

Junction right turn with straight npc vehicle as an example.

CARLA version 0.9.9
"""

import glob
import os
import sys

# carla 099 API is recommended
sys.path.append("/home/lyq/CARLA_simulator/CARLA_099/PythonAPI/carla")
sys.path.append("/home/lyq/CARLA_simulator/CARLA_099/PythonAPI/carla/agents")
carla_path = '/home/lyq/CARLA_simulator/CARLA_099/PythonAPI'

try:
    sys.path.append(glob.glob(carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import time
import random
import numpy as np
import math
import traceback

from srunner.lyq_code.env.BasicEnv import BasicEnv


info_dict = {
    'scenario_info': {
        'scenario_classname': 'junction_right_turn',
        'scenario_code': 'A1_1_1'

    },
    'world_settings': {
        'map': 'Town03',
        'ego_info': {
            'model': 'lincoln.mkz2017',
            'start_location': [],

        },
        'npc_info':{
            'model': 'tesla.model3',
            'start_location': [],

        }
    }
}


class ScenarioEnv(BasicEnv):
    max_time = 10000  # seconds

    def __init__(self):
        """
        todo add npc vehicle module
        """
        self.info_dict = {}

    def get_dict(self):
        """
        Get info dict describe the scenario.
        """
        self.info_dict = info_dict

    def spawn_ego(self):
        """
        todo require API design

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

        return vehicle


        self.ego = actor

    def spawn_traffic_flow(self):
        """
        Use local planner to control npc vehicle.

        :return:
        """
        pass

    def status_check(self):
        """
        Check if finish status is reached.
        """
        done = False

        # todo add content here

        return done

    def run(self):
        """
        Run the simulation.
        """
        # ego
        self.spawn_ego()
        # traffic
        self.spawn_traffic_flow()

        # todo set recorder if required



        time_step = 0


        while True:

            # tick ego vehicle
            control = self.ego.run_step()

            # tick traffic flow
            self.traffic_flow.run_step()

            # todo add finish status check
            done = self.status_check()
            if done:
                break

            self.world.tick()
            time_step += 1
            print('timestamp')



        print('Simulation is finished')







if __name__ == '__main__':
    env = ScenarioEnv()
    env.run()

    print('done')
