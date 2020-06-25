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

from


class ScenarioEnv(BasicEnv):
    max_time = 10000  # seconds

    def __init__(self):
        """
        todo add npc vehicle module
        """
        pass

    def spawn_ego(self):
        """
        todo require API design

        :return:
        """
        pass

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
