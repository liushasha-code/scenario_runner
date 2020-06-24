"""
This is an example of junction scenario.
right turn with a straight npc vehicle.
"""

from __future__ import print_function
import glob
import os
import sys

# carla 098 API is recommended
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
import random
import numpy as np
import math
import traceback

from srunner.lyq_code.env.BasicEnv import BasicEnv


class Env(BasicEnv):
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

    def spawn_npc(self):
        """
        Use local planner to control npc vehicle.

        :return:
        """
        pass


    def run(self):
        """"""

        time_step = 0


        while True:

            # ego and npc vehicle control
            control = self.agent()
            self.world.tick()
            time_step += 1
            print('timestamp')

        print('d')







if __name__ == '__main__':
    env = Env()
    env.run()

    print('done')
