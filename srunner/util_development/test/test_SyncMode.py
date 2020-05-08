"""
Some test for world sync mode.

Sync mode, carla world will update after world.tick()
Async mode (Sync mode=False), world will updated automatically

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
import psutil

from srunner.lyq_code.env.BasicEnv import BasicEnv


class TestSyncMode(BasicEnv):
    """
    Generating a continuous traffic flow.
    """
    def __init__(self, town='Town03', host='localhost', port=2000, client_timeout=3.0):
        super(TestSyncMode, self).__init__(town=town, host=host, port=port, client_timeout=client_timeout)
        self.set_world(sync_mode=True, frame_rate=25.0, no_render_mode=False)
        self.world.tick()
        mode = self.world.get_settings().synchronous_mode
        print('sync mode: ', mode)

    def run(self):
        """"""

        # not available in Async mode
        world_snapshot = self.world.wait_for_tick()

        # will return a world snapshot
        self.world.on_tick(lambda world_snapshot: print(world_snapshot.frame))

        self.world.tick()
        print('d')


def main():
    try:
        test = TestSyncMode()

        while True:
            test.run()

    except:
        traceback.print_exc()
    finally:
        del test

    print(datetime.datetime.fromtimestamp(psutil.boot_time()))

if __name__ == '__main__':
    main()

