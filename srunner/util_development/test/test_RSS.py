"""
Test RSS sensor usage.

"""
from __future__ import print_function

import glob
import os
import sys

# using carla 095
# sys.path.append("/home/lyq/CARLA_simulator/CARLA_095/PythonAPI/carla")
# sys.path.append("/home/lyq/CARLA_simulator/CARLA_095/PythonAPI/carla/agents")
# carla_path = '/home/lyq/CARLA_simulator/CARLA_095/PythonAPI'  # carla egg

# using carla096
# sys.path.append("/home/lyq/CARLA_simulator/CARLA_096/PythonAPI/carla")
# sys.path.append("/home/lyq/CARLA_simulator/CARLA_096/PythonAPI/carla/agents")
# carla_path = '/home/lyq/CARLA_simulator/CARLA_096/PythonAPI'

# using carla098
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

def main():
    try:
        # create a env by instantiation
        test = test_and_visualization()
        test.set_ego_vehicle()  # set ego vehicle
        test.set_npc_vehicle()  # set npc vehicle

        # set spectator and plot origin

        test.quick_spawn_vehicle(test.test_transform)
        test.set_spectator(test.test_transform, view=1)
        print("d")


        # # set spectator
        # test.set_spectator(test.npc_vehicle.get_transform(), view=1)
        # test.set_vehicle(test.test_coord)
        # vehicle_2 = test.set_vehicle(test.collision_test_location)
        # test.set_autopilot(test.npc_vehicle, vehicle_2)
        # print("d")

        middle_waypoint = generate_target_waypoint(test.start_waypoint, 1)
        test.draw_waypoint2(middle_waypoint)
        test.set_spectator(middle_waypoint.transform, view=1)

        # # ==================================================
        # # get route
        gps_route, trajectory = test.generate_route(start_waypoint=test.start_waypoint, turn_flag=1)
        print("d")

        # # print end point
        # end_coord = np.array([trajectory[-1][0].location.x, trajectory[-1][0].location.y, trajectory[-1][0].location.z])
        # print("End point is", end_coord)
        # # plot route
        # test.plot_route(trajectory)
        # # a ref end coord
        # # np.array([5.24, 92.28, 0])
        #
        #
        # # generate npc vehicle route
        # start_coord = np.array([6.0, 171.0, 1.0])
        # npc_start_loc = carla.Location(x=start_coord[0], y=start_coord[1], z=start_coord[2])
        # npc_start_waypoint = test.map.get_waypoint(npc_start_loc)
        #
        # # start_waypoint =
        # # start_waypoint
        # pass_junction_waypoint = generate_target_waypoint(npc_start_waypoint, turn=0)
        #
        # test.draw_waypoint2(npc_start_waypoint)
        # test.draw_waypoint2(pass_junction_waypoint)
        #
        # npc_gps_route, npc_trajectory = test.generate_route(start_waypoint=npc_start_waypoint, turn_flag=0)
        # test.plot_route(npc_trajectory, color=1)
        #
        # print("done")
        # # ==================================================

    except:
        traceback.print_exc()
    finally:
        del test

if __name__ == '__main__':
    main()



