"""
test:
1. methods of carla.Junction usage
2. other self-defined methods

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
# from srunner.util_development.scenario_helper_modified import *
from srunner.util_development.scenario_helper_modified import (choose_at_junction)
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


class TestJunctionUsage(BasicEnv):
    """
    Test usage of carla.Junction
    """
    def __init__(self, town='Town03', host='localhost', port=2000, client_timeout=2.0):
        super(TestJunctionUsage, self).__init__(town=town, host=host, port=port, client_timeout=client_timeout)
        self.set_world(sync_mode=False, frame_rate=25.0, no_render_mode=False)
        self.world.tick()
        print('sync mode: ', self.world.get_settings().synchronous_mode)

        # get a startwaypoint
        self.start_waypoint = self.map.get_waypoint(start_location)
        self.junction = None

    @staticmethod
    def get_front_junction(waypoint):
        """
        Get the junction in forward direction.
        :param waypoint: The start waypoint of a route. carla.Location
        :return: carla.Junction ahead current waypoint
        """
        sampling_radius = 1.0
        while True:
            wp_choice = waypoint.next(sampling_radius)
            #   Choose path at intersection
            if len(wp_choice) > 1:
                if wp_choice[0].is_junction:
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

    def get_waypoint_list(self):
        """"""

        # get waypoint list across the junction
        plan = self.generate_target_waypoint_list(self.start_waypoint, turn=1)

        # plot local bbox
        bbox = self.junction.bounding_box

        # junction_transform = carla.Transform()  # an empty transform will work

        # location = junction_center
        location = bbox.location
        extent = bbox.extent

        # rotation = carla.Rotation()
        rotation = carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)  # for debug

        self.debug.draw_box(bbox, rotation, 0.5, red, 999)  # draw junction in red

        print('junction bbox is drawed.')

        for waypoint in plan:
            is_junction = waypoint.is_junction

            # use carla API
            # in_junciton = bbox.contains(waypoint.transform.location, carla.Transform(carla.Location(), rotation))
            # self-defined method
            in_junciton = self.junction_contains(waypoint.transform.location, bbox)

            if is_junction and not in_junciton:
                print('error')

            if in_junciton:
                self.draw_waypoint(waypoint.transform, [yellow, yan])
            else:
                self.draw_waypoint(waypoint.transform)

        print('d')

    @staticmethod
    def junction_contains(location, bbox):
        """
        Check if a location is contained in a junction bounding box.

        todo: consider a rotated junction, with a yaw angle

        :param location: location point to check
        :param bbox: junction bounding box(carla.BoundingBox)
        :return: bool
        """

        vector = location - bbox.location  # location relative to junction center
        extent = bbox.extent

        if extent.x >= vector.x >= -extent.x and \
                extent.y >= vector.y >= -extent.y:
            return True
        else:
            return False

    def run(self):

        # set spectator on the junction
        self.set_spectator(carla.Transform(junction_center, start_rotation), view=1, h=100)

        self.junction = self.get_front_junction(self.start_waypoint)

        self.get_waypoint_list()

        # get

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
        self.debug.draw_point(start, size=0.1, color=color[0], life_time=99999)
        self.debug.draw_arrow(start, end, thickness=0.1, arrow_size=0.1, color=color[1], life_time=99999)
        self.world.tick()

    @staticmethod
    def generate_target_waypoint_list(waypoint, turn=0):
        """
        This method follow waypoints to a junction and choose path based on turn input.
        Turn input: LEFT -> -1, RIGHT -> 1, STRAIGHT -> 0
        @returns a waypoint list from the starting point to the end point according to turn input
        """
        reached_junction = False
        threshold = math.radians(0.1)
        sampling_radius = 1
        plan = []
        while True:
            wp_choice = waypoint.next(sampling_radius)
            if len(wp_choice) > 1:
                reached_junction = True
                waypoint = choose_at_junction(waypoint, wp_choice, turn)
            else:
                waypoint = wp_choice[0]
            plan.append(waypoint)
            #   End condition for the behavior
            if turn != 0 and reached_junction and len(plan) >= 3:  # turning situation
                v_1 = vector(
                    plan[-2].transform.location,
                    plan[-1].transform.location)
                v_2 = vector(
                    plan[-3].transform.location,
                    plan[-2].transform.location)
                angle_wp = math.acos(
                    np.dot(v_1, v_2) / abs((np.linalg.norm(v_1) * np.linalg.norm(v_2))))
                if angle_wp < threshold:
                    break
            elif reached_junction and not plan[-1].is_intersection:  # go straight situation
                break

        return plan


def main():
    try:
        # create a env by instantiation
        test = TestJunctionUsage()
        test.run()

        print('d')

    except:
        traceback.print_exc()
    finally:
        del test


if __name__ == '__main__':
    main()

