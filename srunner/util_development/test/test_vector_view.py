"""
Test set spectator by a vector in carla 098.

todo: unfinished yet

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

# carla module
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

# scenario_runner module
from srunner.tools.scenario_helper import generate_target_waypoint
from srunner.challenge.utils.route_manipulation import interpolate_trajectory

# self defined module
# from srunner.util_development.util import generate_route
from srunner.util_development.carla_rgb_color import *
from srunner.util_development.util_visualization import *

# def generate_route(world, start_waypoint, turn_flag=0, hop_resolution=1.0):
#     """
#         Generate a local route for vehicle to next intersection
#         todo: check if turn flag is unsuitable in next turn
#     :param vehicle: Vehicle need route
#     :param turn_flag: The flag which indicates turn debavior
#     :param hop_resolution: Distance between each waypoint
#     :return: gps and coordinate route  generated
#     """
#     # Using generate_target_waypoint to generate target waypoint
#     # ref on scenario_helper.py module
#     # turn_flag = 0  # turn_flag by current scenario
#     end_waypoint = generate_target_waypoint(start_waypoint, turn_flag)
#
#     # generate a dense route according to current scenario
#     # Setting up global router
#     waypoints = [start_waypoint.transform.location, end_waypoint.transform.location]
#     # from srunner.challenge.utils.route_manipulation import interpolate_trajectory
#     gps_route, trajectory = interpolate_trajectory(world, waypoints, hop_resolution)
#     return gps_route, trajectory

def get_azimuth(view_vector):
    """
        Get current azimuth from direction vector.
        Using only yaw and pitch angle.
    :param view_vector: vector of view direction in vehicle coord frame
    :return: azimuth in degrees, carla.Rotation
    """

    # transform matrix from World coord to ego vehicle
    # only yaw rotation is considered
    # np.cos(np.radians())
    # M1 = np.array([np.cos])

    # rotation around y axis
    x = view_vector[0]
    y = view_vector[1]
    z = view_vector[2]
    yaw = np.rad2deg(np.arctan2(y, x))
    pitch = np.rad2deg(np.arctan2(z, np.sqrt(x*x+y*y)))

    roll = 0  # always not using roll
    azimuth = {"pitch": pitch, "yaw": yaw, "roll": roll}
    # rotation = carla.Rotation(yaw=yaw, roll=0, pitch=pitch)
    return azimuth

def get_rotation_matrix_2D(transform):
    """
        Get a 2D transform matrix of a specified transform
        from actor reference frame to map coordinate frame
    :param transform: actor transform, actually only use yaw angle
    :return: rotation matrix
    """
    yaw = np.deg2rad(transform.rotation.yaw)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    rotation_matrix_2D = np.array([[cy, -sy],
                                   [sy, cy]])
    return rotation_matrix_2D

def plot_local_coordinate_frame(world, origin_transform, axis_length_scale = 3, life_time=99999, color_scheme=0):
    """
        Plot local coordinate frame.
        Using initial vehicle transform as origin
        todo: add text to identify axis, set x-axis always along longitudinal direction
    :param origin_transform: origin of local frame, in class transform
    :param axis_length_scale: length scale for axis vector
    :return: none, plot vectors of 3 axis in server world
    """

    # for test
    # origin_transform = transform
    # axis_length_scale = 3

    # longitudinal direction(x-axis)
    yaw = np.deg2rad(origin_transform.rotation.yaw)
    # x axis
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # get coords in array and uplift with a height
    h = 3
    Origin_coord = np.array(
        [origin_transform.location.x, origin_transform.location.y, origin_transform.location.z + h])
    # elevate z coordinate
    Origin_location = carla.Location(Origin_coord[0], Origin_coord[1], Origin_coord[2])
    # x axis destination
    x_des_coord = Origin_coord + axis_length_scale * np.array([cy, sy, 0])
    x_des = carla.Location(x_des_coord[0], x_des_coord[1], x_des_coord[2])
    # y axis destination
    y_des_coord = Origin_coord + axis_length_scale * np.array([-sy, cy, 0])
    y_des = carla.Location(y_des_coord[0], y_des_coord[1], y_des_coord[2])
    # z axis destination
    z_des_coord = Origin_coord + axis_length_scale * np.array([0, 0, 1])
    z_des = carla.Location(z_des_coord[0], z_des_coord[1], z_des_coord[2])

    """
        color for each axis, carla.Color
        x-axis red:     (255, 0, 0)
        y-axis green:   (0, 255, 0)
        z-axis blue:    (0, 0, 255)
    """
    if color_scheme == 0:
        x_axis_color = carla.Color(r=255, g=0, b=0)
        y_axis_color = carla.Color(r=0, g=255, b=0)
        z_axis_color = carla.Color(r=0, g=0, b=255)
    elif color_scheme == 1:
        x_axis_color = carla.Color(r=252, g=157, b=154)
        y_axis_color = carla.Color(r=131, g=175, b=155)
        z_axis_color = carla.Color(r=96, g=143, b=159)
    elif color_scheme == 2:
        x_axis_color = carla.Color(r=0, g=255, b=255)
        y_axis_color = carla.Color(r=255, g=162, b=0)
        z_axis_color = carla.Color(r=255, g=255, b=255)

    # axis feature
    # thickness = 0.1f
    # arrow_size = 0.1f

    # draw x axis
    world.debug.draw_arrow(Origin_location, x_des, color=x_axis_color, life_time=life_time)
    # draw y axis
    world.debug.draw_arrow(Origin_location, y_des, color=y_axis_color, life_time=life_time)
    # draw z axis
    world.debug.draw_arrow(Origin_location, z_des, color=z_axis_color, life_time=life_time)

    # draw axis text next to arrow
    offset = 0.5
    x_text = carla.Location(x_des_coord[0]+offset, x_des_coord[1]+offset, x_des_coord[2]+offset)
    y_text = carla.Location(y_des_coord[0]+offset, y_des_coord[1]+offset, y_des_coord[2]+offset)
    z_text = carla.Location(z_des_coord[0]+offset, z_des_coord[1]+offset, z_des_coord[2]+offset)
    world.debug.draw_string(x_text, text='x', color=x_axis_color)
    world.debug.draw_string(y_text, text='y', color=x_axis_color)
    world.debug.draw_string(z_text, text='z', color=x_axis_color)

class test_and_visualization():
    """
    A easy test tool to check if scenario is correctly generated.

    """
    # start waypoint
    start_coord = np.array([53.0, 128.0, 3.0])
    # npc start location
    npc_start_coord = np.array([6.0, 171.0, 1.0])

    def __init__(self, host='localhost', port=2000, client_timeout=2.0):
        # setup client
        self.client = carla.Client(host, port)
        self.client.set_timeout(client_timeout)
        self.world = self.client.get_world()
        # world settings
        self.setup_world(town_name='Town03')

        # necessary module
        self.debug = self.world.debug  # world debug for plot
        self.blueprint_library = self.world.get_blueprint_library()  # blueprint
        self.spectator = None

        # actor in the test
        self.ego_vehicle = None
        self.npc_vehicle = None

        # route of ego vehicle
        self.route = None

        # API: exact_location, waypoint, road_center_location
        _, self.start_waypoint, _ = self.coords_trans(self.start_coord)

        # todo: check carla version before using
        self.traffic_manager = self.client.get_trafficmanager()
        print("d")


    def setup_world(self, town_name='Town03'):
        """
            Setup carla world with certain parameters.
        :return:
        """
        # load world with specified map
        self.world = self.client.load_world(town_name)
        self.map = self.world.get_map()

        # # check and get correct map
        # self.map = self.world.get_map()
        # # map verification
        # if self.map != town_name:
        #     self.world = self.client.load_world(town_name)
        #     self.world.tick()

        # world settings
        frame_rate = 20.0
        # set worldsettings
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / frame_rate
        settings.synchronous_mode = False
        # settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.world.tick()

        # ==================================================
        """
        Test result of world.wait_for_tick in carla098:
        wait_for_tick() should not be used under synchronous_mode,
        settings.synchronous_mode = True
        """
        # self.world.wait_for_tick()
        # ==================================================

        # set weather
        weather = carla.WeatherParameters.ClearNoon  # use ClearNoon if not specified
        self.world.set_weather(weather)
        self.world.tick()

    def coords_trans(self, coords, up_lift=True):
        """
        Transform a coords to carla location and waypoint.
        :param coords: list or ndarray, store the coords of a certain point in map, z coord is not necessary.
        :return: location, waypoint
        """
        if len(coords) < 3:
            coords = np.append(coords, [0])
            pass

        exact_location = carla.Location(x=coords[0], y=coords[1], z=coords[2])
        waypoint = self.map.get_waypoint(exact_location)  # carla.waypoint
        road_center_location = waypoint.transform.location

        if up_lift == True:
            exact_location.z += 1.0
            road_center_location.z += 1.0

        return exact_location, waypoint, road_center_location

    @staticmethod
    def plot_frame(vehicle):
        """
        Plot coordinate frame of a vehicle.
        Using method: plot_local_coordinate_frame

        """
        world = vehicle.get_world()
        map = world.get_map()

        # local waypoint frame
        transform = vehicle.get_transform()
        plot_local_coordinate_frame(world, transform, color_scheme=1)

        # plot local coordinate frame(lcf) of road
        location = transform.location
        lcf_waypoint = map.get_waypoint(location, project_to_road=False)
        lcf_location = location
        lcf_rotation = lcf_waypoint.transform.rotation
        lcf_transform = carla.Transform(lcf_location, lcf_rotation)
        plot_local_coordinate_frame(world, lcf_transform, color_scheme=0)
        # print("done")


    def set_ego_vehicle(self):
        """
        Create ego vehicle.

        todo: Package visulization.
        """
        # mkz as ego vehicle
        bp = self.blueprint_library.find('vehicle.lincoln.mkz2017')
        bp.set_attribute('role_name', 'hero')  # set actor name

        # modify transform to aviod spawn error
        # location = carla.Location()
        location = self.start_waypoint.transform.location
        location.z += 1.0

        # rotation = self.start_waypoint.transform.rotation  # original rotation
        # modified rotation for test
        yaw = -15
        rotation = carla.Rotation(yaw=yaw)
        transform = carla.Transform(location, rotation)

        self.set_spectator(transform, view=0)  #

        # spawn actor
        # self.ego_vehicle = self.world.try_spawn_actor(bp, transform)  # using try spawn
        self.ego_vehicle = self.world.spawn_actor(bp, transform)
        # time.sleep(1.0)  # not necessary in carla 098
        self.world.tick()

        # # ==================================================
        # # visualization the local coordinate frame
        # # ego vehicle frame
        # plot_local_coordinate_frame(self.world, transform, color_scheme=0)
        # # local waypoint frame
        # plot_local_coordinate_frame(self.world, self.start_waypoint.transform, color_scheme=1)
        # # ==================================================

        # package into controller
        # set initial control of ego vehicle
        control = self.ego_vehicle.get_control()
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        self.ego_vehicle.apply_control(control)

    def set_npc_vehicle(self):
        """
        Generate a npc vehicle in current test.
        Test the usage of traffic manager.2

        Caution:
        This method is only available when using carla 098.
        Using traffic manager to control npc vehilce.

        """
        bp = self.blueprint_library.find("vehicle.tesla.model3")  # using model3 as npc vehicle
        bp.set_attribute('role_name', 'npc')

        # get npc location
        # API: exact_location, waypoint, road_center_location
        _, waypoint, road_center_location = self.coords_trans(self.npc_start_coord)
        # get spawn transform
        location = road_center_location
        rotation = waypoint.transform.rotation
        transform = carla.Transform(location, rotation)
        # spawn npc
        self.npc_vehicle = self.world.spawn_actor(bp, transform)
        self.world.tick()

        # self.set_spectator(transform)  # visualization

        print("done")

    def npc_controller(self):
        """
        Using traffic manager to control npc vehicle
        :return:
        """

        # actor = self.ego_vehicle

        # self.traffic_manager.collision_detection(reference_actor = , other_actor, detect_collision)

        # print("done")

        pass

    @staticmethod
    def set_spectator_vector(world, vehicle, vector):
        """
        Set the spectator by a relative vector of vehicle.

        TODO: this method can be generalized to a general method
        calculate azimuth using a orientation vector, transformation without using roll motion.(only yaw an pitch)

        """
        if not vector:
            offset_vector = np.array([-3, -4, 1])  # spectator location respect to vehicle
        else:
            offset_vector = vector

        scalar = 3  # scalar of the ratio
        offset_vector = scalar*offset_vector/np.linalg.norm(offset_vector)

        transform = vehicle.get_transform()

        trans_matrix = get_rotation_matrix_2D(transform)

        # using only raw and pitch rotation
        trans_matrix = np.array([[]])

        offset_vector_global = np.multiply(np.transpose(trans_matrix), offset_vector)  # spectator offset vector in global coord system
        # relative location to selected waypoint in global coord frame
        spectator_location = carla.Location(x=offset_vector_global[0], y=offset_vector_global[1], z=offset_vector_global[2])  # transform to carla.Location class
        location = spectator_location + transform.location

        # calculate rotation
        [x, y, z] = -1*[offset_vector_global[0], offset_vector_global[1], offset_vector_global[2]]
        yaw = np.arctan2(y, x)
        pitch = np.arctan2(z, math.sqrt(y*y + x*x))
        rotation = carla.Rotation(yaw=yaw, pitch=pitch)

        print("Spectator is set to side behind view.")

    def set_spectator(self, transform, view=1):
        """
            Set spectator at different viewpoint.

            param waypoint: a carla waypoint class

        """
        # transform = waypoint.transform
        location = transform.location
        rotation = transform.rotation
        if view == 0:
            print("Spectator is set to behind view.")
            # behind distance - d, height - h
            d = 8
            h = 8
            angle = transform.rotation.yaw
            a = math.radians(180 + angle)
            location = carla.Location(x=d * math.cos(a), y=d * math.sin(a), z=h) + transform.location
            rotation = carla.Rotation(yaw=angle, pitch=-15)
            print("done")
        elif view == 1:
            print("Spectator is set to overhead view.")
            # h = 100
            h = 20
            location = carla.Location(0, 0, h)+transform.location
            rotation = carla.Rotation(yaw=rotation.yaw, pitch=-90)  # rotate to forward direction

        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(location, rotation))
        self.world.tick()

    def set_spectator2(self, transform):
        """
        Set spectator at a certain transform.
        :return:
        """
        spectator = self.world.get_spectator()
        spectator.set_transform(transform)
        self.world.tick()

    def draw_waypoint(self, waypoint):
        """
            Draw a waypoint in current world.
            A point at certain location, and an arrow pointing local x axis.

        """
        scalar = 0.5
        yaw = np.deg2rad(waypoint.transform.rotation.yaw)
        vector = scalar*np.array([np.cos(yaw), np.sin(yaw)])
        start = waypoint.transform.location
        end = start + carla.Location(x=vector[0], y=vector[1], z=start.z)
        # plot the waypoint
        self.debug.draw_arrow(start, end, thickness=0.25, arrow_size=0.25, color=red, life_time=9999)
        self.debug.draw_point(start, size=0.05, color=green, life_time=9999)
        self.world.tick()

    def draw_waypoint2(self, waypoint, color_list=[magenta, yellow], size=0.2, life_time=9999):
        """
            Draw waypoint with specified paras

            Using transform as input.

            Draw a point at certain location.
            With an arrow of its transform.

            arrow in its x-axis
        """
        scalar = 0.5
        yaw = np.deg2rad(waypoint.transform.rotation.yaw)
        vector = scalar*np.array([np.cos(yaw), np.sin(yaw)])
        start = waypoint.transform.location
        end = start + carla.Location(x=vector[0], y=vector[1], z=start.z)
        # plot the waypoint
        self.debug.draw_arrow(start, end, thickness=0.25, arrow_size=0.25, color=color_list[1], life_time=life_time)
        self.debug.draw_point(start, size=size, color=color_list[0], life_time=life_time)
        self.world.tick()

    def generate_route(self, start_waypoint, turn_flag=0, hop_resolution=1.0, after_junction_dist=30):
        """
            This method is to generate route for intersection scenario.

        :param start_waypoint: start waypoint defined by scenario intialization
        :param turn_flag: turn flag of intersection turning behavior, 0-straight 1-left, 2-right
        :param hop_resolution: distance between each generated waypoint
        :param after_junction_dist： distance to go after the junction
        :return: generated gps and coord route
        """
        # get waypoint after the intersection
        # ps: from carla.PythonAPI.carla.
        middle_waypoint = generate_target_waypoint(start_waypoint, turn_flag)  # this function is from scenario_helper
        # final waypoint
        end_waypoint = middle_waypoint.next(after_junction_dist)[0]  # default using first waypoint selection if encounter intersection
        # caution: API is carla.Location
        start_location = start_waypoint.transform.location
        middle_location = middle_waypoint.transform.location
        end_location = end_waypoint.transform.location
        waypoints = [start_location, middle_location, end_location]  # API: first and last waypoint of route
        # get total route
        # ps: from srunner.challenge.utils.route_manipulation
        gps_route, trajectory = interpolate_trajectory(self.world, waypoints, hop_resolution)

        # visualization
        self.draw_waypoint2(start_waypoint)
        self.draw_waypoint2(middle_waypoint)
        self.draw_waypoint2(end_waypoint)
        return gps_route, trajectory

    def plot_route(self, trajectory, color=0):
        """
            Plot the complete route.
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
            if color == 0:
                self.debug.draw_arrow(start, end, thickness=0.25, arrow_size=0.25, color=red, life_time=9999)
                self.debug.draw_point(start, size=0.05, color=green, life_time=9999)
            elif color == 1:
                self.debug.draw_arrow(start, end, thickness=0.25, arrow_size=0.25, color=red, life_time=9999)
                self.debug.draw_point(start, size=0.05, color=blue, life_time=9999)
            self.world.tick()


def get_matrix(angle, axis='z'):
    """
    Get transformation matrix through an angle.
    :param angle: angle in degrees
    :param axis: fixed axis of this rotation
    :return: Transformation Matrix, TM
    """
    angle = np.deg2rad(angle)
    [cos_, sin_] = [np.cos(angle), np.sin(angle)]

    if axis == 'z':
        print("rotation is around z axis.")
        print("yaw angle is received.")
        TM = np.array([[cos_, sin_, 0],
                        [-sin_, cos_, 0],
                        [0, 0, 1]])
    elif axis == 'y':
        print("rotation is around y axis.")
        print("pitch angle is received.")
        TM = np.array([[cos_, 0, -sin_],
                        [0, 1, 0],
                        [sin_, 0, cos_]])
    elif axis == 'y':
        print("rotation is around x axis.")
        print("roll angle is received.")
        TM = np.array([[1, 0, 0],
                        [0, cos_, sin_],
                        [0, -sin_, cos_]])
    return TM




def main():
    try:

        """
        # test rotation method
        # rotation.get_forward_vector
        # Usage:
        # rotation_test = carla.Rotation(yaw=0, roll=0, pitch=0)
        # vector = rotation_test.get_forward_vector()
        # print("d")
        """
        # create a env by instantiation
        test = test_and_visualization()

        test.set_ego_vehicle()  # set vehicle

        # test.set_npc_vehicle()  # set npc vehicle

        test.plot_frame(test.ego_vehicle)

        # print("d")




        # test.set_spectator(test.start_waypoint.transform, view_mode=0)
        # test.draw_waypoint(test.start_waypoint)

        # ==================================================






        # ==================================================



        # set spectator
        # test.set_spectator(test.start_waypoint.transform, view=1)

        # test.draw_waypoint2(test.start_waypoint)


        # ==================================================

        # ego_transform = test.ego_vehicle.get_transform()
        #

        # ================================================== # ==================================================

        # transform_2 = waypoint_to_draw.transform
        # original location
        # location = waypoint_to_draw.transform.location
        # modified location
        # location.z = 3

        # original rotation
        # rotation = carla.Rotation(yaw=waypoint_to_draw.transform.rotation.yaw)
        # modified rotation
        # rotation = carla.Rotation(yaw=waypoint_to_draw.transform.rotation.yaw)

        # set transform_2
        # transform_2 = carla.Transform(location=location, rotation=rotation)







        # draw_waypoint(world, waypoint_to_draw)

        # test.set_spectator(world, transform, view=0)



        spectator_local = np.array([-1, -1, np.sqrt(2)])
        view_direction = -spectator_local

        # azimuth of spectator related to ego vehicle frame
        azimuth = get_azimuth(view_direction)  # get azimuth of this view point

        yaw = azimuth['yaw']
        pitch = azimuth['pitch']

        # ego vehicle rotation
        ego_transform = test.ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_location.z = ego_location.z + 3
        ego_rotation = ego_transform.rotation

        # spectator rotation related to world frame
        # yaw
        yaw_spec_world = azimuth['yaw'] + ego_rotation.yaw
        # azimuth['yaw'] = yaw

        rotation = carla.Rotation(yaw=yaw_spec_world, roll=azimuth['roll'], pitch=azimuth['pitch'])

        # get spectator location
        # get transformation matrix
        # args: get_matrix(angle=, axis='z')
        TM_world_vehicle = get_matrix(angle=ego_rotation.yaw, axis='z')
        TM_yaw = get_matrix(angle=azimuth['yaw'], axis='z')
        TM_pitch = get_matrix(angle=azimuth['pitch'], axis='y')

        TM_world_spec = TM_world_vehicle.dot(TM_yaw).dot(TM_pitch)

        offset_world = np.transpose(TM_world_spec).dot(spectator_local)

        spectator_location = carla.Location(x=offset_world[0], y=offset_world[1],
                                            z=offset_world[2])  # spectator offset in world frame
        spectator_location = ego_location + spectator_location

        spectator_rotation = carla.Rotation(yaw=yaw_spec_world, roll=azimuth['roll'], pitch=azimuth['pitch'])  # caution: add yaw angle
        spectator_transform = carla.Transform(spectator_location, spectator_rotation)

        # draw the view direction vector
        direction = spectator_transform.get_forward_vector()
        direction = carla.Location(direction.x, direction.y, direction.z)
        direction_destination = spectator_location + direction*5

        test.world.debug.draw_arrow(spectator_location, direction_destination, color=red, life_time=9999)



        # test.set_spectator2(spectator_transform)

        print("done")








        # # transform matrix - TM
        # c_yaw = np.cos(np.deg2rad(yaw))
        # s_yaw = np.sin(np.deg2rad(yaw))
        # TM_world_vehicle = np.array([[c_yaw, s_yaw, 0],
        #                 [-s_yaw, c_yaw, 0],
        #                 [0, 0, 1]])
        #
        # c_yaw = np.cos(np.deg2rad(yaw))
        # s_yaw = np.sin(np.deg2rad(yaw))
        # TM_yaw = np.array([[c_yaw, s_yaw, 0],
        #                 [-s_yaw, c_yaw, 0],
        #                 [0, 0, 1]])
        #
        # pitch = azimuth['pitch']
        # c_pitch = np.cos(np.deg2rad(pitch))
        # s_pitch = np.sin(np.deg2rad(pitch))
        # TM_pitch = np.array([[c_pitch, 0, -s_pitch],
        #                 [0, 1, 0],
        #                 [s_pitch, 0, c_pitch]])

        # ================================================== # ==================================================

        # # test view direction
        # ego_rotation = ego_transform.rotation
        # # view_point = np.array([-1, -1, 0])
        # view_point = np.array([-1, -1, np.sqrt(2)])
        # view_direction = -view_point
        # azimuth = get_azimuth(view_direction)
        # yaw = azimuth['yaw'] + ego_rotation.yaw
        # azimuth['yaw'] = yaw
        # rotation = carla.Rotation(yaw=azimuth['yaw'], roll=azimuth['roll'], pitch=azimuth['pitch'])
        #
        # # test.set_spectator(ego_transform, view_mode=0)
        #
        #
        #
        # # view point vector in global frame
        # c_yaw = np.cos(np.radians(ego_rotation.yaw))
        # s_yaw = np.sin(np.radians(ego_rotation.yaw))
        #
        # Trans_Matrix = np.array([[c_yaw, s_yaw, 0],
        #                         [-s_yaw, c_yaw, 0],
        #                         [0, 0, 1]])
        # # view point vector in global frame
        # view_point_global = np.matmul(np.transpose(Trans_Matrix), view_point)
        # view_point_global = carla.Location(x=view_point_global[0], y=view_point_global[1], z=view_point_global[2])
        #
        # star_location = ego_transform.location
        # star_location.z = 1
        # end_location = star_location + view_point_global
        # # draw view point
        # test.world.debug.draw_arrow(star_location, end_location, color=magenta, life_time=9999)
        # test.world.tick()
        #
        # print("d")
        # ==================================================


        # ==================================================
        # print("Spectator is set to side behind view.")
        # scalar = 5
        # offset_vector = np.array([-1, -2, 1])  # relative location of spectator
        # offset_vector = scalar * offset_vector / np.linalg.norm(offset_vector)
        # trans_matrix = get_rotation_matrix_2D(transform)
        #
        # # using only raw and pitch rotation
        # trans_matrix = np.array([[]])
        #
        # offset_vector_global = np.multiply(np.transpose(trans_matrix),
        #                                    offset_vector)  # spectator offset vector in global coord system
        # # relative location to selected waypoint in global coord frame
        # spectator_location = carla.Location(x=offset_vector_global[0], y=offset_vector_global[1],
        #                                     z=offset_vector_global[2])  # transform to carla.Location class
        # location = spectator_location + transform.location
        #
        # # calculate rotation
        # [x, y, z] = -1 * [offset_vector_global[0], offset_vector_global[1], offset_vector_global[2]]
        # yaw = np.arctan2(y, x)
        # pitch = np.arctan2(z, math.sqrt(y * y + x * x))
        # rotation = carla.Rotation(yaw=yaw, pitch=pitch)
        # ==================================================



        # ==================================================
        # # test view direction
        # ego_rotation = carla.Rotation(yaw=50, roll=0, pitch=0)
        # view_point = np.array([-1, -1, 0])
        # view_direction = -view_point
        # azimuth = get_azimuth(view_direction)
        # yaw = azimuth['yaw'] + ego_rotation.yaw
        # azimuth['yaw'] = yaw
        # rotation = carla.Rotation(yaw=azimuth['yaw'], roll=azimuth['roll'], pitch=azimuth['pitch'])
        # print("d")
        # ==================================================



        # ==================================================
        # get route
        gps_route, trajectory = test.generate_route(start_waypoint=test.start_waypoint, turn_flag=1)
        # print end point
        end_coord = np.array([trajectory[-1][0].location.x, trajectory[-1][0].location.y, trajectory[-1][0].location.z])
        print("End point is", end_coord)
        # plot route
        test.plot_route(trajectory)
        # a ref end coord
        # np.array([5.24, 92.28, 0])


        # generate npc vehicle route
        start_coord = np.array([6.0, 171.0, 1.0])
        npc_start_loc = carla.Location(x=start_coord[0], y=start_coord[1], z=start_coord[2])
        npc_start_waypoint = test.map.get_waypoint(npc_start_loc)

        # start_waypoint =
        # start_waypoint
        pass_junction_waypoint = generate_target_waypoint(npc_start_waypoint, turn=0)

        test.draw_waypoint2(npc_start_waypoint)
        test.draw_waypoint2(pass_junction_waypoint)

        npc_gps_route, npc_trajectory = test.generate_route(start_waypoint=npc_start_waypoint, turn_flag=0)
        test.plot_route(npc_trajectory, color=1)
        #
        print("done")


    except:
        traceback.print_exc()
    finally:
        del test

if __name__ == '__main__':
    main()
