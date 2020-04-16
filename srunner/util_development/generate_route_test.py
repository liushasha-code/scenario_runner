"""
    Test generate route at junction.

    All methods using waypoint API, are necessary to test with carla server.

"""

# import carla
# carla 095

import glob
import os
import sys

# python module
sys.path.append("/home/lyq/CARLA_simulator/CARLA_095/PythonAPI/carla")
sys.path.append("/home/lyq/CARLA_simulator/CARLA_095/PythonAPI/carla/agents")
# compiled carla egg
carla_path = '/home/lyq/CARLA_simulator/CARLA_095/PythonAPI'
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
import weakref

from srunner.tools.scenario_helper import *
from srunner.challenge.utils.route_manipulation import interpolate_trajectory
from agents.navigation.local_planner import LocalPlanner

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

#
from srunner.tools.scenario_helper import generate_target_waypoint
from srunner.challenge.utils.route_manipulation import interpolate_trajectory

# from srunner.util_development.util import generate_route


# carla color for different object
Red = carla.Color(r=255, g=0, b=0)
Green = carla.Color(r=0, g=255, b=0)
Blue = carla.Color(r=255, g=0, b=0)
Yellow = carla.Color(r=255, g=255, b=0)
Magenta = carla.Color(r=255, g=0, b=255)

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


class test_scenario_generation():
    """
        A easy test tool to check if scenario is correctly generated.
    """

    # start waypoint
    start_coord = np.array([53.0, 128.0, 1.0])



    def __init__(self, host='localhost', port=2000, client_timeout=2.0):
        # setup client
        self.client = carla.Client(host, port)
        self.client.set_timeout(client_timeout)
        self.world = self.client.get_world()
        # world settings
        self.setup_world(town_name='Town03')
        self.debug = self.world.debug

        # set start waypoint
        self.start_waypoint = self.set_start_waypoint()

        #
        self.ego_vehicle = None
        self.spectator = None

        #
        self.route = None


    def setup_world(self, town_name='Town03'):
        """
            Setup carla world with certain parameters.
        :return:
        """
        # get map
        self.map = self.world.get_map()
        # map verification
        if self.map != town_name:
            self.world = self.client.load_world(town_name)
            self.world.tick()

        # world parameters
        frame_rate = 20.0
        # set worldsettings
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / frame_rate
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

        # set weather
        # use ClearNoon
        weather = carla.WeatherParameters.ClearNoon
        self.world.set_weather(weather)

    def set_start_waypoint(self):
        """
            Set start waypoint.
        """
        start_location = carla.Location(x=self.start_coord[0], y=self.start_coord[1], z=self.start_coord[2])
        start_waypoint = self.map.get_waypoint(start_location)

        return start_waypoint

    def set_spectator(self, transform, view_mode=0):
        """
            Set spectator at different viewpoint.

            param waypoint: a carla waypoint class

        """
        # transform = waypoint.transform
        location = transform.location
        rotation = transform.rotation

        if view_mode == 0:
            print("Spectator is set to behind view.")
            # behind distance - d, height - h
            d = 8
            h = 8
            angle = transform.rotation.yaw
            a = math.radians(180 + angle)
            location = carla.Location(x=d * math.cos(a), y=d * math.sin(a), z=h) + transform.location
            rotation = carla.Rotation(yaw=angle, pitch=-15)
            print("done")
        elif view_mode == 1:
            print("Spectator is set to overhead view.")
            h = 100
            h = 30
            location = carla.Location(0, 0, h)+transform.location
            rotation = carla.Rotation(yaw=rotation.yaw, pitch=-90)  # rotate to forward direction
        elif view_mode == 2:
            print("Spectator is set to side behind view.")
            # todo: finish this view

        elif view_mode == 3:

            print("Spectator is set to side behind view.")
            scalar = 5
            offset_vector = np.array([-1, -2, 1])  # relative location of spectator
            offset_vector = scalar*offset_vector/np.linalg.norm(offset_vector)
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

        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(location, rotation))
        self.world.tick()
        print("")

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
        self.debug.draw_arrow(start, end, thickness=0.25, arrow_size=0.25, color=Red, life_time=9999)
        self.debug.draw_point(start, size=0.05, color=Green, life_time=9999)
        self.world.tick()

    def draw_waypoint2(self, waypoint, color_list=[Magenta, Yellow], size=0.2, life_time=9999):
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

    def plot_route(self, trajectory):
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
            self.debug.draw_arrow(start, end, thickness=0.25, arrow_size=0.25, color=Red, life_time=9999)
            self.debug.draw_point(start, size=0.05, color=Green, life_time=9999)
            self.world.tick()

    def set_vehicle(self):
        """
        Get a vehicle to test.
        :return:
        """
        blueprint_library = self.world.get_blueprint_library()
        bp = random.choice(blueprint_library.filter('vehicle.lincoln.mkz2017'))
        bp.set_attribute('role_name', 'hero')
        # modify transform
        # location = carla.Location()
        location = self.start_waypoint.transform.location
        original_rotation = self.start_waypoint.transform.rotation

        yaw = -15
        rotation = carla.Rotation(yaw=yaw)
        transform = carla.Transform(location, rotation)

        self.ego_vehicle = self.world.try_spawn_actor(bp, transform)
        time.sleep(1.0)
        trans = self.ego_vehicle.get_transform()
        print(trans)

        for i in range(1):
            self.world.tick()

        settings = self.world.get_settings()

        trans = self.ego_vehicle.get_transform()
        print(trans)
        for i in range(1):
            self.world.tick()

        trans = self.ego_vehicle.get_transform()
        print(trans)

        # ego vehicle frame
        plot_local_coordinate_frame(self.world, transform, color_scheme=0)
        # local waypoint frame
        plot_local_coordinate_frame(self.world, self.start_waypoint.transform, color_scheme=1)

        # set initial control to keep vehicle static
        control = self.ego_vehicle.get_control()
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        self.ego_vehicle.apply_control(control)


def main():
    try:
        # test
        # scenario_para_dict = {
        #     'map': "Town03",
        #     'start_point': np.array([53.0, 128.0, 1.0]),
        #     # end_point:
        # }
        # print("d")

        # test rotation mrthod
        # rotation_test = carla.Rotation(yaw=0, roll=0, pitch=0)
        # vector = rotation_test.get_forward_vector()
        # print("d")

        # ==================================================
        # test method in class
        test = test_scenario_generation()
        # test.set_spectator(test.start_waypoint.transform, view_mode=0)
        # test.draw_waypoint(test.start_waypoint)

        test.set_spectator(test.start_waypoint.transform)


        # set vehicle
        test.set_vehicle()
        ego_transform = test.ego_vehicle.get_transform()

        # test view direction
        ego_rotation = ego_transform.rotation
        # view_point = np.array([-1, -1, 0])
        view_point = np.array([-1, -1, np.sqrt(2)])
        view_direction = -view_point
        azimuth = get_azimuth(view_direction)
        yaw = azimuth['yaw'] + ego_rotation.yaw
        azimuth['yaw'] = yaw
        rotation = carla.Rotation(yaw=azimuth['yaw'], roll=azimuth['roll'], pitch=azimuth['pitch'])

        # test.set_spectator(ego_transform, view_mode=0)



        # view point vector in global frame
        c_yaw = np.cos(np.radians(ego_rotation.yaw))
        s_yaw = np.sin(np.radians(ego_rotation.yaw))

        Trans_Matrix = np.array([[c_yaw, s_yaw, 0],
                                [-s_yaw, c_yaw, 0],
                                [0, 0, 1]])
        # view point vector in global frame
        view_point_global = np.matmul(np.transpose(Trans_Matrix), view_point)
        view_point_global = carla.Location(x=view_point_global[0], y=view_point_global[1], z=view_point_global[2])

        star_location = ego_transform.location
        star_location.z = 1
        end_location = star_location + view_point_global
        # draw view point
        test.world.debug.draw_arrow(star_location, end_location, color=Magenta, life_time=9999)
        test.world.tick()

        print("d")



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





        # test view direction
        ego_rotation = carla.Rotation(yaw=50, roll=0, pitch=0)
        view_point = np.array([-1, -1, 0])
        view_direction = -view_point
        azimuth = get_azimuth(view_direction)
        yaw = azimuth['yaw'] + ego_rotation.yaw
        azimuth['yaw'] = yaw
        rotation = carla.Rotation(yaw=azimuth['yaw'], roll=azimuth['roll'], pitch=azimuth['pitch'])

        print("d")

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

        print("done")


    except:
        traceback.print_exc()
    finally:
        del test

if __name__ == '__main__':
    main()








