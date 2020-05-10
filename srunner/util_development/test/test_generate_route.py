"""
    Test generate route at junction.

    All methods using waypoint API, are necessary to test with carla server.

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
# from srunner.tools.scenario_helper import generate_target_waypoint
from srunner.challenge.utils.route_manipulation import interpolate_trajectory

# self defined module
# from srunner.util_development.util import generate_route
from srunner.util_development.carla_rgb_color import *
from srunner.util_development.util_visualization import *
from srunner.util_development.scenario_helper_modified import generate_target_waypoint
from srunner.util_development.scenario_helper_modified import *


class test_and_visualization():
    """
    A easy test tool to check if scenario is correctly generated.
    """
    # start waypoint
    start_coord = np.array([53.0, 128.0, 3.0])

    """
    # origin of junction frame 
    # carla.transform
    # Location(x=5.421652, y=120.832970, z=0.000000)
    # Rotation(pitch=360.000000, yaw=269.637451, roll=0.000000)
    """
    origin_transform = carla.Transform(carla.Location(x=5.421652, y=120.832970, z=0.000000),
                                       carla.Rotation(pitch=360.000000, yaw=269.637451, roll=0.000000))

    test_transform = carla.Transform(carla.Location(x=5.0, y=140.0, z=0.000000),
                                       carla.Rotation(pitch=360.000000, yaw=269.637451, roll=0.000000))

    # npc start location
    npc_start_coord = np.array([6.0, 171.0, 1.0])

    # test location
    test_coord = np.array([60.0, 128.0, 3.0])

    # collision test location
    collision_test_location = np.array([6.0, 160.0, 3.0])

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
        # self.test_TM_function()

        # ==================================================
        self.rss = None
        self.test_rss()

    def test_rss(self):
        """
        Test RSS sensor usage.
        :return:
        """

        blueprint = self.world.get_blueprint_library().find('sensor.other.rss')

        self.set_ego_vehicle()

        transform = carla.Transform(carla.Location(x=0.8, z=1.7))
        sensor = self.world.spawn_actor(blueprint, transform, attach_to=self.ego_vehicle)

        target_transform = carla.Transform(carla.Location(x=5.421652, y=120.832970, z=0.000000),
                            carla.Rotation(pitch=360.000000, yaw=269.637451, roll=0.000000))

        # listen method

        # sensor.listen(lambda data: )

        # sensor.append_routing_target(routing_target=target_transform)


        print("d")


    def quick_spawn_vehicle(self, transform):
        """
        Quick spawn actor at transform.
        :return:
        """
        bp = self.blueprint_library.find('vehicle.tesla.model3')
        if bp.has_attribute('color'):
            # color = random.choice(bp.get_attribute('color').recommended_values)
            color = '0, 255, 0'
            bp.set_attribute('color', color)

        # reset z height
        transform.location.z = 1.5

        vehicle = self.world.spawn_actor(bp, transform)

        return vehicle

    def set_vehicle(self, coords, name=None):
        """
        Spawn a vehicle in current world.
        :param coords: location to spawn, carla.Location
        :param name: if the vehicle has a name
        :return: the spawned vehicle, carla.vehicle
        """

        bp = self.blueprint_library.find('vehicle.tesla.model3')
        if name is not None:
            bp.set_attribute('role_name', name)  # set actor name
        if bp.has_attribute('color'):
            # color = random.choice(bp.get_attribute('color').recommended_values)
            color = '255, 0, 0'
            bp.set_attribute('color', color)

        # get npc location
        # API: exact_location, waypoint, road_center_location
        _, waypoint, road_center_location = self.coords_trans(coords)
        # get spawn transform
        location = road_center_location
        rotation = waypoint.transform.rotation
        transform = carla.Transform(location, rotation)
        # spawn npc
        vehicle = self.world.spawn_actor(bp, transform)
        self.world.tick()

        return vehicle

    def test_TM_function(self):
        """
        This method is simply a test for TM usage.
        """
        TM = self.traffic_manager
        TM.set_synchronous_mode(True)
        self.world.tick()
        settings = self.world.get_settings()

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
        frame_rate = 50.0
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
        # yaw = -15
        # rotation = carla.Rotation(yaw=yaw)

        rotation = self.start_waypoint.transform.rotation
        transform = carla.Transform(location, rotation)

        self.set_spectator(transform, view=1)  #

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

    def set_autopilot(self, ref_vehicle, other_vehicle):
        """
        Using traffic manager to control npc vehicle
        todo: package this method for migration
        Caution:
        Have to set traffic manager already before using this method.
        :param: target vehicle: major interaction object
        """
        # self.traffic_manager.collision_detection(reference_actor = , other_actor, detect_collision)
        ref_vehicle.set_autopilot(True)
        self.traffic_manager.collision_detection(ref_vehicle, other_vehicle, False)

        # print("done")

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
            h = 50
            # h = 20
            location = carla.Location(0, 0, h)+transform.location
            rotation = carla.Rotation(yaw=rotation.yaw, pitch=-90)  # rotate to forward direction

        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(location, rotation))
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
